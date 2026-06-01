from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import time
from typing import Union, List, Tuple, Optional

import numpy as np
import onnxruntime as ort
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice, crop_and_pad_nd
from batchgenerators.utilities.file_and_folder_operations import load_json, join, subdirs
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from scipy.ndimage import zoom
from torch import nn

import nnInteractive
from nnInteractive.interaction.point import PointInteraction_stub
from nnInteractive.trainer.nnInteractiveTrainer import nnInteractiveTrainer_stub
from nnInteractive.utils.crop import crop_and_pad_into_buffer, paste_tensor, pad_cropped, crop_to_valid
import torch

# ---------------------------------------------------------------------------
# Interaction channel index constants.
# The model expects 8 input channels: 1 image + 7 interaction channels.
# Channel layout (confirmed from original nnInteractiveInferenceSession):
#   0 / -7 : current segmentation (running prediction + initial seg input)
#   1 / -6 : bbox positive
#   2 / -5 : bbox negative
#   3 / -4 : point positive
#   4 / -3 : point negative
#   5 / -2 : scribble positive
#   6 / -1 : scribble negative
# Lasso shares the bbox channels (same spatial concept).
# ---------------------------------------------------------------------------
_NUM_INTERACTION_CHANNELS = 7
_CHANNEL_CURRENT_SEG  = 0   # also written as -7
_CHANNEL_BBOX_POS     = 1   # also written as -6
_CHANNEL_BBOX_NEG     = 2   # also written as -5
_CHANNEL_POINT_POS    = 3   # also written as -4
_CHANNEL_POINT_NEG    = 4   # also written as -3
_CHANNEL_SCRIBBLE_POS = 5   # also written as -2
_CHANNEL_SCRIBBLE_NEG = 6   # also written as -1


class SimpleONNXConfig:
    """Minimal configuration object standing in for nnUNet's ConfigurationManager."""
    def __init__(self, patch_size=(192, 192, 192)):
        self.patch_size = list(patch_size)


class nnInteractiveInferenceSessionONNX:
    """
    ONNX-based inference session for nnInteractive.

    Reproduces the interface of nnInteractiveInferenceSession but runs
    inference via ONNX Runtime rather than PyTorch, making it usable on
    CPU without a full PyTorch/CUDA installation.

    Deliberately omits the AutoZoom / diff-map / refinement pipeline from
    the original — these are GPU-oriented and too expensive for CPU ONNX
    inference. Instead, two inference modes are available:
        - Single-patch (predict_entire_image=False): one patch per interaction
        - Sliding window (predict_entire_image=True): full image coverage
    """

    def __init__(
        self,
        model_path,
        device: str = "cpu",
        do_autozoom: bool = False,
        verbose: bool = False,
    ):
        self.verbose = verbose
        self.do_autozoom = do_autozoom
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Inference mode — sliding window covers full image, single patch is faster
        self.predict_entire_image = True

        # Interaction settings — set to nnInteractive defaults
        self.preferred_scribble_thickness = [2, 2, 2]
        self.interaction_decay = 0.9
        self.pad_mode_data = "constant"

        self.configuration_manager = SimpleONNXConfig(patch_size=(192, 192, 192))

        # Model — single session, one source of truth
        self.network: Optional[ort.InferenceSession] = None
        self.input_name: Optional[str] = None
        self.output_name: Optional[str] = None

        # Image / interaction state — set by set_image()
        self.preprocessed_image: Optional[np.ndarray] = None
        self.preprocessed_props: Optional[dict] = None
        self.interactions: Optional[np.ndarray] = None
        self.target_buffer: Optional[np.ndarray] = None
        self.input_shape: Optional[Tuple] = None
        self.original_image_shape: Optional[Tuple] = None

        # Interaction tracking
        self.new_interaction_zoom_out_factors: List[float] = []
        self.new_interaction_centers: List = []
        self.has_positive_bbox: bool = False

        self.point_interaction = PointInteraction_stub(4, True)

        # Background preprocessing thread pool
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.preprocess_future = None
        self.interactions_future = None

        # Load model if path provided (None = PyTorch checkpoint path,
        # handled later by initialize_from_trained_model_folder)
        if model_path is not None:
            self.load_network(model_path)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _get_providers(self) -> List[str]:
        """Select the best available ONNX execution provider."""
        available = ort.get_available_providers()
        if self.verbose:
            print(f"[debug] Available ONNX providers: {available}")
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider"]
        elif "CoreMLExecutionProvider" in available:
            return ["CoreMLExecutionProvider"]
        return ["CPUExecutionProvider"]

    def load_network(self, model_path) -> None:
        """
        Create the ONNX InferenceSession and resolve input/output names.
        This is the only place an ort.InferenceSession should be created.
        """
        model_path = Path(model_path)
        assert model_path.exists(), f"Model file not found: {model_path}"
        assert model_path.suffix == ".onnx", (
            f"Expected .onnx file, got: {model_path.suffix}"
        )

        providers = self._get_providers()
        print(f"[debug] Loading ONNX model: {model_path} (providers: {providers})")

        self.network = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.network.get_inputs()[0].name
        self.output_name = self.network.get_outputs()[0].name

        print(f"[debug] input='{self.input_name}', output='{self.output_name}'")
        print(f"[debug] Model input shape: {self.network.get_inputs()[0].shape}")

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def _reset_session(self) -> None:
        """
        Clear all image-specific state. Waits for background tasks to
        finish before clearing to prevent orphaned thread writes.
        """
        for future_attr in ('preprocess_future', 'interactions_future'):
            future = getattr(self, future_attr, None)
            if future is not None:
                try:
                    future.cancel()
                    future.result()
                except Exception:
                    pass
                setattr(self, future_attr, None)

        self.preprocessed_image = None
        self.target_buffer = None
        self.interactions = None
        self.preprocessed_props = None
        self.original_image_shape = None
        self.has_positive_bbox = False
        self.new_interaction_centers.clear()
        self.new_interaction_zoom_out_factors.clear()

    def set_image(self, image: np.ndarray, image_properties: dict = None) -> None:
        """
        Set the image to segment. Must be called before any interaction methods.
        Image must be 4D: (C, X, Y, Z).
        Preprocessing is offloaded to a background thread.
        """
        if image_properties is None:
            image_properties = {}
        self._reset_session()

        if image.ndim == 3:
            print("[debug] 3D input detected — adding channel dimension.")
            image = image[None]

        assert image.ndim == 4, (
            f"Expected 4D image (C, X, Y, Z), got {image.ndim}D shape {image.shape}"
        )

        if self.verbose:
            print(f"set_image: raw shape {image.shape}")

        self.original_image_shape = image.shape
        self.preprocess_future = self.executor.submit(
            self._background_set_image, image, image_properties
        )

    def set_target_buffer(self, target_buffer: Union[np.ndarray, torch.Tensor]) -> None:
        """Must be a 3D numpy array."""
        self.target_buffer = target_buffer

    def set_do_autozoom(self, do_autozoom: bool, max_num_patches: Optional[int] = None) -> None:
        self.do_autozoom = do_autozoom

    def _finish_preprocessing_and_initialize_interactions(self) -> None:
        """
        Block until background preprocessing is complete.
        Call this at the start of every interaction method.
        """
        if self.preprocess_future is not None:
            try:
                self.preprocess_future.result()
            except Exception as e:
                raise RuntimeError(f"Background preprocessing failed: {e}") from e
            self.preprocess_future = None

        if self.preprocessed_image is None:
            raise RuntimeError("preprocessed_image is None. Was set_image() called?")
        if self.preprocessed_props is None:
            raise RuntimeError("preprocessed_props is None. Was set_image() called?")
        if self.interactions is None:
            raise RuntimeError("interactions is None. Was set_image() called?")

    # ------------------------------------------------------------------
    # Preprocessing (background thread)
    # ------------------------------------------------------------------

    def _background_set_image(self, image: np.ndarray, image_properties: dict) -> None:
        """
        Runs in background thread. Crops to nonzero region then normalizes.
        Matches original nnInteractive preprocessing order:
          1. Find nonzero bbox on original image
          2. Crop
          3. Normalize on cropped region
        """
        print("[debug] _background_set_image started")
        assert image.ndim == 4

        # Find nonzero bbox on spatial dims only (skip channel dim).
        spatial = image[0]
        nonzero_idx = np.nonzero(spatial)

        if any(len(idx) == 0 for idx in nonzero_idx):
            raise RuntimeError("Image is entirely zero — cannot crop to nonzero region.")

        # bbox is 3-element: one interval per spatial dim (no channel dim).
        bbox = [[int(idx.min()), int(idx.max()) + 1] for idx in nonzero_idx]

        if self.verbose:
            print(f"Nonzero bbox: {bbox}")

        # Crop spatial dims, preserve channel dim.
        slicer = (slice(None),) + tuple(slice(b[0], b[1]) for b in bbox)
        image_cropped = image[slicer].astype(np.float32)

        if self.verbose:
            print(f"Cropped shape: {image_cropped.shape}")

        # Normalize on cropped region (matches original behaviour).
        mean = image_cropped.mean()
        std = image_cropped.std()
        if std == 0:
            print("[warning] Image std is zero — skipping normalization.")
        else:
            image_cropped = (image_cropped - mean) / std

        self.preprocessed_image = image_cropped
        # bbox[1:] not needed here since bbox is already 3-element (spatial only)
        self.preprocessed_props = {'bbox_used_for_cropping': bbox}

        # Initialize interactions now that we know final spatial shape.
        self.interactions_future = self.executor.submit(
            self._initialize_interactions, image_cropped
        )
        self.interactions_future.result()
        del self.interactions_future
        self.interactions_future = None

        if self.verbose:
            print(f"Preprocessing done. bbox: {bbox}")

    def _initialize_interactions(self, image: np.ndarray) -> None:
        """
        Allocate interactions tensor and target buffer.
        Called from background thread after _background_set_image crops the image.
        """
        assert image.ndim == 4, (
            f"Expected 4D (C, D, H, W), got {image.ndim}D shape {image.shape}"
        )

        _, d, h, w = image.shape
        self.input_shape = (d, h, w)

        # Derive interaction channel count from the model rather than hardcoding.
        if self.network is not None:
            image_channels = image.shape[0]
            model_input_channels = self.network.get_inputs()[0].shape[1]
            num_interaction_channels = model_input_channels - image_channels
            if num_interaction_channels <= 0:
                raise RuntimeError(
                    f"Model expects {model_input_channels} channels but image has "
                    f"{image_channels} — no room for interaction channels."
                )
        else:
            # Fallback for PyTorch checkpoint path — use confirmed value.
            num_interaction_channels = _NUM_INTERACTION_CHANNELS

        self.interactions = np.zeros(
            (num_interaction_channels, d, h, w), dtype=np.float32
        )
        self.target_buffer = np.zeros((d, h, w), dtype=np.uint8)

        if self.verbose:
            print(f"interactions: {self.interactions.shape}")
            print(f"target_buffer: {self.target_buffer.shape}")

    # ------------------------------------------------------------------
    # Interaction reset
    # ------------------------------------------------------------------

    def reset_interactions(self) -> None:
        """Reset all interactions and segmentation output for the current image."""
        if self.interactions is not None:
            self.interactions.fill(0)
        if self.target_buffer is not None:
            self.target_buffer.fill(0)
        self.has_positive_bbox = False

    # ------------------------------------------------------------------
    # Interaction methods
    # ------------------------------------------------------------------

    def add_point_interaction(
        self,
        coordinates: Tuple[int, ...],
        include_interaction: bool,
        run_prediction: bool = True
    ) -> None:
        print("[debug] add_point_interaction()")
        self._finish_preprocessing_and_initialize_interactions()

        assert self.interactions.dtype == np.float32, (
            f"interactions must be float32, got {self.interactions.dtype}"
        )

        transformed = [
            round(i) for i in transform_coordinates_noresampling(
                coordinates, self.preprocessed_props['bbox_used_for_cropping']
            )
        ]
        self._add_patch_for_point_interaction(transformed)

        self.interactions[_CHANNEL_POINT_POS] *= self.interaction_decay
        self.interactions[_CHANNEL_POINT_NEG] *= self.interaction_decay

        channel = _CHANNEL_POINT_POS if include_interaction else _CHANNEL_POINT_NEG
        self.interactions[channel] = self.point_interaction.place_point(
            transformed, self.interactions[channel]
        )

        if run_prediction:
            self._predict()

    def add_bbox_interaction(
        self,
        bbox_coords,
        include_interaction: bool,
        run_prediction: bool = True
    ) -> None:
        print("[debug] add_bbox_interaction()")
        self._finish_preprocessing_and_initialize_interactions()

        if include_interaction:
            self.has_positive_bbox = True

        lbs = [round(i) for i in transform_coordinates_noresampling(
            [i[0] for i in bbox_coords],
            self.preprocessed_props['bbox_used_for_cropping']
        )]
        ubs = [round(i) for i in transform_coordinates_noresampling(
            [i[1] for i in bbox_coords],
            self.preprocessed_props['bbox_used_for_cropping']
        )]
        transformed_bbox = [[lo, hi] for lo, hi in zip(lbs, ubs)]

        # Clip to image boundaries and prevent collapsed dims.
        image_shape = self.preprocessed_image.shape
        for dim in range(len(transformed_bbox)):
            start, end = transformed_bbox[dim]
            start = max(0, start)
            end = min(image_shape[dim + 1], end)
            if end <= start:
                end = min(start + 1, image_shape[dim + 1])
            transformed_bbox[dim] = [start, end]

        self._add_patch_for_bbox_interaction(transformed_bbox)

        self.interactions[_CHANNEL_BBOX_POS] *= self.interaction_decay
        self.interactions[_CHANNEL_BBOX_NEG] *= self.interaction_decay

        slicer = tuple(slice(*i) for i in transformed_bbox)
        channel = _CHANNEL_BBOX_POS if include_interaction else _CHANNEL_BBOX_NEG
        self.interactions[(channel, *slicer)] = 1

        if run_prediction:
            self._predict()

    def add_scribble_interaction(
        self,
        scribble_image: np.ndarray,
        include_interaction: bool,
        run_prediction: bool = True
    ) -> None:
        assert all(
            i == j for i, j in zip(self.original_image_shape[1:], scribble_image.shape)
        ), (
            f"Scribble shape {scribble_image.shape} != "
            f"image shape {self.original_image_shape[1:]}"
        )
        self._finish_preprocessing_and_initialize_interactions()

        scribble_image = crop_and_pad_nd(
            scribble_image, self.preprocessed_props['bbox_used_for_cropping']
        ).astype(np.float32)

        self._add_patch_for_scribble_interaction(scribble_image)

        self.interactions[_CHANNEL_SCRIBBLE_POS] *= self.interaction_decay
        self.interactions[_CHANNEL_SCRIBBLE_NEG] *= self.interaction_decay

        channel = _CHANNEL_SCRIBBLE_POS if include_interaction else _CHANNEL_SCRIBBLE_NEG
        np.maximum(
            self.interactions[channel], scribble_image,
            out=self.interactions[channel]
        )

        if run_prediction:
            self._predict()

    def add_lasso_interaction(
        self,
        lasso_image: np.ndarray,
        include_interaction: bool,
        run_prediction: bool = True
    ) -> None:
        if self.original_image_shape is None:
            raise RuntimeError("Call set_image() before add_lasso_interaction().")

        assert all(
            i == j for i, j in zip(self.original_image_shape[1:], lasso_image.shape)
        ), (
            f"Lasso shape {lasso_image.shape} != "
            f"image shape {self.original_image_shape[1:]}"
        )

        if lasso_image.ndim == 4 and lasso_image.shape[0] == 1:
            lasso_image = lasso_image.squeeze(0)

        self._finish_preprocessing_and_initialize_interactions()

        lasso_image = crop_and_pad_nd(
            lasso_image, self.preprocessed_props['bbox_used_for_cropping']
        ).astype(np.float32)

        self._add_patch_for_lasso_interaction(lasso_image)

        # Lasso shares the bbox channels — intentional design from original.
        self.interactions[_CHANNEL_BBOX_POS] *= self.interaction_decay
        self.interactions[_CHANNEL_BBOX_NEG] *= self.interaction_decay

        channel = _CHANNEL_BBOX_POS if include_interaction else _CHANNEL_BBOX_NEG
        np.maximum(
            self.interactions[channel], lasso_image,
            out=self.interactions[channel]
        )

        if run_prediction:
            self._predict()

    def add_initial_seg_interaction(
        self,
        initial_seg: np.ndarray,
        run_prediction: bool = False
    ) -> None:
        """WARNING: Resets all existing interactions."""
        print("[debug] add_initial_seg_interaction()")

        assert self.original_image_shape is not None, (
            "Call set_image() first."
        )
        assert initial_seg.shape == self.original_image_shape[1:], (
            f"initial_seg shape {initial_seg.shape} != "
            f"image shape {self.original_image_shape[1:]}"
        )

        self._finish_preprocessing_and_initialize_interactions()
        self.reset_interactions()

        self.target_buffer[:] = initial_seg.astype(np.uint8)

        initial_seg_cropped = crop_and_pad_nd(
            initial_seg, self.preprocessed_props['bbox_used_for_cropping']
        ).astype(np.float32)

        self.interactions[_CHANNEL_CURRENT_SEG] = initial_seg_cropped

        if run_prediction:
            self._add_patch_for_initial_seg_interaction(initial_seg_cropped)
            self._predict()

    # ------------------------------------------------------------------
    # Patch registration helpers
    # ------------------------------------------------------------------

    def _add_patch_for_point_interaction(self, coordinates):
        self.new_interaction_zoom_out_factors.append(1)
        self.new_interaction_centers.append(coordinates)
        print(f"[debug] Point: center={coordinates}, zoom=1")

    def _add_patch_for_bbox_interaction(self, bbox):
        bbox_center = [round((i[0] + i[1]) / 2) for i in bbox]
        bbox_size = [i[1] - i[0] for i in bbox]
        requested_size = [
            i + j // 3
            for i, j in zip(bbox_size, self.configuration_manager.patch_size)
        ]
        zoom = max(
            1,
            max(i / j for i, j in zip(requested_size, self.configuration_manager.patch_size))
        )
        self.new_interaction_zoom_out_factors.append(zoom)
        self.new_interaction_centers.append(bbox_center)
        print(f"[debug] Bbox: center={bbox_center}, zoom={zoom:.2f}")

    def _add_patch_for_scribble_interaction(self, scribble_image):
        return self._generic_add_patch_from_image(scribble_image)

    def _add_patch_for_lasso_interaction(self, lasso_image):
        return self._generic_add_patch_from_image(lasso_image)

    def _add_patch_for_initial_seg_interaction(self, initial_seg):
        return self._generic_add_patch_from_image(initial_seg)

    def _generic_add_patch_from_image(self, image: np.ndarray) -> None:
        if not np.any(image):
            print("[debug] Empty image prompt — skipping patch registration.")
            return

        nonzero_indices = np.argwhere(image)
        mn = nonzero_indices.min(axis=0)
        mx = nonzero_indices.max(axis=0)

        roi = [[int(mn[i]), int(mx[i]) + 1] for i in range(len(mn))]
        roi_center = [round((i[0] + i[1]) / 2) for i in roi]
        roi_size = [i[1] - i[0] for i in roi]
        requested_size = [
            i + j // 3
            for i, j in zip(roi_size, self.configuration_manager.patch_size)
        ]
        zoom = max(
            1,
            max(i / j for i, j in zip(requested_size, self.configuration_manager.patch_size))
        )
        self.new_interaction_zoom_out_factors.append(zoom)
        self.new_interaction_centers.append(roi_center)
        print(f"[debug] Image interaction: center={roi_center}, zoom={zoom:.2f}")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _predict(self) -> None:
        """
        Entry point for inference. Routes to sliding window or single patch.

        Note: The original nnInteractiveInferenceSession has a sophisticated
        AutoZoom + diff-map + refinement pipeline. This ONNX version omits
        that pipeline as it is GPU-oriented and too expensive for CPU inference.
        If GPU ONNX inference becomes available, consider porting _refine_coarse.
        """
        assert self.pad_mode_data == 'constant', 'Only constant padding is supported.'
        assert len(self.new_interaction_centers) == len(self.new_interaction_zoom_out_factors)

        if len(self.new_interaction_centers) == 0:
            print("[debug] No interaction centers — nothing to predict.")
            return

        if self.predict_entire_image:
            self._run_sliding_window()
        else:
            self._predict_single_patch()

        # Update channel 0 (current seg) to match target_buffer,
        # mirroring what the original does via paste_tensor into interactions[0].
        if self.interactions is not None and self.target_buffer is not None:
            self.interactions[_CHANNEL_CURRENT_SEG] = self.target_buffer.astype(np.float32)

        self.new_interaction_centers.clear()
        self.new_interaction_zoom_out_factors.clear()

    def _predict_single_patch(self) -> None:
        """
        Single-patch inference centered on the most recent interaction.
        Fast path for interactive use.
        """
        print("[debug] _predict_single_patch()")
        start = time()

        patch_size = self.configuration_manager.patch_size
        bbox_offset = self.preprocessed_props['bbox_used_for_cropping']

        image_channels = self.preprocessed_image.shape[0]
        model_channels = self.network.get_inputs()[0].shape[1]
        interaction_channels = model_channels - image_channels

        for center, zoom_factor in zip(
            self.new_interaction_centers, self.new_interaction_zoom_out_factors
        ):
            zoom_factor = max(1, min(zoom_factor, 4)) if self.do_autozoom else 1
            scaled_patch_size = [round(p * zoom_factor) for p in patch_size]

            scaled_bbox = []
            for c, p, dim in zip(center, scaled_patch_size, self.preprocessed_image.shape[1:]):
                half = p // 2
                start_c = max(0, c - half)
                end_c = min(dim, c + half + (p % 2))
                scaled_bbox.append([start_c, end_c])

            crop_img, pad_img = crop_to_valid(self.preprocessed_image, scaled_bbox)
            crop_int, pad_int = crop_to_valid(self.interactions, scaled_bbox)

            if any(p != (0, 0) for p in pad_img):
                crop_img = pad_cropped(crop_img, pad_img)
            if any(p != (0, 0) for p in pad_int):
                crop_int = pad_cropped(crop_int, pad_int)

            crop_img = crop_img[:image_channels]
            crop_int = crop_int[:interaction_channels]

            if list(crop_img.shape[1:]) != patch_size:
                crop_img = _resize_np(crop_img, patch_size, order=1)
                crop_int = _resize_np(crop_int, patch_size, order=0)

            input_data = np.concatenate(
                (crop_img, crop_int), axis=0
            )[None].astype(np.float32)

            pred_logits = self.network.run(
                None, {self.input_name: input_data}
            )[0]
            pred_seg = np.argmax(pred_logits[0], axis=0).astype(np.uint8)

            # Map patch coords back to full image coords.
            global_bbox = [
                [s + off[0], e + off[0]]
                for (s, e), off in zip(scaled_bbox, bbox_offset)
            ]

            # Clamp to target buffer bounds.
            skip = False
            clamped_target, clamped_source = [], []
            for i in range(3):
                g0, g1 = global_bbox[i]
                t0 = max(g0, 0)
                t1 = min(g1, self.target_buffer.shape[i])
                if t1 <= t0:
                    print(f"[warning] Bbox out of bounds on axis {i}, skipping.")
                    skip = True
                    break
                s0 = t0 - g0
                s1 = s0 + (t1 - t0)
                clamped_target.append((t0, t1))
                clamped_source.append((s0, s1))

            if skip:
                continue

            pred_cropped = pred_seg[
                clamped_source[0][0]:clamped_source[0][1],
                clamped_source[1][0]:clamped_source[1][1],
                clamped_source[2][0]:clamped_source[2][1],
            ]
            self.target_buffer[
                clamped_target[0][0]:clamped_target[0][1],
                clamped_target[1][0]:clamped_target[1][1],
                clamped_target[2][0]:clamped_target[2][1],
            ] = pred_cropped

        self._update_viewer()
        print(f"[debug] Single patch done in {round(time() - start, 3)}s")

    def _run_sliding_window(self) -> None:
        """
        Full sliding-window inference with 50% overlap and majority vote.
        """
        print("[debug] _run_sliding_window()")
        start = time()

        patch_size = self.configuration_manager.patch_size
        stride = [max(1, s // 2) for s in patch_size]

        image_channels = self.preprocessed_image.shape[0]
        model_channels = self.network.get_inputs()[0].shape[1]
        interaction_channels = model_channels - image_channels

        image = self.preprocessed_image[:image_channels]
        interaction = self.interactions[:interaction_channels]

        D, H, W = image.shape[1:]
        seg_output   = np.zeros((D, H, W), dtype=np.int32)
        count_output = np.zeros((D, H, W), dtype=np.int32)

        for z in range(0, D, stride[0]):
            for y in range(0, H, stride[1]):
                for x in range(0, W, stride[2]):
                    z_end = min(z + patch_size[0], D)
                    y_end = min(y + patch_size[1], H)
                    x_end = min(x + patch_size[2], W)

                    patch_img = image[:, z:z_end, y:y_end, x:x_end]
                    patch_int = interaction[:, z:z_end, y:y_end, x:x_end]

                    # Pad edge patches to full patch size.
                    pad_img = [(0, 0)]
                    pad_int = [(0, 0)]
                    for axis in range(3):
                        pad_img.append((0, max(0, patch_size[axis] - patch_img.shape[axis + 1])))
                        pad_int.append((0, max(0, patch_size[axis] - patch_int.shape[axis + 1])))

                    patch_img = np.pad(patch_img, pad_img, mode='constant')
                    patch_int = np.pad(patch_int, pad_int, mode='constant')

                    input_data = np.concatenate(
                        [patch_img, patch_int], axis=0
                    )[None].astype(np.float32)

                    pred_logits = self.network.run(
                        None, {self.input_name: input_data}
                    )[0]
                    pred_seg = np.argmax(pred_logits[0], axis=0).astype(np.int32)

                    actual_d, actual_h, actual_w = z_end - z, y_end - y, x_end - x
                    seg_output  [z:z_end, y:y_end, x:x_end] += pred_seg[:actual_d, :actual_h, :actual_w]
                    count_output[z:z_end, y:y_end, x:x_end] += 1

        count_output[count_output == 0] = 1
        final_seg = (seg_output / count_output).round().astype(np.uint8)
        self.target_buffer[:] = final_seg

        self._update_viewer()
        print(f"[debug] Sliding window done in {round(time() - start, 3)}s")

    def _update_viewer(self) -> None:
        """Update napari prediction layer if a viewer is attached."""
        if not hasattr(self, '_viewer') or self._viewer is None:
            return
        pred = self.target_buffer
        if "Prediction" not in self._viewer.layers:
            self._viewer.add_labels(pred, name="Prediction")
        else:
            self._viewer.layers["Prediction"].data = pred

    # ------------------------------------------------------------------
    # Model initialization (PyTorch checkpoint path)
    # ------------------------------------------------------------------

    def initialize_from_trained_model_folder(
        self,
        model_training_output_dir: str,
        use_fold: Union[int, str] = None,
        checkpoint_name: str = 'checkpoint_final.pth'
    ) -> None:
        """
        Load a PyTorch checkpoint. Only needed for non-ONNX models.
        For ONNX models, load_network() in __init__ handles everything.
        """
        print("[debug] initialize_from_trained_model_folder:", model_training_output_dir)
        model_training_output_dir = str(model_training_output_dir)

        expected_json = join(model_training_output_dir, 'inference_session_class.json')
        json_content = load_json(expected_json)

        # isinstance check matches original: old convention = str, new = dict
        if isinstance(json_content, str):
            # Old convention — just a class name string, use defaults
            point_interaction_radius = 4
            point_interaction_use_etd = True
            self.preferred_scribble_thickness = [2, 2, 2]
            self.pad_mode_data = "constant"
            self.interaction_decay = 0.9
        else:
            # New convention — full config dict
            point_interaction_radius = json_content['point_radius']
            self.preferred_scribble_thickness = json_content['preferred_scribble_thickness']
            if not isinstance(self.preferred_scribble_thickness, (tuple, list)):
                self.preferred_scribble_thickness = [self.preferred_scribble_thickness] * 3
            self.interaction_decay = json_content.get('interaction_decay', 0.98)
            point_interaction_use_etd = True
            self.pad_mode_data = json_content.get('pad_mode_image', 'constant')

        self.point_interaction = PointInteraction_stub(
            point_interaction_radius, point_interaction_use_etd
        )

        dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
        plans = load_json(join(model_training_output_dir, 'plans.json'))
        plans_manager = PlansManager(plans)

        if use_fold is not None:
            fold_folder = f'fold_{int(use_fold) if use_fold != "all" else use_fold}'
        else:
            fldrs = subdirs(model_training_output_dir, prefix='fold_', join=False)
            assert len(fldrs) == 1, f'Expected 1 fold_ folder, found: {fldrs}'
            fold_folder = fldrs[0]

        checkpoint = torch.load(
            join(model_training_output_dir, fold_folder, checkpoint_name),
            map_location='cpu',
            weights_only=False
        )
        trainer_name = checkpoint['trainer_name']
        configuration_name = checkpoint['init_args']['configuration']
        parameters = checkpoint['network_weights']

        configuration_manager = plans_manager.get_configuration(configuration_name)
        num_input_channels = determine_num_input_channels(
            plans_manager, configuration_manager, dataset_json
        )

        trainer_class = recursive_find_python_class(
            join(nnInteractive.__path__[0], "trainer"),
            trainer_name, 'nnInteractive.trainer'
        )
        if trainer_class is None:
            print(f"Trainer '{trainer_name}' not found — using nnInteractiveTrainer_stub.")
            trainer_class = nnInteractiveTrainer_stub

        # Use the current API signature (matches original inference_session.py)
        network = trainer_class.build_network_architecture(
            plans_manager,
            configuration_manager,
            num_input_channels,
            plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            enable_deep_supervision=False
        )
        network.load_state_dict(parameters)

        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.network = network
        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.label_manager = plans_manager.get_label_manager(dataset_json)

    def manual_initialization(
        self,
        network: nn.Module,
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        dataset_json: dict,
        trainer_name: str
    ) -> None:
        """For nnUNetTrainer validation use only."""
        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.network = network
        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.label_manager = plans_manager.get_label_manager(dataset_json)


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def transform_coordinates_noresampling(
        coords_orig: Union[List[int], Tuple[int, ...]],
        nnunet_preprocessing_crop_bbox: List[Tuple[int, int]]
) -> Tuple[int, ...]:
    """Convert coordinates from original image space to cropped image space."""
    return tuple(
        coords_orig[d] - nnunet_preprocessing_crop_bbox[d][0]
        for d in range(len(coords_orig))
    )


def _resize_np(arr: np.ndarray, new_shape, order: int = 1) -> np.ndarray:
    """
    Resize a 4D (C, D, H, W) or 3D (D, H, W) numpy array to new_shape
    using scipy zoom.
    """
    if arr.ndim == 4:
        return np.stack([
            zoom(c, [n / o for n, o in zip(new_shape, c.shape)], order=order)
            for c in arr
        ], axis=0)
    elif arr.ndim == 3:
        return zoom(arr, [n / o for n, o in zip(new_shape, arr.shape)], order=order)
    else:
        raise ValueError(f"Unsupported array shape: {arr.shape}")