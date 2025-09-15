from concurrent.futures import ThreadPoolExecutor
from time import time
from typing import Union, List, Tuple, Optional
import numpy as np
from scipy.ndimage import zoom, generate_binary_structure, binary_dilation
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice, crop_and_pad_nd
from batchgenerators.utilities.file_and_folder_operations import load_json, join, subdirs
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.helpers import dummy_context, empty_cache
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager


from torch import nn
from torch._dynamo import OptimizedModule
import SimpleITK as sitk
import onnxruntime as ort

import nnInteractive
from nnInteractive.interaction.point import PointInteraction_stub
from nnInteractive.trainer.nnInteractiveTrainer import nnInteractiveTrainer_stub
from nnInteractive.utils.bboxes import generate_bounding_boxes
from nnInteractive.utils.crop import crop_and_pad_into_buffer, paste_tensor, pad_cropped, crop_to_valid
from nnInteractive.utils.erosion_dilation import iterative_3x3_same_padding_pool3d
from nnInteractive.utils.rounding import round_to_nearest_odd
import torch

print("Inference_session is being used here!!")

class nnInteractiveInferenceSession():
    def __init__(self, model_path, device="cpu", do_autozoom=False, verbose=True):
        print("nnInteractiveInferenceSession has run __init__")

        self.predict_entire_image = True
        self.verbose = verbose
        self.do_autozoom: bool = do_autozoom
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_torch_compile = False
        self.preferred_scribble_thickness = [2, 2, 2]
        self.configuration_manager = SimpleONNXConfig(patch_size=(128, 128, 128))

        # Load ONNX model
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        expected_input_shape = self.session.get_inputs()[0].shape
        print(f"[debug] Model input shape: {expected_input_shape}")
        print(f"[debug] Model output shape: {self.session.get_outputs()[0].shape}")

        # Expect NCDHW format
        _, expected_channels, _, _, _ = expected_input_shape

        # ---- IMAGE VS INTERACTIONS ----
        image_channels = 1  # assume grayscale input

        # Define enabled interaction types
        self.enabled_interactions = {
            "point": True,
            "bbox": True,
            "scribble": True,
            "lasso": False,
            "initial_segmentation": True
        }

        # Channels per interaction type
        CHANNELS_PER_TYPE = {
            "point": 2,
            "bbox": 2,
            "scribble": 2,
            "lasso": 2,
            "initial_segmentation": 1
        }

        # Count how many channels are needed
        self.num_interaction_channels = sum(
            CHANNELS_PER_TYPE[name] for name, enabled in self.enabled_interactions.items() if enabled
        )

        # Sanity check against the model
        expected_total_channels = expected_channels
        assert image_channels + self.num_interaction_channels == expected_total_channels, (
            f"Channel mismatch! Model expects {expected_total_channels} total channels, "
            f"but you configured {image_channels + self.num_interaction_channels} "
            f"({image_channels} image + {self.num_interaction_channels} interaction)."
        )

        # Allocate interaction buffer (will be resized once input image is set)
        self.input_shape = (0, 0, 0)  # placeholder, filled later
        self.interactions = np.zeros(
            (self.num_interaction_channels, *self.input_shape), dtype=np.float32
        )

        print(f"[debug] Enabled interactions: {self.enabled_interactions}")
        print(f"[debug] Interaction channels: {self.num_interaction_channels}")
        print(f"[debug] Final total channels: {image_channels + self.num_interaction_channels}")

        # Other session state
        self.preprocessed_image: np.ndarray = None
        self.preprocessed_props = None
        self.target_buffer: Union[np.ndarray, torch.Tensor] = None
        self.original_image_shape = None
        self.pad_mode_data = "constant"

        # For user interactions
        self.new_interaction_zoom_out_factors: List[float] = []
        self.new_interaction_centers = []
        self.has_positive_bbox = False

        # Thread pool for async preprocessing
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.preprocess_future = None
        self.interactions_future = None

    def load_network(self, model_path, providers):
        assert model_path.exists(), f"Model file does not exist: {model_path}"
        assert model_path.suffix == ".onnx", f"Expected .onnx file, got: {model_path.suffix}"
        print(f"[debug] Loading ONNX model from: {model_path}")
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.network.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        print(f"[debug] Model loaded successfully. Input name: {self.input_name}")
        print("[debug] input shape expected by model:", self.session.get_inputs()[0].shape)
        print("[debug] output shape expected by model:", self.session.get_outputs()[0].shape)


        def get_providers():
            available_providers = ort.get_all_providers()
            print(f"[debug] Available providers: {available_providers}")  # print procviders
            if "CUDAExecutionProvider" in available_providers:
                return ["CUDAExecutionProvider"]  # cuda
            elif "CoreMLExecutionProvider" in available_providers:
                return ["CoreMLExecutionProvider"]  # coreml
            else:
                return ["CPUExecutionProvider"]  # cpu (worst choice)

        providers = get_providers()

        if not providers:
            print("No provider found... have you checked your installed drivers?")

        #self.session = ort.InferenceSession("/Users/shayanlatif/PycharmProjects/AIModelConversionKit/interactive_nnunet_fp16.onnx", providers=providers)
        self.label_manager = None
        self.dataset_json = None
        self.trainer_name = None
        self.plans_manager = None
        #self.use_pinned_memory = use_pinned_memory
        #self.use_torch_compile = use_torch_compile

        # Set shape and decay manually
        #self.source_shape = (128,128,128)
        self.interaction_decay = 0.9
        # Define enabled interaction types:
        self.enabled_interactions = {
            "point": True,
            "bbox": True,
            "scribble": True,
            "lasso": True,
            "initial_segmentation": True
        }

        CHANNELS_PER_TYPE = {
            "point": 2,
            "bbox": 2,
            "scribble": 2,
            "lasso": 2,
            "initial_segmentation": 1
        }

        # Count channels needed:
        self.num_interaction_channels = sum([
            2 if enabled else 0
            for enabled in self.enabled_interactions.values()
        ])

        self.input_shape = (0,0,0)
        image_channels = 1 # assume grayscale input
        expected_input_shape = self.session.get_inputs()[0].shape

        expected_total_channels = expected_input_shape[1]  # ONNX uses NCHWD

        self.num_interaction_channels = expected_total_channels - image_channels
        self.interactions = np.zeros((self.num_interaction_channels, *self.input_shape), dtype=np.float32)

        # Determine total expected input channels
        expected_total_channels = expected_input_shape[1]

        #print("[debug] source shape", self.source_shape)
        print("[debug] Session ID (init):", id(self))

        #self.output_name = self.session.get_outputs()[0].name

        # image specific
        #self.interactions: torch.Tensor = None I commented this out cause it was causing problems. I hope this doesnt do anything too bad...
        self.preprocessed_image: np.array = None
        self.preprocessed_props = None
        self.target_buffer: Union[np.ndarray, torch.Tensor] = None

        # this will be set when loading the model (initialize_from_trained_model_folder), commented out cause it was also problem
        #self.pad_mode_data = self.preferred_scribble_thickness = self.point_interaction = None
        self.pad_mode_data = "constant"
        #self.point_interaction = None

        # ill be surprised if this works!!
        self.point_interaction = PointInteraction_stub(
            4,
            True
        )

        self.original_image_shape = None


        self.new_interaction_zoom_out_factors: List[float] = []
        self.new_interaction_centers = []
        self.has_positive_bbox = False

        # Create a thread pool executor for background tasks.
        # this only takes care of preprocessing and interaction memory initialization so there is no need to give it
        # more than 2 workers
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.preprocess_future = None
        self.interactions_future = None

    def set_image(self, image: np.ndarray, image_properties: dict = None):
        """
        Image must be 4D to satisfy nnU-Net needs: [c, x, y, z]
        Offload the processing to a background thread.
        """
        if image_properties is None:
            image_properties = {}
        self._reset_session()
        if image.ndim == 3:
            print("[debug] Input was 3D; adding channel dimension to make it 4D.")
            image = image[None, ...]
        assert image.ndim == 4, f'expected a 4d image as input, got {image.ndim}d. Shape {image.shape} data type {image.dtype}' #CHANGE
        print("[debug] input image channels =", image.shape[0])
        if self.verbose:
            print(f'Initialize with raw image shape {image.shape}')

        # Offload all image preprocessing to a background thread.
        self.preprocess_future = self.executor.submit(self._background_set_image, image, image_properties)
        self.original_image_shape = image.shape

    def _finish_preprocessing_and_initialize_interactions(self):
        print("[debug] _finish_preprocessing_and_initialize_interactions() was called")

        if self.original_image_shape is not None:
            self.preprocessed_props = {
                'bbox_used_for_cropping': [(0, d) for d in self.original_image_shape]
            }

        elif self.preprocessed_props is None:
            raise RuntimeError("Preprocessing still not done after waiting! Check _background_set_image in inference_session.py")

        """
        Block until both the image preprocessing and the interactions tensor initialization
        are finished.
        """
        if self.preprocess_future is not None:
            # Wait for image preprocessing to complete.
            self.preprocess_future.result()
            del self.preprocess_future
            self.preprocess_future = None



    def set_target_buffer(self, target_buffer: Union[np.ndarray, torch.Tensor]):
        print("[debug] set_target_buffer() was called")
        """
        Must be 3d numpy array or torch.Tensor
        """
        self.target_buffer = target_buffer

    def set_do_autozoom(self, do_propagation: bool, max_num_patches: Optional[int] = None):
        print("[debug] set_do_autozoom() was called")
        self.do_autozoom = do_propagation

    def _reset_session(self):
        print("[debug] _reset_session() was called")
        self.interactions_future = None
        self.preprocess_future = None

        del self.preprocessed_image
        del self.target_buffer
        del self.interactions
        del self.preprocessed_props
        self.preprocessed_image = None
        self.target_buffer = None
        self.interactions = None
        self.preprocessed_props = None
        empty_cache(self.device)
        self.original_image_shape = None
        self.has_positive_bbox = False

    def _initialize_interactions(self, image: np.ndarray):
        print("[debug] _intialize_interactions was called")
        if self.verbose:
            print(f'Initialize interactions. Pinned: {self.use_pinned_memory}')
        # Create the interaction tensor based on the target shape.
        print("[debug] given image shape = ", image.shape)

        assert image.ndim == 4, f'expected a 4d image as input, got {image.ndim}d. Shape {image.shape} data type {image.dtype}' #CHANGE
        _, d, h, w = image.shape

        # Extract shape data from image with 4d tuple
        self.input_shape = (d, h, w)
        self.interactions = np.zeros((self.num_interaction_channels, d, h, w), dtype=np.float32)
        self.target_buffer = np.zeros_like(self.interactions[0], dtype=np.uint8)

        print("[debug] initialized interactions shape =", self.interactions.shape)
        print("[debug] model expects input shape:", self.session.get_inputs()[0].shape)

        #self.interactions = np.zeros(
        #    (self.num_interaction_channels, *self.source_shape),
        #    dtype=np.float32
        #)


        if self.interactions is None:
            print("[debug] ⚠ interactions is still None! ⚠")
            return

    def _background_set_image(self, image: np.ndarray, image_properties: dict):
        # Convert and clone the image tensor.
        print("[debug] _background_set_image was called")
        image_torch = (image - np.mean(image)) / np.std(image)
        # Crop to nonzero region.
        if self.verbose:
            print('Cropping input image to nonzero region')
        nonzero_idx = np.nonzero(image_torch)
        # Create bounding box: for each dimension, get the min and max (plus one) of the nonzero indices.
        bbox = [[int(np.min(i)), int(np.max(i)) + 1] for i in nonzero_idx]
        del nonzero_idx
        slicer = bounding_box_to_slice(bbox)  # Assuming this returns a tuple of slices.
        image_torch = image_torch[slicer].astype(np.float32)
        if self.verbose:
            print(f'Cropped image shape: {image_torch.shape}')

        # As soon as we have the target shape, start initializing the interaction tensor in its own thread.
        self.interactions_future = self.executor.submit(self._initialize_interactions, image_torch)

        # Normalize the cropped image.
        if self.verbose:
            print('Normalizing cropped image')
        image_torch -= image_torch.mean()
        image_torch /= image_torch.std()

        self.preprocessed_image = image_torch

        # No pinned memory in ONNX files... but if u change it to torch, dont forget this!!!
        #if self.use_pinned_memory and self.device.type == 'cuda':
        if self.verbose:
            print('Pin memory: image')
            # Note: pin_memory() in PyTorch typically returns a new tensor.
        #self.preprocessed_image = self.preprocessed_image.pin_memory()

        self.preprocessed_props = {'bbox_used_for_cropping': bbox[1:]}

        self.original_image_shape = image.shape

        # we need to wait for this here I believe
        self.interactions_future.result()
        del self.interactions_future
        self.interactions_future = None

        print("[debug] preprocessed_image stats:",
              "shape =", self.preprocessed_image.shape,
              "min =", np.min(self.preprocessed_image),
              "max =", np.max(self.preprocessed_image),
              "mean =", np.mean(self.preprocessed_image),
              "std =", np.std(self.preprocessed_image))

        print("[debug] interactions shape =", self.interactions.shape,
              "nonzero channels =", [i for i in range(self.interactions.shape[0])
                                     if np.any(self.interactions[i] != 0)])

        return self.preprocessed_props, bbox

    def reset_interactions(self):
        """
        Use this to reset all interactions and start from scratch for the current image. This includes the initial
        segmentation!
        """

        print("[debug] reset_interactions was called")

        if self.interactions is not None:
            self.interactions.fill(0)

        if self.target_buffer is not None:
            if isinstance(self.target_buffer, np.ndarray):
                self.target_buffer.fill(0)
            elif isinstance(self.target_buffer, torch.Tensor):
                self.target_buffer.zero_()
        empty_cache(self.device)
        self.has_positive_bbox = False

    def add_bbox_interaction(self, bbox_coords, include_interaction: bool, run_prediction: bool = True) -> np.ndarray:
        print("[debug] add_bbox_interaction was called")
        if include_interaction:
            self.has_positive_bbox = True

        lbs_transformed = [round(i) for i in transform_coordinates_noresampling([i[0] for i in bbox_coords],
                                                             self.preprocessed_props['bbox_used_for_cropping'])]
        ubs_transformed = [round(i) for i in transform_coordinates_noresampling([i[1] for i in bbox_coords],
                                                             self.preprocessed_props['bbox_used_for_cropping'])]
        transformed_bbox_coordinates = [[i, j] for i, j in zip(lbs_transformed, ubs_transformed)]

        if self.verbose:
            print(f'Added bounding box coordinates.\n'
                  f'Raw: {bbox_coords}\n'
                  f'Transformed: {transformed_bbox_coordinates}\n'
                  f"Crop Bbox: {self.preprocessed_props['bbox_used_for_cropping']}")

        # Prevent collapsed bounding boxes and clip to image shape
        image_shape = self.preprocessed_image.shape  # Assuming shape is (C, H, W, D) or similar

        for dim in range(len(transformed_bbox_coordinates)):
            transformed_start, transformed_end = transformed_bbox_coordinates[dim]

            # Clip to image boundaries
            transformed_start = max(0, transformed_start)
            transformed_end = min(image_shape[dim + 1], transformed_end)  # +1 to skip channel dim

            # Ensure the bounding box does not collapse to a single point
            if transformed_end <= transformed_start:
                if transformed_start == 0:
                    transformed_end = min(1, image_shape[dim + 1])
                else:
                    transformed_start = max(transformed_start - 1, 0)

            transformed_bbox_coordinates[dim] = [transformed_start, transformed_end]

        if self.verbose:
            print(f'Bbox coordinates after clip to image boundaries and preventing dim collapse:\n'
                  f'Bbox: {transformed_bbox_coordinates}\n'
                  f'Internal image shape: {self.preprocessed_image.shape}')

        self._add_patch_for_bbox_interaction(transformed_bbox_coordinates)

        # decay old interactions
        self.interactions[-6:-4] *= self.interaction_decay

        # place bbox
        slicer = tuple([slice(*i) for i in transformed_bbox_coordinates])
        channel = -6 if include_interaction else -5
        self.interactions[(channel, *slicer)] = 1

        # forward pass
        if run_prediction:
            self._predict()

    def add_point_interaction(self, coordinates: Tuple[int, ...], include_interaction: bool, run_prediction: bool = True):

        print("[debug] add_point_interaction was called")

        # print("[debug] interactions before calling _finish_preprocessing_and_initialize_interactions:", self.interactions.shape)

        self._finish_preprocessing_and_initialize_interactions()
        assert self.interactions.dtype == np.float32 #Interactions MUST be float 32

        transformed_coordinates = [round(i) for i in transform_coordinates_noresampling(coordinates,
                                                             self.preprocessed_props['bbox_used_for_cropping'])]

        self._add_patch_for_point_interaction(transformed_coordinates)

        # Debug to print interactions
        print("[debug] interactions =", self.interactions.shape)
        print("[debug] decay =", self.interaction_decay)
        # print("[debug] Session ID (add_point_interaction):", id(self))

        # decay old interactions
        self.interactions[-4:-2] *= self.interaction_decay

        interaction_channel = -4 if include_interaction else -3
        self.interactions[interaction_channel] = self.point_interaction.place_point(
            transformed_coordinates, self.interactions[interaction_channel])
        if run_prediction:
            self._predict()
        print("[debug] interactions after add_point_interaction:", np.unique(self.interactions))

    def add_scribble_interaction(self, scribble_image: np.ndarray,  include_interaction: bool, run_prediction: bool = True):
        assert all([i == j for i, j in zip(self.original_image_shape[1:], scribble_image.shape)]), f'Given scribble image must match input image shape. Input image was: {self.original_image_shape[1:]}, given: {scribble_image.shape}'
        self._finish_preprocessing_and_initialize_interactions()

        scribble_image = torch.from_numpy(scribble_image)

        # crop (as in preprocessing)
        scribble_image = crop_and_pad_nd(scribble_image, self.preprocessed_props['bbox_used_for_cropping'])

        self._add_patch_for_scribble_interaction(scribble_image)

        # decay old interactions
        self.interactions[-2:] *= self.interaction_decay

        interaction_channel = -2 if include_interaction else -1
        torch.maximum(self.interactions[interaction_channel], scribble_image.to(self.interactions.device),
                      out=self.interactions[interaction_channel])
        del scribble_image
        empty_cache(self.device)
        if run_prediction:
            self._predict()

        if self.verbose:
            print("Waiting for preprocess_future to finish...")
        self.preprocess_future.result()  # wait until preprocessing done
        self.preprocess_future = None

    def add_lasso_interaction(self, lasso_image: np.ndarray,  include_interaction: bool, run_prediction: bool = True):
        assert all([i == j for i, j in zip(self.original_image_shape[1:], lasso_image.shape)]), f'Given lasso image must match input image shape. Input image was: {self.original_image_shape[1:]}, given: {lasso_image.shape}'
        print("[debug] add_lasso_interaction was called")
        # Stop program if image shape isnt changed
        if self.original_image_shape is None:
            raise RuntimeError(
                "You must call add_image() or run() before using add_lasso_interaction(). original_image_shape is not set.")

        # remove initial channel input
        if lasso_image.shape[0] == 1:
            print("removing channel input of lasso_image...")
            lasso_image = lasso_image[1:]

        self._finish_preprocessing_and_initialize_interactions()

        lasso_image = torch.from_numpy(lasso_image)

        # crop (as in preprocessing)
        lasso_image = crop_and_pad_nd(lasso_image, self.preprocessed_props['bbox_used_for_cropping'])

        self._add_patch_for_lasso_interaction(lasso_image)

        # decay old interactions
        self.interactions[-6:-4] *= self.interaction_decay

        # lasso is written into bbox channel
        interaction_channel = -6 if include_interaction else -5
        torch.maximum(self.interactions[interaction_channel], lasso_image.to(self.interactions.device),
                      out=self.interactions[interaction_channel])
        del lasso_image
        # commented out cause its acting up
        #empty_cache(self.device)
        if run_prediction:
            self._predict()

    def add_initial_seg_interaction(self, initial_seg: np.ndarray, run_prediction: bool = False):
        """
        WARNING THIS WILL RESET INTERACTIONS!
        """

        print("[debug] add_initial_seg_interaction was called")

        assert self.original_image_shape is not None, "Original image shape not set. Did you call set_image first?"

        assert initial_seg.shape == self.original_image_shape[1:], (
            f"Initial seg must match input image shape. "
            f"Expected {self.original_image_shape[1:]}, got {initial_seg.shape}"
        )
        self._finish_preprocessing_and_initialize_interactions()

        self.reset_interactions()

        if isinstance(self.target_buffer, np.ndarray):
            self.target_buffer[:] = initial_seg

        elif isinstance(self.target_buffer, torch.Tensor):
            self.target_buffer[:] = torch.from_numpy(initial_seg)

        # crop (as in preprocessing)
        # commented out cause torch function... idk if this is a bad idea...
        #initial_seg = torch.from_numpy(initial_seg)
        initial_seg = crop_and_pad_nd(initial_seg, self.preprocessed_props['bbox_used_for_cropping'])

        # initial seg is written into initial seg buffer
        interaction_channel = -4
        self.interactions[interaction_channel] = initial_seg
        empty_cache(self.device)
        if run_prediction:
            self._add_patch_for_initial_seg_interaction(initial_seg)
            del initial_seg
            self._predict()
        else:
            del initial_seg

    def _predict_full_image(self, region_center: Optional[Tuple[int, int, int]] = None):
        print("[debug] PredictFullImage was called")
        # Assume: self.preprocessed_image shape is (D, H, W)
        #          self.interactions shape is (C_interaction, D, H, W)
        #          self.patch_size = (pD, pH, pW)

        print("[debug] preprocessed_image shape =", self.preprocessed_image.shape)
        print("[debug] interactions shape =", self.interactions.shape)
        print("[debug] total channels =", self.preprocessed_image.shape[0] + self.interactions.shape[0])
        print("[debug] ONNX expects:", self.session.get_inputs()[0].shape)

        seg_prediction = np.zeros_like(self.interactions[0])  # Shape: (D, H, W)
        count_map = np.zeros_like(seg_prediction)

        print("[debug] Running full image prediction")
        print(f"[debug] count_map is {count_map.shape}")
        patch_size = self.configuration_manager.patch_size
        stride = [s // 2 for s in patch_size]  # 50% overlap
        image = self.preprocessed_image
        print(f"[debug] image is {image.shape}")
        interaction = self.interactions
        print(f"[debug] interaction is {interaction.shape}")

        bbox_offset = self.preprocessed_props["bbox_used_for_cropping"]
        print(f"[debug] bbox_offset is {(bbox_offset)}")

        if len(bbox_offset) == 4:
            _, z_range, y_range, x_range = bbox_offset
        elif len(bbox_offset) == 3:
            z_range, y_range, x_range = bbox_offset

        z0, y0, x0 = z_range[0], y_range[0], x_range[0]

        print("z0, y0, x0", z0, y0, x0)

        start_time = time()

        orig_shape = (
            image.shape[1] + z0,
            image.shape[2] + y0,
            image.shape[3] + x0,
        )

        seg_output = np.zeros(orig_shape, dtype=np.uint8)
        count_output = np.zeros(orig_shape, dtype=np.uint8)


        D, H, W = image.shape[1:]

        print("[debug] self.new_interaction_centers=", self.new_interaction_centers)

        if self.new_interaction_centers is not None:
            print('it worked!! self.new_interaction_centers =', self.new_interaction_centers)

            # Use most recent center
            if len(self.new_interaction_centers[0]) == 4:
                cz, cy, cx = self.new_interaction_centers[-1][1:4]
            else:
                cz, cy, cx = self.new_interaction_centers[-1]
            pD, pH, pW = patch_size
            D, H, W = image.shape[1:]
            print ("[debug] self.new_interaction_centers =", self.new_interaction_centers)
            print ("[debug] patch_size =", patch_size)
            print ("image.shape", {image.shape})


            # Calculate patch bounds
            z = max(0, min(D - pD, cz - pD // 2))
            y = max(0, min(H - pH, cy - pH // 2))
            x = max(0, min(W - pW, cx - pW // 2))

            scaled_bbox = []

            for c, p, dim in zip(self.new_interaction_centers[-1], patch_size, self.preprocessed_image.shape[1:]):
                half = p // 2
                start = max(0, c - half)
                end = min(dim, c + half + (p % 2))
                scaled_bbox.append([start, end])
            print("[debug] scaled_patch_size:", patch_size)
            print("[debug] scaled_bbox:", scaled_bbox)

            print(f"patch bounds z,y,x:{z,y,x}")
            # Extract patch
            patch_image = image[:, z:z + pD, y:y + pH, x:x + pW]
            patch_interaction = interaction[:, z:z + pD, y:y + pH, x:x + pW]

            print("[debug] patch_interaction.shape =", patch_interaction.shape)
            print("[debug] patch_image.shape =", patch_image.shape)

            model_input = np.concatenate([patch_image, patch_interaction], axis=0)[np.newaxis]

            print(f"self.input_name: {self.input_name}")
            print(f"model_input.shape: {model_input.shape}")
            print(f"model_input.dtype: {model_input.dtype}")

            pred = self.onnx_session.run(None, {self.input_name: model_input})[0][0]
            pred_seg = np.argmax(pred, axis=0).astype(np.uint8)  # [D, H, W]

            # Overlay prediction into correct place in the target buffer
            # We may need to uncrop using z0/y0/x0 offsets!
            z0, y0, x0 = z_range[0], y_range[0], x_range[0]
            abs_z = z + z0
            abs_y = y + y0
            abs_x = x + x0

            # Flip axes if needed (match for full image)
            pred_seg = np.flip(pred_seg, axis=(1, 2))

            target_slice = self.target_buffer[
                           abs_z:abs_z + pred_seg.shape[0],
                           abs_y:abs_y + pred_seg.shape[1],
                           abs_x:abs_x + pred_seg.shape[2]
                           ]

            print("before:", np.unique(target_slice))

            if not np.array_equal(pred_seg, target_slice):
                print("This patch is changing something!")

                # Update target buffer directly (like overlaying a window of prediction)
                self.target_buffer[
                abs_z:abs_z + pred_seg.shape[0],
                abs_y:abs_y + pred_seg.shape[1],
                abs_x:abs_x + pred_seg.shape[2]
                ] = pred_seg

        # Concatenate image + interactions globally (not per patch)
        combined = np.concatenate([image, interaction], axis=0)  # (C, D, H, W)
        print("[debug] combined shape before padding =", combined.shape)

        # Pad channels once to 8
        required_channels = 8
        if combined.shape[0] < required_channels:
            new_combined = np.zeros((required_channels, *combined.shape[1:]), dtype=combined.dtype)
            new_combined[:combined.shape[0]] = combined
            combined = new_combined
            print("[debug] Globally padded combined shape =", combined.shape)

        patches = []
        bboxes = []

        # Collect patches in this messy loop that may be improved later
        # for z in range(0, D, stride[0]):
        #     for y in range(0, H, stride[1]):
        #         for x in range(0, W, stride[2]):
        #             print("[debug] z =", z, "y =", y, "x =", x)
        #             patch_bbox = [
        #                 [z, min(z + patch_size[0], D)],
        #                 [y, min(y + patch_size[1], H)],
        #                 [x, min(x + patch_size[2], W)],
        #             ]
        #
        #             patch = combined[:,
        #                     patch_bbox[0][0]:patch_bbox[0][1],
        #                     patch_bbox[1][0]:patch_bbox[1][1],
        #                     patch_bbox[2][0]:patch_bbox[2][1]]
        #
        #             interaction_patch = interaction[:,
        #                                 patch_bbox[0][0]:patch_bbox[0][1],
        #                                 patch_bbox[1][0]:patch_bbox[1][1],
        #                                 patch_bbox[2][0]:patch_bbox[2][1]]
        #
        #             # Pad spatially to patch_size
        #             pad_width = [(0, 0)]
        #             for axis in range(3):
        #                 pad_amount = patch_size[axis] - patch.shape[axis + 1]
        #                 before = 0
        #                 after = pad_amount if pad_amount > 0 else 0
        #                 pad_width.append((before, after))
        #
        #             patch = np.pad(patch, pad_width, mode='constant')
        #             interaction_patch = np.pad(interaction_patch, pad_width, mode='constant')
        #
        #             patches.append(patch)
        #             bboxes.append(patch_bbox)
        #             print("One iteration done")
        # # Stack patches into (N, C, D, H, W)
        # print(f"collection of patches done, patches = {patches}, bboxes = {bboxes}, now its time stack patches to input_data")
        # input_data = np.stack(patches, axis=0).astype(np.float32)
        #
        # # Run ONNX inference on all patches
        # print(f"stacked all of the patches into input data successfully, input_data = {input_data}, now its time to run onnx")
        # # adjust based on memory limits
        # batch_size = 32
        # pred_logits_list = []
        # for i in range(0, len(patches), batch_size):
        #     print(f"patches[0].shape: {patches[0].shape}")
        #     print(f"patches[i].shape: {patches[i].shape}")
        #
        #     batch = np.stack(patches[i:i + batch_size], axis=0).astype(np.float32)
        #     pred_logits_batch = self.onnx_session.run(None, {self.input_name: batch})[0]
        #     pred_logits_list.append(pred_logits_batch)
        #
        # # Concatenate all results back into (N, num_classes, D, H, W)
        # pred_logits = np.concatenate(pred_logits_list, axis=0)
        # print("Ran onnx successfully")
        # print(f"Pred_logits.shape: {pred_logits.shape}")
        #
        # # Stitch predictions back to pred_seg
        # for i, patch_bbox in enumerate(bboxes):
        #     print("Stitching with", i, patch_bbox)
        #     pred_seg = np.argmax(pred_logits[0], axis=0).astype(np.uint8)
        #
        #     dz = patch_bbox[0][1] - patch_bbox[0][0]
        #     dy = patch_bbox[1][1] - patch_bbox[1][0]
        #     dx = patch_bbox[2][1] - patch_bbox[2][0]
        #     pred_seg = pred_seg[:dz, :dy, :dx]
        #
        #     z_start = patch_bbox[0][0] + z0
        #     y_start = patch_bbox[1][0] + y0
        #     x_start = patch_bbox[2][0] + x0
        #
        #     z_end = z_start + dz
        #     y_end = y_start + dy
        #     x_end = x_start + dx
        #
        #     seg_output[z_start:z_end, y_start:y_end, x_start:x_end] += pred_seg
        #     count_output[z_start:z_end, y_start:y_end, x_start:x_end] += 1

        # Iterate over 3D volume in patches
        for z in range(0, D, stride[0]):
            for y in range(0, H, stride[1]):
                for x in range(0, W, stride[2]):

                    # Define patch bounding box (start and end coordinates)
                    patch_bbox = [
                        [z, min(z + patch_size[0], D)],
                        [y, min(y + patch_size[1], H)],
                        [x, min(x + patch_size[2], W)],
                    ]

                    # Extract patches from inputs
                    patch = combined[
                            :,
                            patch_bbox[0][0]:patch_bbox[0][1],
                            patch_bbox[1][0]:patch_bbox[1][1],
                            patch_bbox[2][0]:patch_bbox[2][1]
                            ]

                    interaction_patch = interaction[
                                        :,
                                        patch_bbox[0][0]:patch_bbox[0][1],
                                        patch_bbox[1][0]:patch_bbox[1][1],
                                        patch_bbox[2][0]:patch_bbox[2][1]
                                        ]

                    # Pad patch spatially to match patch_size
                    pad_width = [(0, 0)]  # No padding for channel dimension
                    for axis in range(3):
                        pad_amount = patch_size[axis] - patch.shape[axis + 1]
                        before = 0
                        after = pad_amount if pad_amount > 0 else 0
                        pad_width.append((before, after))

                    patch = np.pad(patch, pad_width, mode='constant')
                    interaction_patch = np.pad(interaction_patch, pad_width, mode='constant')

                    # Prepare input for model
                    input_data = patch[None].astype(np.float32)  # Add batch dimension
                    print("[debug] input_data.shape =", input_data.shape)

                    # Run ONNX model
                    pred_logits = self.onnx_session.run(None, {self.input_name: input_data})[0]
                    pred_seg = np.argmax(pred_logits[0], axis=0).astype(np.uint8)

                    # Crop to original patch shape
                    dz = patch_bbox[0][1] - patch_bbox[0][0]
                    dy = patch_bbox[1][1] - patch_bbox[1][0]
                    dx = patch_bbox[2][1] - patch_bbox[2][0]
                    pred_seg = pred_seg[:dz, :dy, :dx]

                    # Compute global coordinates
                    z_start = patch_bbox[0][0] + z0
                    y_start = patch_bbox[1][0] + y0
                    x_start = patch_bbox[2][0] + x0
                    z_end = z_start + pred_seg.shape[0]
                    y_end = y_start + pred_seg.shape[1]
                    x_end = x_start + pred_seg.shape[2]
                    print("z_start", z_start, "y_start", y_start, "x_start", x_start)
                    print("z_end", z_end, "y_end", y_end, "x_end", x_end)

                    # Accumulate prediction and count
                    seg_output[z_start:z_end, y_start:y_end, x_start:x_end] += pred_seg
                    count_output[z_start:z_end, y_start:y_end, x_start:x_end] += 1

        print("[debug] count_output =", count_output.shape)
        print("[debug] seg_output =", seg_output.shape)
        count_output[count_output == 0] = 1
        final_seg = (seg_output / count_output).round().astype(np.uint8)

        final_seg = (1 - final_seg)
        final_seg = np.flip(final_seg, axis=(1, 2))
        spacing = [1.5, 1.5, 1.5]

        z0 = max(0, min(z0, final_seg.shape[0] - self.target_buffer.shape[0]))
        y0 = max(0, min(y0, final_seg.shape[1] - self.target_buffer.shape[1]))
        x0 = max(0, min(x0, final_seg.shape[2] - self.target_buffer.shape[2]))

        print(f"final_seg.shape: {final_seg.shape}")
        print(f"z0, y0, x0: {z0}, {y0}, {x0}")
        print(f"target_buffer.shape: {self.target_buffer.shape}")

        self.target_buffer[:] = final_seg[
                                z0:z0 + self.target_buffer.shape[0],
                                y0:y0 + self.target_buffer.shape[1],
                                x0:x0 + self.target_buffer.shape[2]
                                ]

        # print(f"self.interactions[0] = {self.interactions[0].shape}, pred = {pred.shape}, scaled_bbox = {scaled_bbox}")
        # paste_tensor(self.interactions[0], pred, scaled_bbox)
        #
        # # Paste into full-resolution target_buffer (account crop offset)
        # bbox_full = [[i[0] + bbc[0], i[1] + bbc[0]] for i, bbc in
        #              zip(scaled_bbox, self.preprocessed_props['bbox_used_for_cropping'])]
        #
        # print(f"self.target_buffer = {self.target_buffer.shape}, pred = {pred.shape}, bbox_full = {bbox_full}")
        # paste_tensor(self.target_buffer, pred, bbox_full)

        print("[debug] full image prediction complete")

        if self._viewer is not None:
            if "Prediction" not in self._viewer.layers:
                image_layer = next((layer for layer in self._viewer.layers if layer.__class__.__name__ == "Image"),
                                   None)
                if image_layer is not None:
                    seg_layer = self._viewer.add_labels(final_seg, name="Prediction")
                    if len(image_layer.scale) >= 3:
                        seg_layer.scale = spacing
                        seg_layer.translate = tuple(-o * s for o, s in zip((z0, y0, x0), [1,1,1]))
                else:
                    print("[warning] No image layer found! Could not align prediction.")
            else:
                self._viewer.layers["Prediction"].data = final_seg

            print(f"Prediction complete in {round(time() - start_time, 3)} seconds.")
            print("[debug] pred_seg unique:", np.unique(pred_seg))
            print("[debug] pred_logits.shape =", pred_logits.shape)
            print("[debug] pred_seg.shape =", pred_seg.shape)
            print("[debug] target_buffer.shape =", self.target_buffer.shape)
            return None
        return None

    @torch.inference_mode
    def _predict(self):
        if self.predict_entire_image:
            self._predict_full_image()
            return
        assert self.pad_mode_data == 'constant', 'Only constant padding is supported.'
        assert len(self.new_interaction_centers) == len(self.new_interaction_zoom_out_factors)

        if len(self.new_interaction_centers) == 0:
            print('No patch queued for prediction. Nothing to do.')
            return
        if len(self.new_interaction_centers) > 1:
            print('Multiple interactions since last prediction; only the last one will be used as center!')

        prediction_center = self.new_interaction_centers[-1]
        zoom_out_factor = min(4, self.new_interaction_zoom_out_factors[-1])

        start_predict = time()

        # ===== Initial (possibly zoomed-out) prediction =====
        start_initial = time()
        print("starting build network input")
        inp_cdhw, scaled_patch_size, scaled_bbox = self._build_network_input(prediction_center, zoom_out_factor)
        print(f"we got past build network input. inp_cdhw: {inp_cdhw.shape}, scaled_patch_size: {scaled_patch_size}, scaled_bbox: {scaled_bbox}")
        print("min/max inp_cdhw:", inp_cdhw.min(), inp_cdhw.max())
        print("now trying onnx run")
        logits = self.onnx_session.run(None, {self.input_name: inp_cdhw[None]})[0]  # (1,2,D,H,W)
        pred = logits[0].argmax(0).astype(np.uint8)  # (D,H,W)
        print(f"great, we finished onnx run. logits.shape = {logits.shape}. pred = {pred.shape}")
        print("unique pred values:", np.unique(pred))


        # Compare vs previous prediction (cropped to scaled_bbox)
        prev_crop = crop_and_pad_nd(self.interactions[0], scaled_bbox)  # your existing util
        if list(prev_crop.shape) != list(pred.shape):
            print(f"[debug] prev_crop.shape != pred.shape. prev_crop.shape = {prev_crop.shape}, pred.shape = {pred.shape}")
            # nearest to compare masks
            zf = [pred.shape[0] / prev_crop.shape[0], pred.shape[1] / prev_crop.shape[1],
                  pred.shape[2] / prev_crop.shape[2]]
            prev_crop = zoom(prev_crop, zf, order=0).astype(prev_crop.dtype)

        has_change = self._detect_change_at_border(prev_crop, pred)
        print(f"[debug] has_change = {has_change}")
        print(f'Took {round(time() - start_initial, 3)} s for initial pred at zoom {zoom_out_factor}')

        # ===== Auto-zoom if borders changed =====
        if getattr(self, 'do_autozoom', True):
            growth = 1.5
            while has_change and zoom_out_factor < 4:
                zoom_out_factor = min(4, zoom_out_factor * growth)
                inp_cdhw, scaled_patch_size, scaled_bbox = self._build_network_input(prediction_center, zoom_out_factor)
                logits = self.onnx_session.run(None, {self.input_name: inp_cdhw[None]})[0]
                pred = logits[0].argmax(0).astype(np.uint8)

                prev_crop = crop_and_pad_nd(self.interactions[0], scaled_bbox)
                if list(prev_crop.shape) != list(pred.shape):
                    zf = [pred.shape[0] / prev_crop.shape[0], pred.shape[1] / prev_crop.shape[1],
                          pred.shape[2] / prev_crop.shape[2]]
                    prev_crop = zoom(prev_crop, zf, order=0).astype(prev_crop.dtype)

                has_change = self._detect_change_at_border(prev_crop, pred)
                print(f'AutoZoom -> zoom {zoom_out_factor}, border_change={has_change}')
            print("autozooming done")
        # ===== Place coarse result =====
        # If zoom_out_factor != 1, resize pred to the scaled_patch_size (base size at this zoom)
        scaled_patch_size = scaled_patch_size # u might be asking why? my asnwer is... im lazy
        print(f"scaled patch size was set. it is = {scaled_patch_size}")
        if list(pred.shape) != scaled_patch_size:
            print("[debug] pred.shape != scaled_patch_size")
            zf = [scaled_patch_size[0] / pred.shape[0], scaled_patch_size[1] / pred.shape[1],
                  scaled_patch_size[2] / pred.shape[2]]
            print(f"zf = {zf}")
            pred = zoom(pred, zf, order=0).astype(np.uint8)
            print(f"pred.shape: {pred.shape}")

        # Paste into interactions[0] (coarse seg in preprocessed space)
        print(f"self.interactions[0] = {self.interactions[0].shape}, pred = {pred.shape}, scaled_bbox = {scaled_bbox}")
        paste_tensor(self.interactions[0], pred, scaled_bbox)

        # Paste into full-resolution target_buffer (account crop offset)
        bbox_full = [[i[0] + bbc[0], i[1] + bbc[0]] for i, bbc in
                     zip(scaled_bbox, self.preprocessed_props['bbox_used_for_cropping'])]
        print(f"self.target_buffer = {self.target_buffer.shape}, pred = {pred.shape}, bbox_full = {bbox_full}")
        paste_tensor(self.target_buffer, pred, bbox_full)

        # ===== Optional refinement if zoom_out_factor > 1 =====
        if zoom_out_factor > 1:
            # compute diff map across the full volume, based on the coarse placement
            diff_map = self._compute_diff_map(pred, self.interactions[0], scaled_bbox, scaled_patch_size, dilate_iters=1)
            print(f"[debug] diff_map = {diff_map}")

            # Create a working "prediction_with_coarse" volume: start from current seg channel
            prediction_with_coarse = self.interactions[0].copy()

            # Ensure the coarse pred is present in that working map (already pasted above, but safe)
            paste_tensor(prediction_with_coarse, pred, scaled_bbox)

            # refine on diff regions
            self._refine_coarse(diff_map, prediction_with_coarse)
        else:
            print('No refinement necessary')

        print(f'Done. Total time {round(time() - start_predict, 3)}s')
        self.new_interaction_centers = []
        self.new_interaction_zoom_out_factors = []

    def _build_network_input(self, prediction_center, zoom_out_factor):
        """
        Returns:
          input_for_predict: np.ndarray (C=8, D,H,W) ready for ONNX (you will add batch dim later)
          scaled_patch_size: [pD,pH,pW]
          scaled_bbox: [[z0,z1],[y0,y1],[x0,x1]] in preprocessed space
        """
        # Compute scaled patch and bbox around center
        base_ps = self.configuration_manager.patch_size  # (pD,pH,pW)
        scaled_patch_size = [int(round(p * zoom_out_factor)) for p in base_ps]
        cz, cy, cx = prediction_center
        scaled_bbox = [[cz - s // 2, cz - s // 2 + s] for s, cz in zip(scaled_patch_size, prediction_center)]

        # Crop image + interactions to bbox (allow partial out-of-bounds)
        # crop_to_valid should return (crop, pad_info) for later pad_cropped if you need it.
        img = self.preprocessed_image  # (Cimg,D,H,W) or (D,H,W) -> you already ensured 4D upstream
        itx = self.interactions  # (Cint,D,H,W)

        crop_img, pad_img = crop_to_valid(img, scaled_bbox)  # shape (Cimg, dz,dy,dx)
        crop_itx, pad_itx = crop_to_valid(itx, scaled_bbox)  # shape (Cint, dz,dy,dx)

        target_ps = list(base_ps)

        # Pad cropped regions back to the scaled_patch_size
        crop_img = pad_cropped(crop_img, pad_img)
        crop_itx = pad_cropped(crop_itx, pad_itx)

        # If zoom_out_factor != 1, resample both to base patch size
        if zoom_out_factor != 1:
            zf = [base_ps[0] / inp_cdhw.shape[1],
                  base_ps[1] / inp_cdhw.shape[2],
                  base_ps[2] / inp_cdhw.shape[3]]
            inp_cdhw = zoom(inp_cdhw, [1] + zf, order=1).astype(np.float32)
            scaled_patch_size = base_ps

        def _resample_volume(vol, target_shape, is_label=False):
            # vol: (C, d,h,w) -> resample each channel with ndimage.zoom
            c, d, h, w = vol.shape
            if [d, h, w] == target_shape:
                return vol
            zoom_factors = [target_shape[0] / d, target_shape[1] / h, target_shape[2] / w]
            out = []
            for ch in range(c):
                out.append(zoom(vol[ch], zoom_factors, order=0 if is_label else 1))
            return np.stack(out, axis=0)

        # Preserve interactions during downsample (like the pooling trick)
        if zoom_out_factor != 1:
            # Slight dilation to avoid losing scribbles/points when downsampling
            try:
                for ch in range(min(crop_itx.shape[0], 7)):  # be robust if fewer chans
                    # pool/dilate-ish (you already have iterative_3x3...)
                    crop_itx[ch:ch + 1] = iterative_3x3_same_padding_pool3d(crop_itx[ch:ch + 1][None], kernel_size=3)[0]
            except Exception:
                pass
            crop_img = _resample_volume(crop_img, target_ps, is_label=False)
            crop_itx = _resample_volume(crop_itx, target_ps, is_label=False)
        else:
            # If no resample, but padding needed (edges), finish padding to base patch size
            if any([x for row in pad_itx for x in row]):
                crop_itx = pad_cropped(crop_itx, pad_itx)
            if any([x for row in pad_img for x in row]):
                crop_img = pad_cropped(crop_img, pad_img)

            # If still not exact shape (rare), force resize
            if list(crop_img.shape[1:]) != target_ps:
                crop_img = _resample_volume(crop_img, target_ps, is_label=False)
            if list(crop_itx.shape[1:]) != target_ps:
                crop_itx = _resample_volume(crop_itx, target_ps, is_label=False)

        # Compose 8 channels: concatenate and pad to 8
        input_for_predict = np.concatenate([crop_img, crop_itx], axis=0)  # (Cimg+Cint, D,H,W)
        if input_for_predict.shape[0] < 8:
            pad_c = 8 - input_for_predict.shape[0]
            input_for_predict = np.concatenate(
                [input_for_predict, np.zeros((pad_c, *input_for_predict.shape[1:]), dtype=input_for_predict.dtype)],
                axis=0
            )
        elif input_for_predict.shape[0] > 8:
            input_for_predict = input_for_predict[:8]

        return input_for_predict.astype(np.float32), list(base_ps), scaled_bbox

    def _refine_coarse(self, diff_map: np.ndarray, prediction_with_coarse: np.ndarray):
        """
        Use diff_map to generate refinement bboxes and update self.interactions[0] and self.target_buffer.
        """
        bboxes_ordered = generate_bounding_boxes(
            diff_map, self.configuration_manager.patch_size,
            stride='auto', margin=(10, 10, 10), max_depth=3
        )
        if len(bboxes_ordered) == 0:
            # Fallback: refine at the interaction center
            center = self.new_interaction_centers[-1]
            ps = self.configuration_manager.patch_size
            bboxes_ordered = [[[c - p // 2, c - p // 2 + p] for c, p in zip(center, ps)]]

        # Preallocate channel tensor
        ps = self.configuration_manager.patch_size
        prealloc = np.zeros((8, *ps), dtype=np.float32)

        # Channel layout (robust to fewer chans):
        # ch0: image[0], ch1: prediction_with_coarse, ch2..: interactions[1:]
        for refinement_bbox in bboxes_ordered:
            prealloc.fill(0)

            crop_and_pad_into_buffer(prealloc[0], refinement_bbox, self.preprocessed_image[0])
            crop_and_pad_into_buffer(prealloc[1], refinement_bbox, prediction_with_coarse)

            # Fill remaining channels from interactions[1:]
            itx_rest = self.interactions[1:]
            used = min(6, itx_rest.shape[0])  # leave total at 8 chans
            for k in range(used):
                crop_and_pad_into_buffer(prealloc[2 + k], refinement_bbox, itx_rest[k])

            # ONNX run
            logits = self.onnx_session.run(None, {self.input_name: prealloc[None]})[0]  # (1,2,D,H,W)
            pred = logits[0].argmax(0).astype(np.uint8)  # (D,H,W)

            # Paste into working coarse map and into full target buffer (with bbox offset)
            paste_tensor(self.interactions[0], pred, refinement_bbox)

            # account for crop offset from preprocessing
            bbox_full = [[i[0] + bbc[0], i[1] + bbc[0]] for i, bbc in
                         zip(refinement_bbox, self.preprocessed_props['bbox_used_for_cropping'])]
            paste_tensor(self.target_buffer, pred, bbox_full)

    def _detect_change_at_border(self,
                                    prev_seg_patch: np.ndarray,
                                    new_seg_patch: np.ndarray,
                                    border_width: int = 1) -> bool:
        """True if segmentations differ along the patch borders."""
        assert prev_seg_patch.shape == new_seg_patch.shape
        D, H, W = prev_seg_patch.shape
        borders = [
            (slice(0, border_width), slice(None), slice(None)),  # z-min
            (slice(D - border_width, D), slice(None), slice(None)),  # z-max
            (slice(None), slice(0, border_width), slice(None)),  # y-min
            (slice(None), slice(H - border_width, H), slice(None)),  # y-max
            (slice(None), slice(None), slice(0, border_width)),  # x-min
            (slice(None), slice(None), slice(W - border_width, W)),  # x-max
        ]
        for sl in borders:
            if not np.array_equal(prev_seg_patch[sl], new_seg_patch[sl]):
                return True
        return False

    def _compute_diff_map(self,
                             pred_patch: np.ndarray,
                             prev_full_seg: np.ndarray,
                             scaled_bbox: list,
                             scaled_patch_size: list,
                             dilate_iters: int = 1) -> np.ndarray:
        """
        Build a full-volume binary diff map from the patch prediction vs the corresponding
        crop of prev_full_seg (self.interactions[0]).
        """
        # Where in the full volume does this patch live?
        Df, Hf, Wf = prev_full_seg.shape
        (z0, z1), (y0, y1), (x0, x1) = scaled_bbox
        z0v, y0v, x0v = max(0, z0), max(0, y0), max(0, x0)
        z1v, y1v, x1v = min(z1, Df), min(y1, Hf), min(x1, Wf)

        diff_map = np.zeros_like(prev_full_seg, dtype=np.uint8)
        if z0v >= z1v or y0v >= y1v or x0v >= x1v:
            return diff_map  # bbox outside — nothing to do

        # Crop previous seg to the visible bbox
        prev_crop = prev_full_seg[z0v:z1v, y0v:y1v, x0v:x1v]

        # If pred_patch size doesn’t match the visible area (edges), trim it
        dz, dy, dx = prev_crop.shape
        pred_vis = pred_patch[:dz, :dy, :dx]

        # Compute differences
        diff_local = (pred_vis != prev_crop).astype(np.uint8)
        diff_map[z0v:z1v, y0v:y1v, x0v:x1v] = diff_local

        # (Optional) thicken regions to get fewer, more stable refinement boxes
        if dilate_iters > 0:
            struct = generate_binary_structure(3, 1)
            diff_map = binary_dilation(diff_map, structure=struct, iterations=dilate_iters).astype(np.uint8)

        # Light 3x3 open/close (if you have your pooled ops, they’re great to use here)
        try:
            dm = diff_map.astype(np.float32)
            dm = iterative_3x3_same_padding_pool3d(dm[None, None], kernel_size=5, use_min_pool=True)[0, 0]
            dm = iterative_3x3_same_padding_pool3d(dm[None, None], kernel_size=5, use_min_pool=False)[0, 0]
            diff_map = (dm > 0.5).astype(np.uint8)
        except Exception:
            pass

        return diff_map

    def _add_patch_for_point_interaction(self, coordinates):
        self.new_interaction_zoom_out_factors.append(1)
        self.new_interaction_centers.append(coordinates)
        print(f'Added new point interaction: center {self.new_interaction_zoom_out_factors[-1]}, scale {self.new_interaction_centers}')

    def _add_patch_for_bbox_interaction(self, bbox):
        bbox_center = [round((i[0] + i[1]) / 2) for i in bbox]
        bbox_size = [i[1]-i[0] for i in bbox]
        # we want to see some context, so the crop we see for the initial prediction should be patch_size / 3 larger
        requested_size = [i + j // 3 for i, j in zip(bbox_size, self.configuration_manager.patch_size)]
        self.new_interaction_zoom_out_factors.append(max(1, max([i / j for i, j in zip(requested_size, self.configuration_manager.patch_size)])))
        self.new_interaction_centers.append(bbox_center)
        # maybe im stupid... but why is it printing zoom out factors?
        # this is old code: print(f'Added new bbox interaction: center {self.new_interaction_zoom_out_factors[-1]}, scale {self.new_interaction_centers}')
        print(f'Added new bbox interaction: center {self.bbox_center}, scale {self.new_interaction_centers}')



    def _add_patch_for_scribble_interaction(self, scribble_image):
        return self._generic_add_patch_from_image(scribble_image)

    def _add_patch_for_lasso_interaction(self, lasso_image):
        return self._generic_add_patch_from_image(lasso_image)

    def _add_patch_for_initial_seg_interaction(self, initial_seg):
        return self._generic_add_patch_from_image(initial_seg)

    def _generic_add_patch_from_image(self, image: torch.Tensor):
        # if not torch.any(image):
        #     print('Received empty image prompt. Cannot add patches for prediction')
        #     return
        nonzero_indices = np.array(np.nonzero(image)).T
        mn = np.min(nonzero_indices, dim=0)[0]
        mx = np.max(nonzero_indices, dim=0)[0]
        roi = [[i.item(), x.item() + 1] for i, x in zip(mn, mx)]
        roi_center = [round((i[0] + i[1]) / 2) for i in roi]
        roi_size = [i[1]- i[0] for i in roi]
        requested_size = [i + j // 3 for i, j in zip(roi_size, self.configuration_manager.patch_size)]
        self.new_interaction_zoom_out_factors.append(max(1, max([i / j for i, j in zip(requested_size, self.configuration_manager.patch_size)])))
        self.new_interaction_centers.append(roi_center)
        print(f'Added new image interaction: scale {self.new_interaction_zoom_out_factors[-1]}, center {self.new_interaction_centers}')

    def initialize_from_trained_model_folder(self, model_training_output_dir: str,
                                             use_fold: Union[int, str] = None,
                                             checkpoint_name: str = 'checkpoint_final.pth'):
        print("[debug] initialize_from_trained_model_folder called. Model is: ", model_training_output_dir)
        model_training_output_dir = str(model_training_output_dir)
        """
        This is used when making predictions with a trained model
        """
        # load trainer specific settings

        if model_training_output_dir.endswith(".onnx"):
            self.initialize_from_onnx_model(model_training_output_dir)
            return

        expected_json_file = join(model_training_output_dir, 'inference_session_class.json')
        json_content = load_json(expected_json_file)
        if isinstance(json_content, str):
            point_interaction_radius = json_content['point_radius']
            self.preferred_scribble_thickness = json_content['preferred_scribble_thickness']
            if not isinstance(self.preferred_scribble_thickness, (tuple, list)):
                self.preferred_scribble_thickness = [self.preferred_scribble_thickness] * 3
            self.interaction_decay = json_content['interaction_decay'] if 'interaction_decay' in json_content.keys() else 0.9
            point_interaction_use_etd = True # so far this is not defined in that file so we stick with default
            self.point_interaction = PointInteraction_stub(point_interaction_radius, point_interaction_use_etd)

        else:
            # padding mode for data. See nnInteractiveTrainerV2_nodelete_reflectpad
            self.pad_mode_data = json_content['pad_mode_image'] if 'pad_mode_image' in json_content.keys() else "constant"
            # ... you are probably gonna have to change this
            # old convention where we only specified the inference class in this file. Set defaults for stuff
            point_interaction_radius = 4
            point_interaction_use_etd = True
            self.point_interaction = PointInteraction_stub(
                point_interaction_radius,
                point_interaction_use_etd)
            self.pad_mode_data = "constant"
            self.interaction_decay = 0.9
        dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
        plans = load_json(join(model_training_output_dir, 'plans.json'))
        plans_manager = PlansManager(plans)

        if use_fold is not None:
            use_fold = int(use_fold) if use_fold != 'all' else use_fold
            fold_folder = f'fold_{use_fold}'
        else:
            fldrs = subdirs(model_training_output_dir, prefix='fold_', join=False)
            assert len(fldrs) == 1, f'Attempted to infer fold but there is != 1 fold_ folders: {fldrs}'
            fold_folder = fldrs[0]

        checkpoint = torch.load(join(model_training_output_dir, fold_folder, checkpoint_name),
                                map_location=self.device, weights_only=False)
        trainer_name = checkpoint['trainer_name']
        configuration_name = checkpoint['init_args']['configuration']

        parameters = checkpoint['network_weights']

        configuration_manager = plans_manager.get_configuration(configuration_name)
        # restore network
        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        trainer_class = recursive_find_python_class(join(nnInteractive.__path__[0], "trainer"),
                                                    trainer_name, 'nnInteractive.trainer')
        if trainer_class is None:
            print(f'Unable to locate trainer class {trainer_name} in nnInteractive.trainer. '
                               f'Please place it there (in any .py file)!')
            print('Attempting to use default nnInteractiveTrainer_stub. If you encounter errors, this is where you need to look!')
            trainer_class = nnInteractiveTrainer_stub

        network = trainer_class.build_network_architecture(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            enable_deep_supervision=False
        ).to(self.device)
        network.load_state_dict(parameters)

        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.network = network
        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.label_manager = plans_manager.get_label_manager(dataset_json)
        if self.use_torch_compile and not isinstance(self.network, OptimizedModule):
            print('Using torch.compile')
            self.network = torch.compile(self.network)

    def initialize_from_onnx_model(self, model_path: str):
        # stub values for onnx

        global onnx_session_initialized

        print("[debug] initialize_from_onnx_model HAS BEEN CALLED!")
        point_radius = 4
        self.pad_mode_data = "constant"
        self.point_interaction = PointInteraction_stub(
            point_radius,
            use_etd=True
        )
        self.interaction_decay = 0.9
        # Dummy or default configuration
        # wahh! i commented this all out! i am too lazy to find the actual dummies for these.
        # self.configuration_manager = configuration_manager
        # dummy_plans = PlansManager.default_plans()
        # dummy_labels = dummy_plans.get_label_manager({})

        # self.plans_manager = dummy_plans
        # self.label_manager = dummy_labels
        # self.dataset_json = {}

        # Run ONNX session
        import onnxruntime as ort
        self.onnx_session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


        for inp in self.onnx_session.get_inputs():
            print("Input name:", inp.name)
            print("Shape:", inp.shape)
            print("Type:", inp.type)

        self.network = ONNXWrapper(self.onnx_session)
        print("ONNX model loaded successfully.")


    def manual_initialization(self, network: nn.Module, plans_manager: PlansManager,
                              configuration_manager: ConfigurationManager, dataset_json: dict, trainer_name: str):
        """
        This is used by the nnUNetTrainer to initialize nnUNetPredictor for the final validation
        """

        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.network = network
        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.label_manager = plans_manager.get_label_manager(dataset_json)

        if self.use_torch_compile and not isinstance(self.network, OptimizedModule):
            print('Using torch.compile')
            self.network = torch.compile(self.network)

        if not self.use_torch_compile and isinstance(self.network, OptimizedModule):
            self.network = self.network._orig_mod

        self.network = self.network.to(self.device)

def transform_coordinates_noresampling(
        coords_orig: Union[List[int], Tuple[int, ...]],
        nnunet_preprocessing_crop_bbox: List[Tuple[int, int]]
) -> Tuple[int, ...]:
    print("[debug] transform_coordinates_noresampling has been called")
    """
    converts coordinates in the original uncropped image to the internal cropped representation. Man I really hate
    nnU-Net's crop to nonzero!
    """
    return tuple([coords_orig[d] - nnunet_preprocessing_crop_bbox[d][0] for d in range(len(coords_orig))])

def resize_np(arr: np.ndarray, new_shape, order=1):
    print("[debug] resize_np was called")
    """
    Resize a 3D or 4D NumPy array to the given shape using scipy.ndimage.zoom.

    Args:
        arr (np.ndarray): The array to resize. Shape (C, D, H, W) or (D, H, W)
        new_shape (tuple): The desired spatial shape (D, H, W)
        order (int): Interpolation order: 0 = nearest, 1 r= trilinear/linear

    Returns:
        np.ndarray: Resized array of shape (C, D, H, W) or (D, H, W)
    """
    if arr.ndim == 4:
        # Multi-channel (C, D, H, W)
        channels = []
        for c in arr:
            zoom_factors = [n / o for n, o in zip(new_shape, c.shape)]
            resized = zoom(c, zoom_factors, order=order)
            channels.append(resized)
        return np.stack(channels, axis=0)
    elif arr.ndim == 3:
        # Single-channel (D, H, W)
        zoom_factors = [n / o for n, o in zip(new_shape, arr.shape)]
        return zoom(arr, zoom_factors, order=order)
    else:
        raise ValueError(f"Unsupported array shape: {arr.shape}")


class SimpleONNXConfig:
    print("[debug] Class SimpleONNXConfig has been called")
    def __init__(self, patch_size=(64, 64, 64)):
        self.patch_size = list(patch_size)

class ONNXWrapper:
    def __init__(self, session):
        self.session = session

    def __call__(self, input_tensor: torch.Tensor):
        input_np = input_tensor.cpu().numpy()
        outputs = self.session.run(None, {"input": input_np})  # you may need to check the input name
        return torch.from_numpy(outputs[0])  # Adjust based on actual output shape

if __name__ == "__main__":
    print("Running the file now!!")

    session = nnInteractiveInferenceSession()
    print(f"Session type: {type(session)}")

    print(hasattr(session, 'add_point_interaction'))
