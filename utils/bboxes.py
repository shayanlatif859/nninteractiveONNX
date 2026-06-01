from time import time
from time import time
from typing import List, Union, Tuple

import numpy as np
import torch


from time import time
from typing import List, Union, Tuple

import numpy as np
import torch


def generate_bounding_boxes(
        mask: torch.Tensor,
        bbox_size=(192, 192, 192),
        stride: Union[List[int], Tuple[int, int, int], str] = (16, 16, 16),
        margin=(10, 10, 10),
        max_depth: int = 5,
        current_depth: int = 0
) -> List:
    """
    Generate a covering set of bounding boxes over a 3D binary mask using a
    greedy set-cover algorithm.

    Parameters
    ----------
    mask : torch.Tensor
        3D binary tensor (uint8 or bool). Non-zero voxels are the target region.
    bbox_size : tuple of int
        Size of each bounding box in (x, y, z).
    stride : tuple of int or 'auto'
        Spacing between candidate centers. 'auto' derives stride from object extent.
    margin : tuple of int
        Coverage margin — voxels within this margin of the box edge are not marked
        as covered, ensuring meaningful overlap between adjacent boxes.
    max_depth : int
        Maximum recursion depth for covering residual voxels.
    current_depth : int
        Current recursion depth (used internally).

    Returns
    -------
    List of [[x0,x1],[y0,y1],[z0,z1]] bounding boxes as half-open intervals.
    """
    # Early exit if mask is empty.
    if not torch.any(mask):
        return []

    # Prevent infinite recursion.
    if current_depth > max_depth:
        return random_sampling_fallback(mask, bbox_size, margin, n_samples=25)

    bbox_size = list(bbox_size)
    margin = list(margin)

    half_size  = [bs // 2 for bs in bbox_size]
    end_offset = [bs - hs for bs, hs in zip(bbox_size, half_size)]

    object_voxels = torch.nonzero(mask, as_tuple=False)
    if object_voxels.numel() == 0:
        return []

    min_coords = object_voxels.min(dim=0)[0]
    max_coords = object_voxels.max(dim=0)[0]

    if isinstance(stride, str) and stride == 'auto':
        stride = [
            max(1, round((j.item() - i.item()) / 4))
            for i, j in zip(min_coords, max_coords)
        ]
    stride = list(stride)

    # Build candidate centers: grid points inside the mask.
    potential_centers = []
    for x in range(min_coords[0].item(), min(mask.shape[0], max_coords[0].item() + 1), stride[0]):
        for y in range(min_coords[1].item(), min(mask.shape[1], max_coords[1].item() + 1), stride[1]):
            for z in range(min_coords[2].item(), min(mask.shape[2], max_coords[2].item() + 1), stride[2]):
                if mask[x, y, z]:
                    potential_centers.append([x, y, z])

    if len(potential_centers) == 0:
        return generate_bounding_boxes(
            mask, bbox_size,
            [max(1, s // 2) for s in stride],
            margin, max_depth, current_depth + 1
        )

    # Keep as tensor for vectorised center pruning.
    potential_centers = torch.tensor(potential_centers, device=mask.device)
    uncovered = mask.clone().byte()
    bboxes = []

    # Greedy set-cover: evaluate ALL candidates, commit the best, repeat.
    while len(potential_centers) > 0 and uncovered.any():
        best_idx     = None   # integer index into potential_centers
        best_covered = 0
        best_bounds  = None

        for idx in range(len(potential_centers)):
            center = potential_centers[idx]
            c_x, c_y, c_z = center[0].item(), center[1].item(), center[2].item()

            x_start = max(0,             c_x - half_size[0] + margin[0])
            x_end   = min(mask.shape[0], c_x + end_offset[0] - margin[0])
            y_start = max(0,             c_y - half_size[1] + margin[1])
            y_end   = min(mask.shape[1], c_y + end_offset[1] - margin[1])
            z_start = max(0,             c_z - half_size[2] + margin[2])
            z_end   = min(mask.shape[2], c_z + end_offset[2] - margin[2])

            num_covered = uncovered[x_start:x_end, y_start:y_end, z_start:z_end].sum().item()

            if num_covered > best_covered:
                best_covered = num_covered
                best_idx     = idx   # integer — used to index potential_centers tensor
                best_bounds  = (x_start, x_end, y_start, y_end, z_start, z_end)

        # No candidate covers any uncovered voxel — done.
        if best_covered == 0 or best_idx is None:
            break

        # Commit the best box — extract coordinate from tensor by integer index.
        c_x, c_y, c_z = [i.item() for i in potential_centers[best_idx]]
        bboxes.append([
            [c_x - half_size[0], c_x + end_offset[0]],
            [c_y - half_size[1], c_y + end_offset[1]],
            [c_z - half_size[2], c_z + end_offset[2]],
        ])

        # Mark coverage region.
        x_s, x_e, y_s, y_e, z_s, z_e = best_bounds
        uncovered[x_s:x_e, y_s:y_e, z_s:z_e] = 0

        # Prune centers whose voxel is now covered — vectorised via tensor indexing.
        potential_centers = potential_centers[
            uncovered[tuple(potential_centers.T)] > 0
        ]

    # Cover any residual voxels.
    if uncovered.any():
        residual = uncovered.sum().item()
        small_residual = residual < np.prod([i // 3 for i in bbox_size])

        if small_residual:
            bboxes.extend(
                random_sampling_fallback(uncovered, bbox_size, margin, n_samples=25)
            )
        else:
            bboxes.extend(
                generate_bounding_boxes(
                    uncovered, bbox_size,
                    [max(1, s // 2) for s in stride],
                    margin, max_depth, current_depth + 1
                )
            )

    return bboxes


def random_sampling_fallback(
        mask: torch.Tensor,
        bbox_size=(192, 192, 192),
        margin=(10, 10, 10),
        n_samples: int = 25
) -> List:
    """
    Cover remaining mask voxels by randomly sampling candidate centers and
    greedily picking the one that covers the most uncovered voxels.

    Used as a fallback for small residual regions after the main greedy pass,
    or when max recursion depth is exceeded.

    Parameters
    ----------
    mask : torch.Tensor
        3D binary tensor of uncovered voxels. Modified in place.
    bbox_size : tuple of int
        Size of each bounding box.
    margin : tuple of int
        Coverage margin (same semantics as generate_bounding_boxes).
    n_samples : int
        Number of random candidates to evaluate per iteration.

    Returns
    -------
    List of [[x0,x1],[y0,y1],[z0,z1]] bounding boxes.
    """
    half_size  = [bs // 2 for bs in bbox_size]
    end_offset = [bs - hs for bs, hs in zip(bbox_size, half_size)]
    bboxes = []

    while mask.any():
        indices = torch.nonzero(mask, as_tuple=False)  # (N, 3)

        best_center  = None
        best_covered = 0
        best_bounds  = None

        # Sample without replacement to avoid evaluating the same voxel twice.
        sample_size = min(n_samples, len(indices))
        sample_idxs = torch.randperm(len(indices))[:sample_size]

        for idx in sample_idxs:
            center = indices[idx]
            c_x, c_y, c_z = center[0].item(), center[1].item(), center[2].item()

            x_start = max(0,             c_x - half_size[0] + margin[0])
            x_end   = min(mask.shape[0], c_x + end_offset[0] - margin[0])
            y_start = max(0,             c_y - half_size[1] + margin[1])
            y_end   = min(mask.shape[1], c_y + end_offset[1] - margin[1])
            z_start = max(0,             c_z - half_size[2] + margin[2])
            z_end   = min(mask.shape[2], c_z + end_offset[2] - margin[2])

            num_covered = mask[x_start:x_end, y_start:y_end, z_start:z_end].sum().item()

            if num_covered > best_covered:
                best_covered = num_covered
                best_center  = center   # coordinate tensor row, not an integer index
                best_bounds  = (x_start, x_end, y_start, y_end, z_start, z_end)

        # Guard: if nothing was covered (e.g. all candidates had zero coverage), stop.
        if best_center is None or best_covered == 0:
            break

        c_x, c_y, c_z = best_center[0].item(), best_center[1].item(), best_center[2].item()
        bboxes.append([
            [c_x - half_size[0], c_x + end_offset[0]],
            [c_y - half_size[1], c_y + end_offset[1]],
            [c_z - half_size[2], c_z + end_offset[2]],
        ])

        x_s, x_e, y_s, y_e, z_s, z_e = best_bounds
        mask[x_s:x_e, y_s:y_e, z_s:z_e] = 0

    return bboxes


if __name__ == '__main__':
    times = []
    torch.set_num_threads(8)

    for _ in range(3):
        st = time()
        mask = torch.zeros((256, 256, 256), dtype=torch.uint8)
        mask[50:150, 50:150, 50:150] = 1

        bboxes = generate_bounding_boxes(
            mask,
            bbox_size=(192, 192, 192),
            stride=(16, 16, 16),
            margin=(10, 10, 10),
        )
        print(f"Number of bounding boxes: {len(bboxes)}")
        times.append(time() - st)

    print(f"Times: {times}")
    print(f"Mean: {sum(times)/len(times):.3f}s")