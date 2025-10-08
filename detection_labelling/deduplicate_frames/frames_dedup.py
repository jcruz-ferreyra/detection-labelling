from collections import Counter
from functools import reduce
import json
import logging
import os
from pathlib import Path
import shutil
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv

from .types import FramesDeduplicationContext

logger = logging.getLogger(__name__)
cv2.setRNGSeed(42)


def _initialize_sift_flann(ctx: FramesDeduplicationContext) -> None:
    """Initialize SIFT feature detector and FLANN-based matcher. Add them to context"""
    logger.info("Initializing SIFT and FLANN matchers")

    ctx.sift = cv2.SIFT_create()
    ctx.flann = cv2.FlannBasedMatcher(ctx.flann_index_params, ctx.flann_search_params)

    logger.debug("SIFT and FLANN matchers initialized successfully")


def _get_video_ids_with_files(ctx: FramesDeduplicationContext) -> Dict[str, List[str]]:
    """Extract unique video IDs with their corresponding filenames."""
    logger.info("Extracting unique video IDs and filenames")

    images_dir = ctx.frames_dir / "images"
    video_files = {}

    for img_path in images_dir.iterdir():
        if img_path.is_file() and img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            parts = img_path.stem.split("_")

            if len(parts) >= 2:
                video_id = "_".join(parts[:-1])
                filename_stem = img_path.stem

                if video_id not in video_files:
                    video_files[video_id] = []
                video_files[video_id].append(filename_stem)

    logger.info(f"Found {len(video_files)} unique video IDs to process")
    for video_id, files in video_files.items():
        logger.debug(f"Video {video_id}: {len(files)} files")

    return video_files


def _move_data_to_temp_dir(
    ctx: FramesDeduplicationContext, video_id: str, filenames: List[str]
) -> None:
    """Move video's images and annotations to temp directories."""
    logger.info(f"Moving {len(filenames)} files for video {video_id} to temp directories")

    # Create temp directories
    temp_images_dir = ctx.frames_dir / "images" / "temp"
    temp_annotations_dir = ctx.frames_dir / "annotations_oob" / "temp"

    temp_images_dir.mkdir(exist_ok=True)
    temp_annotations_dir.mkdir(exist_ok=True)

    # Move files
    moved_images = 0
    moved_annotations = 0

    for filename_stem in filenames:
        for ext in [".png", ".jpg", ".jpeg"]:
            src_img = ctx.frames_dir / "images" / f"{filename_stem}{ext}"
            if src_img.exists():
                dst_img = temp_images_dir / f"{filename_stem}{ext}"
                shutil.move(str(src_img), str(dst_img))
                moved_images += 1
                break

        # Move annotation file
        src_ann = ctx.frames_dir / "annotations_oob" / f"{filename_stem}.xml"
        if src_ann.exists():
            dst_ann = temp_annotations_dir / f"{filename_stem}.xml"
            shutil.move(str(src_ann), str(dst_ann))
            moved_annotations += 1

    logger.info(
        f"Moved {moved_images} images and {moved_annotations} annotations to temp directories"
    )


def _load_dataset_as_supervision(ctx: FramesDeduplicationContext) -> sv.DetectionDataset:
    """Load the dataset using Supervision from Pascal VOC format."""
    logger.info(f"Loading dataset from: {ctx.frames_dir}")

    try:
        ds = sv.DetectionDataset.from_pascal_voc(
            images_directory_path=ctx.frames_dir / "images" / "temp",
            annotations_directory_path=ctx.frames_dir / "annotations_oob" / "temp",
        )
        logger.info(f"Loaded dataset with {len(ds)} samples")
        return ds

    except Exception as e:
        logger.error(f"Failed to load dataset from {ctx.frames_dir}: {e}")
        raise


def _get_polygons_from_json(
    ctx: FramesDeduplicationContext,
) -> Dict[str, list[sv.PolygonZone]]:
    """Load parking polygons from a JSON file and create PolygonZone objects per camera."""
    logger.info(f"Loading parking zone polygons from: {ctx.polygons_json_path}")

    try:
        with open(ctx.polygons_json_path, "r") as f:
            camera_coords = json.load(f)
        logger.info(f"Successfully loaded polygons from {ctx.polygons_json_path}")
    except FileNotFoundError:
        logger.error(f"Polygons file not found at {ctx.polygons_json_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON polygons file: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error reading polygons file: {e}")
        raise

    camera_polygons = {}
    anchors = [sv.Position.CENTER]

    for camera, coords_list in camera_coords.items():
        polygons_list = [
            sv.PolygonZone(polygon=np.array(coords), triggering_anchors=anchors)
            for coords in coords_list
        ]

        camera_polygons[camera] = polygons_list

    logger.info(f"Created polygon zones for {len(camera_polygons)} cameras")
    return camera_polygons


def _get_polygons_for_filename(
    filename_stem: str, camera_polygons: Dict["str", List[sv.PolygonZone]]
) -> Optional[List[sv.PolygonZone]]:
    logger.debug(f"Retrieving polygon for: {filename_stem}")

    parts = filename_stem.split("_")

    if len(parts) < 4:
        logger.error(f"Filename stem '{filename_stem}' does not contain expected parts.")
        return None

    camera_id = "_".join(parts[:3])
    date = parts[3]

    # Try camera_id first, then camera_id_date
    if camera_id in camera_polygons:
        return camera_polygons[camera_id]

    camera_id_date = f"{camera_id}_{date}"
    if camera_id_date in camera_polygons:
        return camera_polygons[camera_id_date]

    logger.debug(f"No polygons found for '{camera_id}' or '{camera_id_date}'")
    return None


def _get_mask_from_annotations(
    width: int, height: int, annotations: Dict[str, sv.Detections]
) -> np.ndarray:
    """Create a binary mask from bounding box annotations."""
    mask = np.zeros((height, width), dtype=np.uint8)

    # If no annotations, return full white mask (no masking)
    if not hasattr(annotations, "xyxy") or len(annotations.xyxy) == 0:
        mask[:, :] = 255
        return mask

    # Create mask from bounding boxes
    for x1, y1, x2, y2 in annotations.xyxy.astype(int):
        # Clamp coordinates to image bounds
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, width), min(y2, height)

        # Only fill valid boxes
        if x1 < x2 and y1 < y2:
            mask[y1:y2, x1:x2] = 255
        else:
            logger.warning(f"Invalid box ({x1}, {y1}, {x2}, {y2}) skipped in mask creation.")

    return mask


def _calculate_kp_and_descriptors(
    ctx: FramesDeduplicationContext,
    ds: sv.DetectionDataset,
    camera_polygons: Dict[str, List[sv.PolygonZone]],
) -> List[dict]:
    """Compute SIFT keypoints and descriptors for non-parked four-wheeler detections."""
    logger.info("Calculating keypoints and descriptors for all images")

    # Get four-wheeler class indices
    four_wheelers_ids = [idx for idx, v in enumerate(ds.classes) if v in ["car", "bus", "truck"]]
    logger.info(
        f"Found {len(four_wheelers_ids)} four-wheeler classes: {[ds.classes[i] for i in four_wheelers_ids]}"
    )

    features_list = []

    for img_path, img, ann in ds:
        # Get parking polygons for current image
        filename_stem = Path(img_path).stem
        parking_polygons = _get_polygons_for_filename(filename_stem, camera_polygons)

        if parking_polygons:
            # Filter out parked vehicles
            is_four_wheeler = np.isin(ann.class_id, four_wheelers_ids)
            is_in_parking_zones = [zone.trigger(detections=ann) for zone in parking_polygons]
            is_parked = reduce(lambda x, y: x | y, is_in_parking_zones)

            # Keep only non-parked four-wheelers
            ann = ann[is_four_wheeler & ~is_parked]
        else:
            # No parking polygons - keep all four-wheelers
            is_four_wheeler = np.isin(ann.class_id, four_wheelers_ids)
            ann = ann[is_four_wheeler]

        # Calculate SIFT features
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = _get_mask_from_annotations(img.shape[1], img.shape[0], ann)

        try:
            kp, desc = ctx.sift.detectAndCompute(img_gray, mask)
            if desc is None:
                desc = np.array([])
        except cv2.error as e:
            logger.warning(f"SIFT computation failed for {filename_stem}: {e}")
            kp, desc = [], np.array([])
        except Exception as e:
            logger.error(f"Unexpected error computing SIFT for {filename_stem}: {e}")
            kp, desc = [], np.array([])

        features_list.append({"kp": kp, "desc": desc})

    logger.info(f"Calculated features for {len(features_list)} images")
    return features_list


def _calculate_dup_score(
    ctx: FramesDeduplicationContext, features: List[dict], idx_a: int, idx_b: int
) -> float:
    """Compute duplication score between two frames using SIFT descriptors and FLANN matching."""
    kp_a, desc_a = features[idx_a]["kp"], features[idx_a]["desc"]
    kp_b, desc_b = features[idx_b]["kp"], features[idx_b]["desc"]

    if desc_a is None or desc_b is None or len(desc_a) < 2 or len(desc_b) < 2:
        return 0.0

    try:
        matches = ctx.flann.knnMatch(desc_a, desc_b, k=2)

        # Filter good matches using Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        # Calculate similarity score
        matches_count = len(good_matches)
        max_keypoints = max(len(kp_a), len(kp_b))
        return matches_count / max_keypoints

    except cv2.error as e:
        logger.debug(f"FLANN matching failed for frames {idx_a}-{idx_b}: {e}")
        return 0.0
    except Exception as e:
        logger.warning(f"Unexpected error in duplication score calculation: {e}")
        return 0.0


def _check_large_streaks(
    ctx: FramesDeduplicationContext,
    is_repeated: np.ndarray,
    streak_idxs: Tuple[int, int],
    features: List[dict],
) -> np.ndarray:
    """Recursively check and mark distinct frames in large repeated streaks."""
    pre_idx, post_idx = streak_idxs
    subset_start = pre_idx + 5
    subset_end = post_idx - 5

    if subset_start >= subset_end:
        logger.debug(f"Streak too small to subdivide between indices {pre_idx} and {post_idx}")
        return is_repeated

    for idx in range(subset_start, subset_end + 1):
        # Check similarity to first frame after repeated streak
        score_vs_post = _calculate_dup_score(ctx, features, idx, post_idx)
        if score_vs_post > ctx.is_repeated_threshold:
            # Frame is similar to post - all subsequent frames likely similar too
            logger.debug(f"Frame {idx} similar to post frame {post_idx}, stopping search")
            break

        # Check similarity to last frame before the repeated streak
        score_vs_pre = _calculate_dup_score(ctx, features, idx, pre_idx)
        if score_vs_pre < ctx.is_repeated_threshold:
            # Frame is distinct from both boundaries
            is_repeated[idx] = False
            logger.debug(f"Marked frame {idx} as distinct within streak")

            # Recursively check remaining streak if large enough
            remaining_streak_size = post_idx - idx
            if remaining_streak_size >= 10:
                _check_large_streaks(ctx, is_repeated, (idx, post_idx), features)

            break

    return is_repeated


def _check_if_is_repeated(
    ctx: FramesDeduplicationContext, features_list: List[dict]
) -> np.ndarray:
    """Detect repeated frames using frame descriptors and FLANN matcher."""
    logger.info("Checking for repeated frames")

    list_scores = [0.0]  # First frame has no previous frame to compare

    # Calculate duplication scores
    for i in range(1, len(features_list)):  # Start from 1 to compare with previous frame
        score = _calculate_dup_score(ctx, features_list, i, i - 1)
        list_scores.append(score)

    is_repeated = np.array(list_scores) > ctx.is_repeated_threshold

    # Identify and process large streaks of repeated frames
    distinct_frame_idxs = np.where(~is_repeated)[0]
    for i in range(len(distinct_frame_idxs) - 1):
        streak_start_idx = distinct_frame_idxs[i]
        streak_end_idx = distinct_frame_idxs[i + 1]
        streak_size = streak_end_idx - streak_start_idx

        if streak_size >= 10:
            is_repeated = _check_large_streaks(
                ctx, is_repeated, (streak_start_idx, streak_end_idx), features_list
            )

    repeated_count = sum(is_repeated)
    logger.info(f"Found {repeated_count} repeated frames out of {len(is_repeated)} total")
    return is_repeated


def _move_repeated_frames(
    ctx: FramesDeduplicationContext, ds: sv.DetectionDataset, is_repeated: np.ndarray
) -> None:
    """Move repeated frames to a separate directory."""
    repeated_count = np.sum(is_repeated)
    logger.info(f"Moving {repeated_count} repeated frames to archive directory")

    target_dir = ctx.frames_dir / "repeated"
    (target_dir / "images").mkdir(parents=True, exist_ok=True)
    (target_dir / "annotations_oob").mkdir(parents=True, exist_ok=True)

    # Temp directories
    temp_images_dir = ctx.frames_dir / "images" / "temp"
    temp_annotations_dir = ctx.frames_dir / "annotations_oob" / "temp"

    repeated_idxs = np.where(is_repeated)[0]
    failed_moves = []

    for idx in repeated_idxs:
        file_path = Path(ds.image_paths[idx])
        filename_stem = file_path.stem

        # Move image file
        src_img = temp_images_dir / file_path.name
        dst_img = target_dir / "images" / file_path.name

        try:
            shutil.move(str(src_img), str(dst_img))
        except Exception as e:
            logger.warning(f"Failed to move image {file_path.name}: {e}")
            failed_moves.append(file_path.name)
            continue

        # Move annotation file
        src_ann = temp_annotations_dir / f"{filename_stem}.xml"
        dst_ann = target_dir / "annotations_oob" / f"{filename_stem}.xml"

        try:
            shutil.move(str(src_ann), str(dst_ann))
        except Exception as e:
            logger.warning(f"Failed to move annotation {filename_stem}.xml: {e}")

    successful_moves = repeated_count - len(failed_moves)
    logger.info(f"Successfully moved {successful_moves}/{repeated_count} repeated frames")

    if failed_moves:
        logger.warning(f"Failed to move {len(failed_moves)} files: {failed_moves[:5]}...")

    return


def _recover_data_from_temp(ctx: FramesDeduplicationContext) -> None:
    """Move all remaining files from temp directories back to original directories."""
    logger.info("Recovering remaining data from temp directories")

    # Temp and original directories
    temp_images_dir = ctx.frames_dir / "images" / "temp"
    temp_annotations_dir = ctx.frames_dir / "annotations_oob" / "temp"
    original_images_dir = ctx.frames_dir / "images"
    original_annotations_dir = ctx.frames_dir / "annotations_oob"

    moved_images = 0
    moved_annotations = 0

    # Move all remaining images
    if temp_images_dir.exists():
        for img_file in temp_images_dir.iterdir():
            if img_file.is_file():
                dst = original_images_dir / img_file.name
                try:
                    shutil.move(str(img_file), str(dst))
                    moved_images += 1
                except Exception as e:
                    logger.warning(f"Failed to recover image {img_file.name}: {e}")

    # Move all remaining annotations
    if temp_annotations_dir.exists():
        for ann_file in temp_annotations_dir.iterdir():
            if ann_file.is_file():
                dst = original_annotations_dir / ann_file.name
                try:
                    shutil.move(str(ann_file), str(dst))
                    moved_annotations += 1
                except Exception as e:
                    logger.warning(f"Failed to recover annotation {ann_file.name}: {e}")

    logger.info(f"Recovered {moved_images} images and {moved_annotations} annotations from temp")

    # Remove temp directories
    shutil.rmtree(temp_images_dir, ignore_errors=True)
    shutil.rmtree(temp_annotations_dir, ignore_errors=True)
    logger.debug("Removed temp directories")


def _log_dedup_stats(ctx: FramesDeduplicationContext, ds: sv.DetectionDataset) -> None:
    """Compute and log frame statistics per video after deduplication."""
    logger.info("Calculating deduplication statistics")

    # Get removed frame counts per video
    removed_dir = ctx.frames_dir / "repeated/images"
    try:
        removed_frame_files = list(removed_dir.glob("*"))
        removed_frame_filenames = [f.stem for f in removed_frame_files if f.is_file()]
        removed_video_filenames = ["_".join(x.split("_")[:-1]) for x in removed_frame_filenames]
        removed_counts = Counter(removed_video_filenames)

    except Exception as e:
        logger.error(f"Failed to scan remaining frames in {images_dir}: {e}")
        raise

    # Get remaining frame counts per video
    images_dir = ctx.frames_dir / "images"
    try:
        keep_frame_files = list(images_dir.glob("*"))
        keep_frame_filenames = [f.stem for f in keep_frame_files if f.is_file()]
        keep_video_filenames = ["_".join(x.split("_")[:-1]) for x in keep_frame_filenames]
        keep_counts = Counter(keep_video_filenames)

    except Exception as e:
        logger.error(f"Failed to scan remaining frames in {images_dir}: {e}")
        raise

    all_video_ids = set(removed_counts) | set(keep_counts)
    total_keep = sum(keep_counts.values())
    total_removed = sum(removed_counts.values())
    total_initial = total_keep + total_removed

    logger.info("Deduplication statistics by video:")
    logger.info("-" * 60)

    for video_id in sorted(all_video_ids):
        removed = removed_counts.get(video_id, 0)
        keep = keep_counts.get(video_id, 0)
        initial = keep + removed
        removed_pct = (removed / initial * 100) if initial > 0 else 0

        logger.info(f"{video_id}: {initial} → {keep} ({removed} removed, {removed_pct:.1f}%)")

    # Overall summary
    logger.info("-" * 60)
    overall_pct = (total_removed / total_initial * 100) if total_initial > 0 else 0
    logger.info(
        f"Overall: {total_initial} → {total_keep} ({total_removed} removed, {overall_pct:.1f}%)"
    )

    return


def deduplicate_frames(ctx: FramesDeduplicationContext) -> None:
    """
    Remove duplicate frames from video datasets using SIFT feature matching.

    Args:
        ctx: FramesDeduplicationContext containing all necessary parameters and objects
    """
    logger.info("Starting frame deduplication process")

    # Set up sift and flann based matcher and add them to context
    _initialize_sift_flann(ctx)

    video_files = _get_video_ids_with_files(ctx)

    # Load parking zone polygons to filter out annotations
    camera_polygons = _get_polygons_from_json(ctx)

    # Process each video separately
    for video_id, filenames in video_files.items():

        _move_data_to_temp_dir(ctx, video_id, filenames)

        # Load dataset
        ds = _load_dataset_as_supervision(ctx)

        # Calculate key points and descriptors for each frame
        features_list = _calculate_kp_and_descriptors(ctx, ds, camera_polygons)

        # Get 1d array labelling each frame as repeated or not
        is_repeated = _check_if_is_repeated(ctx, features_list)

        # Move repeated frames as an archive folder
        _move_repeated_frames(ctx, ds, is_repeated)

        # Move retained data from temp folder to original one
        _recover_data_from_temp(ctx)

    # Calculate retained and removed statistics per video
    _log_dedup_stats(ctx, ds)

    logger.info("Frame deduplication process completed")
