import json
import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

from detection_labelling.utils import get_ultralytics_detections, save_as_pascal_voc

from .frame_handling import FrameHandler
from .frame_scoring import QuadrantScorer, compute_conf_weight, compute_diversity_score
from .types import FrameExtractionContext, ProcessingStats

logger = logging.getLogger(__name__)


def _initialize_video_components(ctx: FrameExtractionContext) -> None:
    """Initialize video generator and video info. Add them to context."""
    logger.info("Creating video generator and extracting video info")

    try:
        ctx.generator = sv.get_video_frames_generator(str(ctx.video_path))
        ctx.video_info = sv.VideoInfo.from_video_path(str(ctx.video_path))

        if ctx.video_info.total_frames == 0:
            raise ValueError("Video has no frames")

        logger.info(
            f"Video info - Total frames: {ctx.video_info.total_frames}, FPS: {ctx.video_info.fps}, "
            f"Resolution: {ctx.video_info.width}x{ctx.video_info.height}"
        )

    except Exception as e:
        logger.error(f"Failed to create video generator: {e}")
        raise


def _initialize_model(ctx: FrameExtractionContext) -> None:
    """Initialize YOLO model. Add it to context."""
    logger.info(f"Loading YOLO model: {ctx.yolo_path}")

    try:
        ctx.model = YOLO(ctx.yolo_path)
        ctx.model.fuse()
        logger.info(f"Successfully loaded and fused YOLO model: {ctx.yolo_path}")

    except Exception as e:
        logger.error(f"Failed to load YOLO model {ctx.yolo_path}: {e}")
        raise


def _initialize_processing_components(ctx: FrameExtractionContext) -> None:
    """Initialize frame handler and quadrant scorer. Add them to context."""
    logger.info("Initializing frame handler and quadrant scorer")

    # Initialize frame handler
    ctx.frame_handler = FrameHandler(ctx.video_info.fps, **ctx.frame_handler_params)
    logger.info(
        f"Frame handler initialized with: "
        f"FPS of interest: {ctx.frame_handler.fps_of_interest}, "
        f"intermediate frames: {ctx.frame_handler.n_intermediate}"
    )

    # Initialize quadrant scorer
    frame_shape = (ctx.video_info.width, ctx.video_info.height)
    ctx.quadrant_scorer = QuadrantScorer(frame_shape=frame_shape, **ctx.quadrant_scorer_params)
    logger.info(
        f"Quadrant scorer initialized with:"
        f"grid size, ({ctx.quadrant_scorer.rows}, {ctx.quadrant_scorer.cols}), "
        f"k coefficient: {ctx.quadrant_scorer.k}"
    )

    logger.info("Processing components initialized successfully")


def _load_existing_saved_frames(ctx: FrameExtractionContext) -> None:
    """Load existing saved frames from JSON file if it exists."""
    json_path = ctx.output_dir / "saved_frames.json"

    if not json_path.exists():
        logger.info("No existing saved frames file found, starting fresh")
        return

    try:
        logger.info(f"Loading existing saved frames from: {json_path}")
        with open(json_path, "r") as f:
            existing_data = json.load(f)

        total_existing = 0
        for category, frames in existing_data.items():
            ctx.saved_frames[category].extend(frames)
            total_existing += len(frames)

        logger.info(
            f"Loaded {total_existing} existing saved frames across {len(existing_data)} categories"
        )

    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Failed to load existing saved frames: {e}")
        logger.info("Starting with empty saved frames dictionary")


def _advance_frame_counter(ctx: FrameExtractionContext, pbar) -> int:
    """Update progress bar and advance frame counter."""
    pbar.update(1)
    idx = ctx.frame_handler.update_frame_counter()

    return idx


def _get_frame_detections(
    ctx: FrameExtractionContext, frame: np.ndarray, idx: int
) -> Optional[sv.Detections]:
    """Get YOLO detections for current frame."""
    logger.debug(f"Running detection on frame {idx}")

    try:
        dets = get_ultralytics_detections(frame, ctx.model, ctx.yolo_params, ctx.class_confidence)
        logger.debug(f"Found {len(dets)} detections in frame {idx}")
        return dets

    except Exception as e:
        logger.warning(f"Detection failed on frame {idx}: {e}")
        return sv.Detections.empty()


def _get_intermediate_frames_dets(
    frames_with_dets: Dict[int, Tuple[np.ndarray, sv.Detections]],
    ctx: FrameExtractionContext,
) -> Dict[int, Tuple[np.ndarray, sv.Detections]]:
    """Get detections for frames that don't have them yet."""
    for frame_idx, (frame, dets) in frames_with_dets.items():
        if dets is None:
            try:
                dets = get_ultralytics_detections(
                    frame, ctx.model, ctx.yolo_params, ctx.class_confidence
                )
                frames_with_dets[frame_idx] = (frame, dets)
            except Exception as e:
                logger.warning(f"Failed to get detections for intermediate frame {frame_idx}: {e}")
                frames_with_dets[frame_idx] = (frame, sv.Detections.empty())

    return frames_with_dets


def _collect_frames_with_detections(
    current_frame: np.ndarray,
    current_dets: sv.Detections,
    idx: int,
    ctx: FrameExtractionContext,
    pbar,
) -> Dict[int, Tuple[np.ndarray, sv.Detections]]:
    """Collect frames with dets until next frame of interest."""
    # Get previous intermediate frames stored in the frame handler
    frames = ctx.frame_handler.intermediate_frames.copy()
    frames_with_dets = {k: (v, None) for k, v in frames.items()}  # include placeholder for dets

    # Store current frame and dets
    frames_with_dets[idx] = (current_frame, current_dets)

    # Calculate and store following intermediate frames
    ctx.frame_handler.update_intermediate_frames_idx(idx)

    # Safety counter to prevent infinite loops
    max_iterations = ctx.frame_handler.fps * (1 / ctx.frame_handler.fps_of_interest)
    iterations = 0

    while not ctx.frame_handler.is_frame_of_interest(idx + 1) and iterations < max_iterations:
        pbar.update(1)  # Update progress bar
        idx = ctx.frame_handler.update_frame_counter()

        try:
            frame = next(ctx.generator)
        except StopIteration:
            logger.warning("Video ended while collecting intermediate frames")
            break

        if idx in ctx.frame_handler.intermediate_frames_idx:
            frames_with_dets[idx] = (frame, None)

        iterations += 1

    if iterations >= max_iterations:
        logger.warning(f"Max iterations reached while collecting frames at idx {idx}")

    # Get dets for all frames in the collection
    frames_with_dets = _get_intermediate_frames_dets(frames_with_dets, ctx)

    logger.debug(f"Collected {len(frames_with_dets)} frames for processing")
    return frames_with_dets


def _get_best_frame_with_dets(
    ctx: FrameExtractionContext, frames_with_dets: Dict[int, tuple], classes_of_interest: list[int]
) -> Dict[str, Any]:
    """Select the best frame with detections based on quadrant scoring and diversity."""
    logger.info("Selecting best twowheeler frame using detections")

    if not frames_with_dets:
        logger.warning("No frames provided for best frame selection")
        return {}

    best_frame_with_det = {}
    best_score = float("-inf")
    best_quadrant_indices = None

    for frame_idx, (frame, dets) in frames_with_dets.items():
        # Filter to classes of interest
        is_of_interest = np.isin(dets.class_id, classes_of_interest)
        dets_oi = dets[is_of_interest]

        if len(dets_oi) == 0:
            continue

        try:
            # Get detection centers and quadrant information
            dets_oi_centers = dets_oi.get_anchors_coordinates(sv.Position.CENTER)
            quadrant_indices = ctx.quadrant_scorer.get_quadrants(dets_oi_centers)
            quadrant_scores = ctx.quadrant_scorer.get_quadrant_scores(quadrant_indices)

            # Calculate scoring components
            conf_weights = compute_conf_weight(dets_oi.confidence)
            n_unique = len(np.unique(dets.class_id))
            diversity_score = compute_diversity_score(n_unique)

            # Calculate final frame score
            frame_score = np.sum(quadrant_scores * conf_weights) + diversity_score

            # Update best frame if this one is better
            if frame_score > best_score:
                best_frame_with_det = {"idx": frame_idx, "frame": frame, "dets": dets}
                best_score = frame_score
                best_quadrant_indices = quadrant_indices

        except Exception as e:
            logger.warning(f"Error scoring frame {frame_idx}: {e}")
            continue

    if best_quadrant_indices is not None:
        ctx.quadrant_scorer.update_quadrants_count(best_quadrant_indices)

    return best_frame_with_det


def _get_filename_stem(ctx: FrameExtractionContext, idx: int) -> str:
    """Generate filename stem from video name and frame index."""
    return f"{ctx.video_path.stem}_{idx:06d}"


def _process_twowheels_detections(
    ctx: FrameExtractionContext,
    frame: np.ndarray,
    dets: sv.Detections,
    idx: int,
    pbar,
) -> bool:
    """Process frames containing two-wheeled vehicles."""
    logger.debug(f"Frame {idx}: Processing twowheels detections")

    # Collect frames for best frame selection
    frames_with_dets = _collect_frames_with_detections(frame, dets, idx, ctx, pbar)

    # Select best frame from collection
    best_frame_with_det = _get_best_frame_with_dets(
        ctx,
        frames_with_dets,
        ctx.category_classes["twowheels"],
    )

    if not best_frame_with_det:
        logger.warning(f"No suitable twowheels frame found for idx {idx}")
        return False

    # Save the best frame and annotations
    filename_stem = _get_filename_stem(ctx, best_frame_with_det["idx"])

    try:
        save_as_pascal_voc(
            best_frame_with_det["frame"],
            best_frame_with_det["dets"],
            ctx.output_dir,
            filename_stem,
            ctx.class_label,
        )

        # Update tracking
        ctx.saved_frames["twowheels"].append(filename_stem)
        ctx.frame_handler.update_last_saved_frame_idx(idx + 1)

        logger.info(
            f"Saved twowheels frame {best_frame_with_det['idx']} (best of {len(frames_with_dets)})"
        )
        return True

    except Exception as e:
        logger.error(f"Failed to save twowheels frame {filename_stem}: {e}")
        return False


def _process_non_twowheels_detections(
    ctx: FrameExtractionContext,
    frame: np.ndarray,
    dets: sv.Detections,
    idx: int,
) -> bool:
    """Process frames that don't contain two-wheeled vehicles."""
    logger.debug(f"Frame {idx}: Processing non-twowheels detections")

    # Check and process frames with heavy vehicles
    any_heavy = np.any(np.isin(dets.class_id, ctx.category_classes["heavy"]))
    if any_heavy:
        filename_stem = _get_filename_stem(ctx, idx)

        try:
            save_as_pascal_voc(frame, dets, ctx.output_dir, filename_stem, ctx.class_label)

            # Update tracking
            ctx.saved_frames["heavy"].append(filename_stem)
            ctx.frame_handler.update_last_saved_frame_idx(idx)

            logger.info(f"Saved heavy vehicle frame {idx}")
            return True

        except Exception as e:
            logger.error(f"Failed to save heavy vehicle frame {idx}: {e}")
            return False

    # Check and process frames with cars if long time since last save
    any_car = np.any(np.isin(dets.class_id, ctx.category_classes["car"]))
    long_time_since_last_save = ctx.frame_handler.is_long_time_since_last_save(idx)

    if any_car and long_time_since_last_save:
        filename_stem = _get_filename_stem(ctx, idx)

        try:
            save_as_pascal_voc(frame, dets, ctx.output_dir, filename_stem, ctx.class_label)

            # Update tracking
            ctx.saved_frames["car"].append(filename_stem)
            ctx.frame_handler.update_last_saved_frame_idx(idx)

            logger.info(f"Saved car frame {idx} (10+ seconds since last save)")
            return True

        except Exception as e:
            logger.error(f"Failed to save car frame {idx}: {e}")
            return False

    logger.debug(f"Frame {idx}: No non-twowheels detections saved")
    return False


def _process_frame_detections(
    ctx: FrameExtractionContext,
    frame: np.ndarray,
    dets: sv.Detections,
    idx: int,
    pbar,
) -> bool:
    """Route frame processing based on detection content."""
    any_twowheels = np.any(np.isin(dets.class_id, ctx.category_classes["twowheels"]))

    if any_twowheels:
        was_saved = _process_twowheels_detections(ctx, frame, dets, idx, pbar)
    else:
        was_saved = _process_non_twowheels_detections(ctx, frame, dets, idx)

    return was_saved


def _process_video(ctx: FrameExtractionContext) -> ProcessingStats:
    """Process all video frames with detection and filtering."""
    logger.info("Starting main processing loop")

    stats = ProcessingStats()

    with tqdm(total=ctx.video_info.total_frames) as pbar:
        for frame in ctx.generator:
            idx = _advance_frame_counter(ctx, pbar)

            # Skip non-interest frames
            if not ctx.frame_handler.is_frame_of_interest(idx):
                stats.update(skipped_sampling=True)

                if idx in ctx.frame_handler.intermediate_frames_idx:
                    ctx.frame_handler.update_intermediate_frames(idx, frame)
                continue

            # Calculate and store following intermediate frame idx
            ctx.frame_handler.update_intermediate_frames_idx(idx)

            # Skip processing frames for n seconds after the last save
            if ctx.frame_handler._should_skip_frame_processing(idx):
                stats.update(skipped_timing=True)
                continue

            # Get detections for the current frame
            dets = _get_frame_detections(ctx, frame, idx)
            if dets is None:
                stats.update(no_detections=True)
                continue

            # Process detections and save if appropriate
            was_saved = _process_frame_detections(ctx, frame, dets, idx, pbar)
            stats.update(processed=True, saved=was_saved)

            # Testing limit
            if ctx.testing_frame_limit > 0 and idx > ctx.testing_frame_limit:
                logger.info(f"Reached testing limit at frame {idx}")
                break

    logger.info("Main processing loop completed")
    return stats


def _saved_frames_to_json(ctx: FrameExtractionContext) -> None:
    """
    Save the saved_frames dictionary to JSON file.
    """
    json_path = ctx.output_dir / "saved_frames.json"

    try:
        # Convert defaultdict to regular dict and remove duplicates
        regular_dict = {
            category: list(set(frames)) for category, frames in ctx.saved_frames.items()
        }

        # Create parent directory if needed
        json_path.parent.mkdir(parents=True, exist_ok=True)

        with open(json_path, "w") as f:
            json.dump(regular_dict, f, indent=2)

        total_frames = sum(len(frames) for frames in ctx.saved_frames.values())
        logger.debug(f"Saved frames dictionary updated: {json_path} ({total_frames} unique frames)")

    except Exception as e:
        logger.error(f"Failed to save frames dictionary: {e}")


def extract_frames(ctx: FrameExtractionContext) -> None:
    """
    Process video frames to detect and save frames containing vehicles.

    Args:
        ctx: FrameExtractionContext containing all necessary parameters and objects
    """
    logger.info(f"Starting video processing for: {ctx.video_path}")

    # Initialize all components and add them to context
    _initialize_video_components(ctx)
    _initialize_model(ctx)
    _initialize_processing_components(ctx)
    _load_existing_saved_frames(ctx)

    # Main processing
    stats = _process_video(ctx)

    # Cleanup
    _saved_frames_to_json(ctx)
    stats.log_summary(logger, ctx.video_info.total_frames)

    logger.info("Video processing completed successfully")
