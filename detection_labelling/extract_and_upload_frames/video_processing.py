import json
import logging
from typing import Dict, Tuple

import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

from detection_labelling.utils import get_ultralytics_detections, save_as_pascal_voc

from .frame_handling import FrameHandler
from .frame_scoring import QuadrantScorer
from .frame_selection import get_best_frame_with_dets
from .types import VideoProcessingContext

# Set up logger for this module
logger = logging.getLogger(__name__)


def _get_filename_stem(ctx: VideoProcessingContext, idx: int) -> str:
    """
    Generate a filename stem based on the video context and frame index.

    Args:
        ctx: VideoProcessingContext containing video metadata
        idx: Current frame index
    Returns:
        str: Filename stem in the format "camera_id_video_date_video_num_frame_idx"
    """
    return f"{ctx.video_path.stem}_{idx:06d}"


def _get_intermediate_frames_dets(
    frames_with_dets: Dict[int, Tuple[np.ndarray, sv.Detections]],
    ctx: VideoProcessingContext,
) -> Dict[int, Tuple[np.ndarray, sv.Detections]]:
    """
    Get dets for frames before and after the main frame of interest.
    """
    for frame_idx, (frame, dets) in frames_with_dets.items():
        if dets is None:
            dets = get_ultralytics_detections(
                frame, ctx.model, ctx.model_params, ctx.class_confidence
            )
            frames_with_dets[frame_idx] = (frame, dets)

    return frames_with_dets


def _collect_frames_with_detections(
    current_frame: np.ndarray,
    current_dets: sv.Detections,
    idx: int,
    ctx: VideoProcessingContext,
    pbar,
) -> Dict[int, Tuple[np.ndarray, sv.Detections]]:
    """
    Collect frames with dets until next frame of interest.

    Returns:
        dict: Dictionary mapping frame indices to (frame, dets) tuples
    """
    # Get previous intermediate frames stored in the frame handler
    frames = ctx.frame_handler.intermediate_frames.copy()
    frames_with_dets = {k: (v, None) for k, v in frames.items()}  # include placeholder for dets

    # Store current frame and dets
    frames_with_dets[idx] = (current_frame, current_dets)

    # Calculate and store following intermediate frames
    ctx.frame_handler.update_intermediate_frames_idx(idx)

    while not ctx.frame_handler.is_frame_of_interest(idx + 1):
        pbar.update(1)  # Update progress bar
        idx = ctx.frame_handler.update_frame_counter()

        try:
            frame = next(ctx.generator)
        except StopIteration:
            logger.warning("Video ended while collecting intermediate frames")
            break

        if idx in ctx.frame_handler.intermediate_frames_idx:
            frames_with_dets[idx] = (frame, None)

    # Get dets for all frames in the collection
    frames_with_dets = _get_intermediate_frames_dets(frames_with_dets, ctx)

    logger.debug(f"Collected {len(frames_with_dets)} frames for processing")
    return frames_with_dets


def _process_twowheels_detections(
    ctx: VideoProcessingContext,
    frame: np.ndarray,
    dets: sv.Detections,
    idx: int,
    pbar,
) -> bool:
    """Process frames containing two-wheeled vehicles."""
    frames_with_dets = _collect_frames_with_detections(frame, dets, idx, ctx, pbar)

    best_frame_with_det = get_best_frame_with_dets(
        frames_with_dets,
        ctx.quadrant_scorer,
        ctx.category_classes["twowheels"],
    )

    # Save the best frame and annotations
    filename_stem = _get_filename_stem(ctx, best_frame_with_det["idx"])
    save_as_pascal_voc(
        best_frame_with_det["frame"],
        best_frame_with_det["dets"],
        ctx.output_dir,
        filename_stem,
        ctx.class_label,
    )

    ctx.saved_frames["twowheels"].append(filename_stem)
    ctx.frame_handler.update_last_saved_frame_idx(idx + 1)

    logger.info(
        f"Saved twowheels frame {best_frame_with_det['idx']} (best of {len(frames_with_dets)})"
    )
    return True


def _process_non_twowheels_detections(
    ctx: VideoProcessingContext,
    frame: np.ndarray,
    dets: sv.Detections,
    idx: int,
) -> bool:
    """
    Process frames that don't contain two-wheeled vehicles.

    Returns:
        bool: True if frame was processed and saved, False otherwise
    """
    # Check and process frames with heavy vehicles
    any_heavy = np.any(np.isin(dets.class_id, ctx.category_classes["heavy"]))
    if any_heavy:
        filename_stem = _get_filename_stem(ctx, idx)
        save_as_pascal_voc(frame, dets, ctx.output_dir, filename_stem, ctx.class_label)

        ctx.saved_frames["heavy"].append(filename_stem)
        ctx.frame_handler.update_last_saved_frame_idx(idx)

        logger.info(f"Saved heavy vehicle frame {idx}")
        return True

    # Check and process frames with cars if long time since last save
    any_car = np.any(np.isin(dets.class_id, ctx.category_classes["car"]))
    long_time_since_last_save = ctx.frame_handler.is_long_time_since_last_save(idx)
    if any_car and long_time_since_last_save:
        filename_stem = _get_filename_stem(ctx, idx)
        save_as_pascal_voc(frame, dets, ctx.output_dir, filename_stem, ctx.class_label)

        ctx.saved_frames["car"].append(filename_stem)
        ctx.frame_handler.update_last_saved_frame_idx(idx)

        logger.info(f"Saved car frame {idx} (10+ seconds since last save)")
        return True

    logger.debug(f"Frame {idx}: No non-twowheels detections saved")
    return False


def _initialize_video_components(ctx: VideoProcessingContext):
    """Initialize video generator and extract video information."""
    try:
        logger.info("Creating video generator and extracting video info")
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


def _initialize_model(ctx: VideoProcessingContext):
    """Initialize and fuse YOLO model."""
    logger.info(f"Loading YOLO model: {ctx.model_name}")
    try:
        ctx.model = YOLO(ctx.model_name)
        ctx.model.fuse()
        logger.info(f"Successfully loaded and fused YOLO model: {ctx.model_name}")
    except Exception as e:
        logger.error(f"Failed to load YOLO model {ctx.model_name}: {e}")
        raise


def _initialize_processing_components(ctx: VideoProcessingContext):
    """Initialize frame handler and quadrant scorer."""
    logger.info("Initializing frame handler and quadrant scorer")
    try:
        ctx.frame_handler = FrameHandler(ctx.video_info.fps, **ctx.frame_handler_params)
        logger.info(
            f"Frame handler initialized with: "
            f"FPS of interest: {ctx.frame_handler.fps_of_interest}, "
            f"intermediate frames: {ctx.frame_handler.n_intermediate}"
        )

        frame_shape = (ctx.video_info.width, ctx.video_info.height)
        ctx.quadrant_scorer = QuadrantScorer(frame_shape=frame_shape, **ctx.quadrant_scorer_params)
        logger.info(
            f"Quadrant scorer initialized with:"
            f"grid size, ({ctx.quadrant_scorer.rows}, {ctx.quadrant_scorer.cols}), "
            f"k coefficient: {ctx.quadrant_scorer.k}"
        )

    except Exception as e:
        logger.error(f"Failed to initialize processing components: {e}")
        raise


def _load_existing_saved_frames(ctx: VideoProcessingContext):
    """Load existing saved frames from JSON file if it exists."""
    json_path = ctx.output_dir / "saved_frames.json"

    if json_path.exists():
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
    else:
        logger.info("No existing saved frames file found, starting fresh")


def _save_frames_to_json(ctx: VideoProcessingContext):
    """Save the saved_frames dictionary to JSON file."""
    json_path = ctx.output_dir / "saved_frames.json"
    try:
        # Convert defaultdict to regular dict for JSON serialization
        regular_dict = {
            category: list(set(frames)) for category, frames in ctx.saved_frames.items()
        }

        with open(json_path, "w") as f:
            json.dump(regular_dict, f, indent=2)

        total_frames = sum(len(frames) for frames in ctx.saved_frames.values())
        logger.debug(f"Saved frames dictionary updated: {json_path} ({total_frames} total frames)")

    except Exception as e:
        logger.error(f"Failed to save frames dictionary: {e}")


def process_video(ctx: VideoProcessingContext):
    """
    Process video frames to detect and save frames containing vehicles.

    Args:
        ctx: VideoProcessingContext containing all necessary parameters and objects
    """
    logger.info(f"Starting video processing for: {ctx.video_path}")

    # Initialize all components
    _initialize_video_components(ctx)
    _initialize_model(ctx)
    _initialize_processing_components(ctx)
    _load_existing_saved_frames(ctx)

    logger.info("Starting main processing loop")

    n_processed = 0
    n_saved = 0

    # Main processing loop
    with tqdm(total=ctx.video_info.total_frames) as pbar:
        for frame in ctx.generator:
            pbar.update(1)
            idx = ctx.frame_handler.update_frame_counter()

            # Skip non-interest frames
            if not ctx.frame_handler.is_frame_of_interest(idx):
                if idx in ctx.frame_handler.intermediate_frames_idx:
                    ctx.frame_handler.update_intermediate_frames(idx, frame)
                continue

            # Calculate and store following intermediate frame idx
            ctx.frame_handler.update_intermediate_frames_idx(idx)

            # Skip processing frames for n seconds after the last save
            if ctx.frame_handler._should_skip_frame_processing(idx):
                continue

            # Get detections for the current frame
            logger.debug(f"Running detection on frame {idx}")
            try:
                dets = get_ultralytics_detections(
                    frame, ctx.model, ctx.model_params, ctx.class_confidence
                )
            except Exception as e:
                logger.error(f"Detection failed on frame {idx}: {e}")
                continue

            if len(dets) == 0:
                logger.debug(f"No detections found in frame {idx}")
                continue

            n_processed += 1

            # Process frame based on presence of two-wheeled vehicles
            any_twowheels = np.any(np.isin(dets.class_id, ctx.category_classes["twowheels"]))
            if any_twowheels:
                logger.debug(f"Frame {idx}: Processing twowheels detections")
                was_saved = _process_twowheels_detections(ctx, frame, dets, idx, pbar)
            else:
                logger.debug(f"Frame {idx}: Processing non-twowheels detections")
                was_saved = _process_non_twowheels_detections(ctx, frame, dets, idx)

            if was_saved:
                n_saved += 1

            # Break condition for testing
            if idx > 10_000:
                break

    _save_frames_to_json(ctx)

    logger.info("Video processing completed")
    logger.info("Final statistics:")
    logger.info(f"  - Total frames in video: {ctx.video_info.total_frames}")
    logger.info(f"  - Frames processed: {n_processed}")
    logger.info(f"  - Frames saved: {n_saved}")
