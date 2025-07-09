import yaml
from pathlib import Path
import inspect

import supervision as sv
from ultralytics import YOLO
from detection_labelling.extract_and_upload_frames import VideoProcessingContext

# Import configurations
from detection_labelling.config import HOME


# Create paths for input and output directories
INPUT_MAIN_DIR = DATA_DIR / "raw/cctv_videos"
OUTPUT_DIR = "raw/cctv_frames"

# Get script specific configs
SCRIPT_NAME = Path(inspect.getfile(inspect.currentframe())).stem
CONFIG_PATH = HOME / "detection_labelling" / "configs" / f"{SCRIPT_NAME}.yaml"

with open(CONFIG_PATH, "r") as f:
    script_config = yaml.safe_load(f)

CAMERA_ID = script_config["camera_id"]
VIDEO_DATE = script_config["video_date"]
VIDEO_NUM = script_config["video_num"]
VIDEO_EXTENSION = script_config["video_extension"]

MODEL = script_config["model"]
MODEL_PARAMS = script_config["model_params"]

COCO_CLASS = {int(k): v for k, v in script_config["coco_class"].items()}
CATEGORY_CLASSES = script_config["category_classes"]
CATEGORY_CONFIDENCE = script_config["category_confidence"]


# Ensure video file exists
video_filename = f"{CAMERA_ID}_{VIDEO_DATE}_{VIDEO_NUM}{VIDEO_EXTENSION}"
video_path = INPUT_MAIN_DIR / CAMERA_ID / str(VIDEO_DATE) / video_filename

if not video_path.exists():
    raise FileNotFoundError(f"Video file does not exist at: {video_path}")

# Initialize YOLO model
model = YOLO(MODEL)
model.fuse()

# Create generator and retrieve video info.
generator = sv.get_video_frames_generator(str(video_path))
video_info = sv.VideoInfo.from_video_path(str(video_path))

context = VideoProcessingContext(
    video_path=video_path,
    output_dir=OUTPUT_DIR,
    model=model,
    model_params=MODEL_PARAMS,
    coco_class=COCO_CLASS,
    category_classes=CATEGORY_CLASSES,
    category_confidence=CATEGORY_CONFIDENCE,
    video_info=video_info,
    frame_generator=generator,
)
















