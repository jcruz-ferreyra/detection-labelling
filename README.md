# Detection Dataset Labelling

Dataset preparation pipeline for training object detection models on CCTV footage.

> **Part of the [Cyclist Census](https://github.com/jcruz-ferreyra/cyclist_census) research project** - See the main repository for methodology, results, and the complete development pipeline.

<br>

## Overview

Multi-stage pipeline for preparing detection datasets from CCTV videos. Extracts frames, performs intelligent sampling, removes duplicates, and generates training-ready annotations in multiple formats (YOLO, COCO). Designed to create high-quality, diverse datasets for urban vehicle detection tasks.

### Capabilities

- **Frame extraction**: Automated frame sampling from CCTV videos with spatial distribution scoring and two-wheeler vehicle prioritization
- **Duplicate removal**: SIFT and keypoint matching-based deduplication to reduce dataset redundancy
- **Intelligent sampling**: BYOL-based unsupervised feature learning for diverse frame selection
- **Automatic pre-annotation**: OOB or custom-trained models for annotating new data and reducing manual labor in active learning approach
- **Annotation merging**: NMS-based combination of multiple annotation sources
- 
### Output

- **Extracted frames** - JPG images organized by video source
- **Deduplicated datasets** - Cleaned frame collections without visual duplicates
- **Selected batches** - Strategically sampled subsets for annotation (train/test splits)
- **Preliminar annotations** - OOB annotations for manual tuning or direct use in distilled learning approach

<br>

## Installation
### Prerequisites
   - Python 3.11+
   - Poetry (for dependency management)
   - GPU recommended (CUDA-compatible)
### Steps
1. **Clone the repository**
   ```bash
   git clone https://github.com/jcruz-ferreyra/detection-labelling.git
   cd detection-labelling
   ```
   
2. **Install dependencies**
   ```bash
   poetry install
   ```
   
3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your paths:
   # DATA_DIR=/path/to/your/main/data/folder
   # MODELS_DIR=/path/to/your/main/models/folder
   ```
   
4. **Prepare your data**
   ```bash
   data/
   └── path/to/videos/folder/
       └── {camera_id}/
           └── {video_date}/
               └── {camera_id}_{video_date}_{video_num}.avi
   ```

5. **Download model weights** (Optional)
   
   Place trained model weights in your `MODELS_DIR` if you want to use a custom trained YOLO model:
   ```
   models/
   └── path/to/yolo/model/weights.pt
   ```
   
<br>

## Quick Start

### Task 1: [Extract CCTV Frames](detection_labelling/extract_frames)

Extracts frames from CCTV videos with intelligent sampling based on vehicle detection and spatial distribution scoring.

**Configuration**:

Processing Configuration ([`config.yaml`](detection_labelling/extract_frames/config.yaml))

YAML file defining video identification, class mappings, and extraction logic:

```yaml
# Video Identification
camera_id: ID00000                     # Camera identifier
video_date: YYMMDD                     # Video date (YYMMDD format)
video_num: 1                           # Video number for this date
video_extension: .avi                  # Video file extension
input_folder: path/to/videos/folder/   # Directory containing input videos (relative to DATA_DIR)
output_folder: path/to/frames/folder/  # Directory for extracted frames (relative to DATA_DIR)

# Class Mapping
class_label:                           # Mapping of class IDs to labels
  "0": person
  "1": bicycle
  "2": car
  "3": motorcycle
  "5": bus
  "7": truck

category_classes:                      # Group class IDs into categories
  twowheels: [1, 3]                    # Frames with these classes are prioritized
  heavy: [5, 7]                        # Frames with these classes are saved separately
  car: [2]                             # Frames with these classes are saved if no other saved recently
```

For a complete reference with all available options and detailed comments, see [`config_full.yaml`](detection_labelling/extract_frames/config_full.yaml).

**Run**:
```bash
poetry run python -m detection_labelling.extract_frames
```

**Output** (saved to `output_folder/{camera_id}`):
- `{camera_id}_{video_date}_{video_num}_{frame_number}.jpg` - Extracted frames organized by camera
- `saved_frames.json` - Tracking file with metadata for saved frames
- `processing_stats.json` - Statistics on extraction process (frames processed, saved, skipped)

<br>

## Structure

### Task Architecture

Each task within `cctv_inference` folder follows a consistent structure:

```
process_cctv/
├── __init__.py                 # Package initialization
├── __main__.py                 # Entry point - handles CLI and orchestration
├── config_min.yaml             # Minimum configuration reference
├── config_full.yaml            # Complete configuration reference
├── config.yaml                 # Processing configuration (user's working copy)
├── types.py                    # Context dataclass definition
├── cctv_processing.py          # Core processing logic (called from __main__.py)
└── *.py                        # Modular helper functions (called from cctv_processing.py)
```

**Context Pattern**:

All tasks use a context object to eliminate parameter passing complexity:

```python
@dataclass
class CCTVProcessingContext:
   # Configuration from YAML
   data_dir: Path
   models_dir: Path
   frame_processing: Dict[str, Any]
   detection: Dict[str, Any]
   ...
    
   # Runtime objects (initialized during setup)
   detection_model: Optional[Any] = None
   tracker: Optional[Any] = None
   ...
```

This pattern provides:
- Centralized configuration and state
- Automated path computation using `@property` decorators
- Initial validation using `__post_init__` method
- Intelligent defaults for missing optional configurations

<br>

## How it works

### Task 1: [Process CCTV Video](cctv_inference/process_cctv)

Runs the complete inference pipeline on a single video file, producing count data and optional annotated video output.

<details>
<summary><b>Details</b></summary>
<br>

**Processing Pipeline**:

1. Intialization
   - Initialize models and tracker
   - Initialize line counters
   - Initialize annotators (if video output enabled)

2. Processing Loop
   - Read frame from video
   - Run object detection (YOLO/RFDETR)
   - Update tracker with detections
   - Extract cyclists (person + bicycle IoU matching)
   - Classify cyclists (gender prediction)
   - Trigger line counters for each detection
   - Annotate frame (if video output enabled)

3. Partition Completion (every N minutes)
   - Aggregate gender classifications using temporal weighting
   - Calculate counts by vehicle type, gender, lane, and direction
   - Save results (JSON/CSV)
   - Save cyclist crops (if enabled)
   - Rotate video output file
   - Reset counters for next partition

4. Finalization
   - Process final partition
   - Clean up temporary files
   - Close video output

**Key algorithms**

- Cyclist extraction: IoU-based matching between person and bicycle detections
- Gender aggregation: Weighted mean of classifications for each track (near camera classifications weighted more)
- Line counting: Direction-aware crossing detection with double-count prevention using tracker ID tracking

</details>

<br>

## Additional Resources

For complete methodology, research context, and the full development pipeline, see the main **[Cyclist Census](https://github.com/jcruz-ferreyra/cyclist_census)** repository.

### Related Repositories

- **[detection_labelling](https://github.com/yourusername/detection_labelling)** - Dataset preparation for object detection models
- **[detection_training](https://github.com/yourusername/detection_training)** - YOLO/RFDETR model training pipeline
- **[classification_labelling](https://github.com/yourusername/classification_labelling)** - Dataset preparation for gender classification
- **[classification_training](https://github.com/yourusername/classification_training)** - CNN classifier training with Optuna optimization

### Support

For questions or issues:
- **GitHub Issues**: [cctv-inference/issues](https://github.com/jcruz-ferreyra/cctv-inference/issues)

### Citation

If you use this tool in your research, please cite:
```bibtex
@software{cyclist_census2025,
  title={Cyclist Census: Automated Demographic Analysis from CCTV},
  author={Ferreyra, Juan Cruz},
  institution={Universidad de Buenos Aires},
  year={2025},
  url={https://github.com/jcruz-ferreyra/cyclist_census}
}
```

### License

MIT License - see [LICENSE](LICENSE) file for details.

