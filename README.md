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

**Output** (saved to `output_folder`):
- `images/{camera_id}_{video_date}_{video_num}_{frame_number}.jpg` - Extracted frames organized by camera
- `annotations_oob/{camera_id}_{video_date}_{video_num}_{frame_number}.xml` - Annotations of extracted frames
- `saved_frames.json` - Tracking file with metadata for saved frames
- Processing logs show statistics on extraction process (frames processed, saved, skipped)

### Task 2: [Deduplicate Frames](detection_labelling/deduplicate_frames)

Removes visually similar frames using SIFT keypoint matching to reduce annotation redundancy.

**Configuration**:

Processing Configuration ([`config.yaml`](detection_labelling/deduplicate_frames/config.yaml))

YAML file defining frame locations and SIFT matching parameters:

```yaml
frames_folder: path/to/frames/folder           # Directory containing frames (relative to DATA_DIR)
polygons_json_filename: parking_polygons.json  # Parking area masks to exclude static regions (relative to script location)
```

For a complete reference with all available options and detailed comments, see [`config_full.yaml`](detection_labelling/deduplicate_frames/config_full.yaml).

Parking Polygons Configuration ([`parking_polygons.json`](detection_labelling/deduplicate_frames/parking_polygons.json))

JSON file defining static regions (parking areas) to exclude from keypoint matching, organized by camera ID. Each polygon is an array of `[x, y]` coordinates:
```json
{
  "F012_27F_MAT_240801": [
    [[111, 141], [150, 141], [261, 0], [200, 0], [111, 90]]
  ],
  "F029_PEL_COR": [
    [[260, 78], [260, 88], [90, 576], [0, 576], [0, 488], [237, 78]],
    [[464, 86], [495, 86], [704, 208], [704, 275], [464, 92]]
  ],
  ...
}
```

**Note**: Use `[[0, 0]]` for cameras without parking areas to exclude.

**Run**:
```bash
poetry run python -m detection_labelling.deduplicate_frames
```

**Output** (processed in-place in `frames_folder`):
- Duplicate frames and annotations are moved to `repeated` folder in the same directory
- Unique frames remain organized by camera/video source
- Processing logs show deduplication statistics per video

### Task 3: [Select Initial Batch](detection_labelling/select_initial_batch)

Uses BYOL (Bootstrap Your Own Latent) unsupervised learning to create diverse embeddings, then performs stratified sampling for train/test splits.

**Configuration**:

Processing Configuration ([`config.yaml`](detection_labelling/select_initial_batch/config.yaml))

YAML file defining input/output paths, BYOL model settings, and sampling strategy:

```yaml
# Input/Output
input_folder: path/to/frames/folder    # Directory containing extracted frames (relative to DATA_DIR)
output_folder: path/to/output/folder   # Directory for selected frames output (relative to DATA_DIR)

# Model Files
byol_filename: path/to/byol.pt         # Path to BYOL model weights (relative to MODELS_DIR, trains if not exists)
embed_filename: path/to/embed.parquet  # Path to embeddings file (relative to DATA_DIR, calculates if not exists, .csv or .parquet)

# Sampling Strategy
sampling:                              # Stratified sampling configuration by dataset split
  test:                                # Test split sampling rules
    - str_match: ["camera_id_1"]       #  Match frames containing this string in filename
      samples: 100                     #  Number of frames to sample from matches
  train:                               # Train split sampling rules
    - str_match: ["camera_id_2"]       #  Match frames containing this string
      samples: 15                      #  Number of frames to sample
    - str_match: null                  #  Match remaining frames (complement of all other matches)
      samples: 150                     #  Number of frames to sample from remaining
```

For a complete reference with all available options and detailed comments, see [`config_full.yaml`](detection_labelling/select_initial_batch/config_full.yaml).

**Run**:
```bash
poetry run python -m detection_labelling.select_initial_batch
```

**Output** (datasets saved to `output_folder`):
- `train/` - Selected training frames based on sampling strategy
- `test/` - Selected test frames based on sampling strategy
  
- `DATA_DIR/{embed_filename}` - Embeddings file (CSV or Parquet) for all frames
- `MODELS_DIR/{byol_filename}` - Trained BYOL model weights (if trained during this run)

<br>

## Structure

### Task Architecture

Each task within `detection_labelling` folder follows a consistent structure:

```
extract_frames/
├── __init__.py                 # Package initialization
├── __main__.py                 # Entry point - handles CLI and orchestration
├── config_min.yaml             # Minimum configuration reference (in tasks with verbose config_full)
├── config_full.yaml            # Complete configuration reference
├── config.yaml                 # Processing configuration (user's working copy)
├── types.py                    # Context dataclass definition
├── frame_extraction.py         # Core processing logic (called from __main__.py)
└── *.py                        # Modular helper functions (called from frame_extraction.py)
```

**Context Pattern**:

All tasks use a context object to eliminate parameter passing complexity:

```python
@dataclass
class FrameExtractionContext:
   # Configuration from YAML
   video_path: Path
   output_dir: Path
   class_label: Dict[int, str]
   category_classes: Dict[str, list]
   ...
    
   # Runtime objects (initialized during setup)
   model: Optional[YOLO] = None
   generator: Optional[Generator] = None
   ...
```

This pattern provides:
- Centralized configuration and state
- Automated path computation using `@property` decorators
- Initial validation using `__post_init__` method
- Intelligent defaults for missing optional configurations

<br>

## How it works

### Task 1: [Extract Frames](detection_labelling/extract_and_upload_frames)
Extracts vehicle-containing frames from CCTV videos using detection-based sampling with temporal and spatial filtering.

<details>
<summary><b>Details</b></summary>
<br>

**Processing Pipeline**:
1. Initialization
   - Load YOLO detection model
   - Initialize FrameHandler (temporal sampling logic)
   - Initialize QuadrantScorer (spatial diversity scoring)
   - Load saved frames tracking (resume capability)
2. Video Processing Loop
   - Read frame at sampling interval (fps_of_interest)
   - Run YOLO detection with category-specific confidence thresholds
   - Filter detections by category (two-wheels, cars, heavy vehicles)
   - Check temporal constraints (skip period after recent save)
   - Score frame spatial distribution (quadrant-based)
   - Save frame if criteria met
3. Category-Specific Logic
   - **Two-wheels (bicycle, motorcycle)**: Check near frames and save the one with the highest score.
   - **Heavy vehicles (bus, truck)**: Save always to increase representativity
   - **Cars**: Only saved if no other category saved recently (max_seconds_without_save threshold)
4. Scoring two-wheelers frames
   - Score higher frames with low confidence two-wheeler detections (hard to detect with pretrained models)
   - Divide frame into grid (default 4×4) and score higher frames with detections in low frequency quadrants
   - Score higher frames with high detection class diversity
6. Output Organization
   - Frames saved in a single folder: `{output_dir}/{filename}.jpg`
   - Tracking JSON maintains list of saved frames per video per vehicle category

**Key Algorithms**:
- **FrameHandler**: Manages sampling intervals, skip periods, and timeout thresholds for temporal filtering
- **QuadrantScorer**: Scores frames based on detection spatial distribution to maximize dataset diversity
- **Lazy Loading**: Processes videos individually to handle large datasets (50k+ frames) without memory issues

</details>

---

### Task 2: [Deduplicate Frames](detection_labelling/deduplicate_frames)
Removes duplicate frames using SIFT feature matching to ensure dataset uniqueness while preserving parking area visibility.

<details>
<summary><b>Details</b></summary>
<br>

**Processing Pipeline**:
1. Initialization
   - Load parking area polygons (regions to exclude from comparison)
   - Initialize SIFT feature detector
   - Initialize FLANN matcher with KD-tree index
   - Group frames by video ID for memory-efficient processing
2. Video-by-Video Processing
   - Load all frames for single video into temporary directory
   - For each frame:
     - Extract SIFT keypoints and descriptors
     - Mask out parking area regions (exclude parked vehicles)
     - Compare with previous frames using FLANN matching
     - Mark as duplicate if similarity > threshold
   - Move non-duplicate frames to output directory
   - Clean up temporary directory
3. Duplicate Detection
   - Calculate good matches using Lowe's ratio test
   - Compute similarity score: `good_matches / min(kp1, kp2)`
   - Mark as duplicate if similarity > is_repeated_threshold (default 0.05)
4. Memory Management
   - Process one video at a time (prevents OOM with 53k+ frames)
   - Use temporary directory for isolated processing
   - Clean up after each video completion

**Key Algorithms**:
- **SIFT Feature Extraction**: Scale-invariant features robust to lighting/perspective changes
- **FLANN Matching**: Fast approximate nearest neighbor search for high-dimensional descriptors
- **Parking Masking**: Excludes static parked vehicles from similarity comparison
- **Video-Grouped Processing**: Handles large datasets by processing videos in isolation

</details>

---

### Task 3: [Select Initial Batch](detection_labelling/select_initial_batch)
Selects a diverse, stratified subset of frames for annotation using BYOL-based embeddings and camera-aware sampling.

<details>
<summary><b>Details</b></summary>
<br>

**Processing Pipeline**:
1. BYOL Training (if model doesn't exist)
   - Train self-supervised BYOL model on extracted frames
   - Learn visual representations without labels
   - Save model and checkpoints for resumption
2. Embedding Calculation (if embeddings don't exist)
   - Load trained BYOL model
   - Extract embeddings for all frames
   - Save embeddings to Parquet/CSV for reuse
3. Stratified Sampling
   - Load embeddings and frame metadata
   - Group frames by camera ID and dataset split (train/test)
   - Apply sampling rules:
     - **String matching**: Select N samples from frames matching camera pattern
     - **Complement sampling**: Select N samples from remaining unmatched frames
   - Ensure diverse representation across cameras and conditions
   - Prioritize frames with two-wheelers detections
4. Output Generation
   - Copy selected frames to split-specific directories (train/test)
   - Maintain original filenames for traceability

**Sampling Configuration**:
```yaml
sampling:
  test:
    - str_match: ["F053_LAG_H35"]  # Specific camera
      samples: 100
  train:
    - str_match: ["domo"]          # Camera pattern
      samples: 15
    - str_match: null              # Remaining frames
      samples: 150
```

**Key Algorithms**:
- **BYOL (Bootstrap Your Own Latent)**: Self-supervised learning creates visual embeddings without labels
- **Stratified Sampling**: Ensures balanced representation across cameras, preventing bias toward high-traffic locations
- **String Matching**: Flexible camera selection using substring patterns in filenames
- **Complement Sampling**: `str_match: null` selects from frames not matched by other rules

**Use Case**: Creates initial labeled dataset for training detection models, balancing camera diversity and dataset size constraints.

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
