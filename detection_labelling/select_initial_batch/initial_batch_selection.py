import logging
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

from .types import BatchSelectionContext

logger = logging.getLogger(__name__)


def _validate_sampling(ctx: BatchSelectionContext) -> None:
    """Validate sampling configuration allows only one null str_match."""
    logger.info("Validating sampling configuration")

    null_count = 0
    for subset_config in ctx.sampling.values():
        null_count += sum(1 for item in subset_config if item["str_match"] is None)

    if null_count > 1:
        raise ValueError(
            "Multiple null str_match found in sampling configuration. Only one null allowed."
        )

    logger.info("Sampling config validation passed.")


def _load_embeddings(ctx: BatchSelectionContext) -> pd.DataFrame:
    """Load embeddings from CSV or Parquet file."""
    try:
        logger.info(f"Loading pretrained embeddings: {ctx.embed_path}")
        filename_suffix = ctx.embed_path.suffix.lower()
        if filename_suffix == ".csv":
            df_embed = pd.read_csv(ctx.embed_path)
        elif filename_suffix == ".parquet":
            df_embed = pd.read_parquet(ctx.embed_path, engine="pyarrow")
        else:
            raise ValueError(f"Unsupported file type: {filename_suffix}")

        logger.info(f"Embeddings successfully loaded: {len(df_embed)} samples")
        return df_embed

    except Exception as e:
        logger.error(f"Failed to load embeddings: {e}")
        raise


def _get_filenames_and_embeddings(df_embed: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """Extract filenames and embeddings array from dataframe."""
    try:
        filenames_list = df_embed["filename"].tolist()
        embeddings_array = df_embed.drop(columns=["filename"]).values
        if np.isnan(embeddings_array).any():
            raise ValueError("Embeddings array contains NaN values")

        logger.info(
            f"Extracted {len(filenames_list)} filenames, embeddings shape: {embeddings_array.shape}"
        )
        return filenames_list, embeddings_array

    except (KeyError, ValueError) as e:
        logger.error(f"Problem with embeddings data: {e}")
        raise


def _create_subset_mask(filenames_list: list[str], str_match_list: list[str]) -> np.ndarray:
    """Create mask for filenames containing any string in str_match_list."""
    if not str_match_list:  # Empty list
        return np.array([False] * len(filenames_list))

    return np.array(
        [any(match_str in filename for match_str in str_match_list) for filename in filenames_list]
    )


def _create_stratified_masks(
    ctx: BatchSelectionContext, filenames_list: list[str]
) -> dict[str, list[dict]]:
    """Create stratified masks based on script configuration."""
    logger.info("Creating stratified masks")

    all_masks = {}
    used_mask = np.array([False] * len(filenames_list))

    # Process all non-null items
    complement_samples = None
    complement_name = None

    for split_name, subset_list in ctx.sampling.items():
        all_masks[split_name] = []

        for subset_config in subset_list:
            if subset_config["str_match"] is None:
                complement_samples = subset_config["samples"]
                complement_name = split_name
                continue

            mask = _create_subset_mask(filenames_list, subset_config["str_match"])
            all_masks[split_name].append({"mask": mask, "samples": subset_config["samples"]})

            # Update globally used mask
            used_mask = used_mask | mask

            count = np.sum(mask)
            logger.info(
                f"Mask for '{split_name}' with {subset_config['str_match']}: {count} samples"
            )

    # Process null items (complement)
    if complement_samples is not None:
        complement_mask = ~used_mask

        all_masks[complement_name].append({"mask": complement_mask, "samples": complement_samples})

        count = np.sum(complement_mask)
        logger.info(f"Complement mask for '{complement_name}': {count} samples")

    logger.info("Stratified masks created successfully")
    return all_masks


def _select_dissimilar_embeddings(embeddings: np.ndarray, k: int = 100) -> list[int]:
    """Select k dissimilar embeddings using farthest-first traversal."""
    n = embeddings.shape[0]

    if k >= n:
        return list(range(n))

    if k <= 0:
        return []

    selected_indices = []

    # Randomly pick the first point
    idx = np.random.choice(n)
    selected_indices.append(idx)

    # Precompute pairwise distances
    distances = pairwise_distances(embeddings, embeddings[[idx]], metric="euclidean").squeeze()

    for _ in range(1, k):
        # Select the point with the **max min-distance** to selected set
        next_idx = np.argmax(distances)
        selected_indices.append(next_idx)

        # Update min distances
        new_distances = pairwise_distances(
            embeddings, embeddings[[next_idx]], metric="euclidean"
        ).squeeze()
        distances = np.minimum(distances, new_distances)

    return selected_indices


def _get_selected_filenames(
    filenames_list: list[str], embeddings_array: np.ndarray, mask: np.ndarray, n_samples: int
) -> list[str]:
    """Get selected filenames from masked subset using dissimilar embeddings."""
    subset_indices = np.where(mask)[0]
    subset_filenames = [filenames_list[i] for i in subset_indices]
    subset_embeddings = embeddings_array[subset_indices]

    if len(subset_embeddings) == 0:
        logger.warning("Empty subset found, returning empty list")
        return []

    selected_subset_indices = _select_dissimilar_embeddings(subset_embeddings, k=n_samples)
    selected_filenames = [subset_filenames[i] for i in selected_subset_indices]

    logger.info(
        f"Selected {len(selected_filenames)} from {len(subset_filenames)} available samples"
    )
    return selected_filenames


def _process_split(
    subset_list: list[dict], filenames_list: list[str], embeddings_array: np.ndarray
) -> list[str]:
    """Process all subsets within a split and return selected filenames."""
    logger.info(f"Processing split with {len(subset_list)} subsets")

    split_selected_filenames = []

    for i, subset_config in enumerate(subset_list):
        selected_filenames = _get_selected_filenames(
            filenames_list,
            embeddings_array,
            subset_config["mask"],
            subset_config["samples"],
        )
        split_selected_filenames.extend(selected_filenames)
        logger.debug(f"Subset {i+1}: selected {len(selected_filenames)} filenames")

    logger.info(f"Split processing completed: {len(split_selected_filenames)} total filenames")
    return split_selected_filenames


def _copy_selected_filenames(
    ctx: BatchSelectionContext, selected_filenames: list[str], subset_type: str
) -> None:
    """Copy selected image and annotation files to subset directory."""
    logger.info(f"Copying {len(selected_filenames)} files to {subset_type} directory")

    # Create output directories
    output_subset_dir = ctx.output_dir / subset_type
    (output_subset_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_subset_dir / "annotations_oob").mkdir(parents=True, exist_ok=True)

    for filename in selected_filenames:
        try:
            # Copy images
            src = ctx.input_dir / "images" / filename
            dst = ctx.output_dir / subset_type / "images" / filename
            shutil.copy2(src, dst)

            # Copy annotations
            filename_stem = Path(filename).stem
            src = ctx.input_dir / "annotations" / f"{filename_stem}.xml"
            dst = ctx.output_dir / subset_type / "annotations_oob" / f"{filename_stem}.xml"
            shutil.copy2(src, dst)

        except FileNotFoundError as e:
            logger.error(f"File not found when copying {filename}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error copying {filename}: {e}")
            raise

    logger.info(f"Successfully copied {len(selected_filenames)} files to {subset_type} directory")


def select_initial_batch(ctx: BatchSelectionContext) -> None:
    """
    Select initial labelling batch for active learning.

    Args:
        ctx: BatchSelectionContext containing paths, model config, and runtime objects
    """
    logger.info("Starting initial batch selection")

    # Validate sampling structure loaded from script configuration
    _validate_sampling(ctx)

    # Load embeddings dataframe and get filenames and embeddings as different elements
    df_embed = _load_embeddings(ctx)
    filenames_list, embeddings_array = _get_filenames_and_embeddings(df_embed)

    # Calculate stratified masks foe each split + subset
    stratified_masks = _create_stratified_masks(ctx, filenames_list)

    # For each split, select frames and copy them into a new directory
    for split, subset_list in stratified_masks.items():
        split_selected_filenames = _process_split(subset_list, filenames_list, embeddings_array)
        _copy_selected_filenames(ctx, split_selected_filenames, split)

    logger.info("Initial batch selection completed successfully")
