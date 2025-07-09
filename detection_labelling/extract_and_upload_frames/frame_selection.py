import numpy as np
import supervision as sv
from utils.inference import get_ultralytics_detections

from detection_labelling.extract_and_upload_frames import (
    compute_conf_weight,
    compute_diversity_score,
)


def get_best_frame_with_det(
    frames_with_dets, quadrant_scorer, two_wheelers_ids, model, model_params, confidence_list
):
    best_frame_with_det = {}
    best_score = float("-inf")
    best_quadrant_indices = None
    for frame_idx, (frame, dets) in frames_with_dets.items():
        if dets is None:
            dets = get_ultralytics_detections(frame, model, model_params, confidence_list)
            frames_with_dets[frame_idx] = (frame, dets)

        # Calculate quadrant scores
        is_two_wheeler = np.isin(dets.class_id, two_wheelers_ids)
        dets_tw = dets[is_two_wheeler]
        two_wheeler_center = dets_tw.get_anchors_coordinates(sv.Position.CENTER)

        quadrant_indices = quadrant_scorer.get_quadrants(two_wheeler_center)
        quadrant_scores = quadrant_scorer.get_quadrant_scores(quadrant_indices)

        # Calculate confidence weights
        conf_weights = compute_conf_weight(dets_tw.confidence)

        # Calculate diversity score
        n_unique = len(np.unique(dets.class_id))
        diversity_score = compute_diversity_score(n_unique)

        # Calculate frame score
        frame_score = np.sum(quadrant_scores * conf_weights) + diversity_score

        if frame_score > best_score:
            best_frame_with_det["idx"] = frame_idx
            best_frame_with_det["frame"] = frame
            best_frame_with_det["dets"] = dets

            best_score = frame_score
            best_quadrant_indices = quadrant_indices

    return best_frame_with_det, best_quadrant_indices
