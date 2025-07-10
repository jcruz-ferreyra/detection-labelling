import numpy as np
import supervision as sv

from .frame_scoring import compute_conf_weight, compute_diversity_score


def get_best_frame_with_dets(
    frames_with_dets, quadrant_scorer, classes_of_interest
):
    best_frame_with_det = {}
    best_score = float("-inf")
    best_quadrant_indices = None
    for frame_idx, (frame, dets) in frames_with_dets.items():
        # Calculate quadrant scores
        is_of_interest = np.isin(dets.class_id, classes_of_interest)
        dets_oi = dets[is_of_interest]
        dets_oi_centers = dets_oi.get_anchors_coordinates(sv.Position.CENTER)

        quadrant_indices = quadrant_scorer.get_quadrants(dets_oi_centers)
        quadrant_scores = quadrant_scorer.get_quadrant_scores(quadrant_indices)

        # Calculate confidence weights
        conf_weights = compute_conf_weight(dets_oi.confidence)

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

    quadrant_scorer.update_quadrants_count(best_quadrant_indices)

    return best_frame_with_det
