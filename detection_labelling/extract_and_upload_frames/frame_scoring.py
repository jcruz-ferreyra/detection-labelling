from functools import lru_cache

import numpy as np


class QuadrantScorer:
    def __init__(self, grid_size, frame_shape, k=1):
        """
        Initializes the QuadrantScorer with grid size, frame shape, and constant k.

        :param grid_size: Tuple (rows, cols) defining the number of grid partitions (e.g., 3x3).
        :param frame_shape: Tuple (width, height) of the frame.
        :param k: Constant to use in the score calculation. Default is 1.
        """
        self.rows, self.cols = grid_size
        self.width, self.height = frame_shape
        self.k = k

        # Calculate horizontal and vertical lines dividing the quadrants
        self.h_lines = np.linspace(0, self.height, self.rows + 1)[:-1]
        self.v_lines = np.linspace(0, self.width, self.cols + 1)[:-1]

        # Initialize counts and scores for dets per quadrant
        self.num_quadrants = self.rows * self.cols
        self.quadrant_counts = np.zeros(self.num_quadrants, dtype=int)
        self.quadrant_scores = np.zeros_like(self.quadrant_counts, dtype=float)

        # Precompute initial scores
        self._update_all_scores()

    def update_quadrants_count(self, quadrant_indices):
        """
        Updates the count of detections saved by quadrant and recalculates scores.

        :param quadrant_indices: Array of quadrant indices for detections to update counts.
        """
        # Record the maximum count before updating
        previous_max = self.quadrant_counts.max()

        # Update the counts for the specified quadrants
        for index in quadrant_indices:
            self.quadrant_counts[index] += 1

        # Check if the maximum count has changed
        current_max = self.quadrant_counts.max()

        if current_max != previous_max:
            # If the maximum changed, update all scores
            self._update_all_scores()
        else:
            # Otherwise, update only the quadrants that were changed
            self._update_specific_scores(quadrant_indices)

    def _update_all_scores(self):
        """Recalculates scores for all quadrants."""
        max_count = self.quadrant_counts.max()
        for idx in range(self.num_quadrants):
            self.quadrant_scores[idx] = self._calculate_score(max_count, self.quadrant_counts[idx])

    def _update_specific_scores(self, quadrant_indices):
        """Recalculates scores only for the updated quadrants."""
        max_count = self.quadrant_counts.max()
        for index in quadrant_indices:
            self.quadrant_scores[index] = self._calculate_score(
                max_count, self.quadrant_counts[index]
            )

    def get_quadrants(self, points):
        """
        Determines the quadrant (1D index) for each detection based on its center point.

        :param points: List or numpy array of tuples (x, y) representing detection centers.
        :return: Numpy array of 1D quadrant indices.
        """
        points = np.array(points)
        x_coords, y_coords = points[:, 0], points[:, 1]

        row_indices = np.digitize(y_coords, self.h_lines) - 1
        col_indices = np.digitize(x_coords, self.v_lines) - 1

        # Convert 2D (row, col) indices to 1D indices
        return row_indices * self.cols + col_indices

    def get_quadrant_scores(self, quadrant_indices=None):
        """
        Returns the scores for specified quadrants or all quadrants.

        :param quadrant_indices: Optional array of 1D quadrant indices. If None, return all scores.
        :return: 1D numpy array of scores for the specified quadrants or all quadrants.
        """
        if quadrant_indices is None:
            # If no specific quadrants are requested, return all scores
            return self.quadrant_scores
        else:
            # Return the scores for the specified quadrant indices
            return self.quadrant_scores[quadrant_indices]

    @lru_cache(maxsize=8)
    def _calculate_score(self, h, c):
        """
        Calculates the score for a single quadrant based on the formula.

        :param h: Highest count of saved detections in any quadrant.
        :param c: Count of saved detections in the current quadrant.
        :param k: Constant k used in the formula.
        :return: Calculated score.
        """
        return np.log(1 + ((h + self.k) / (c + self.k)))


def compute_conf_weight(conf):
    return np.log(1 + 1 / (conf + 1))


def compute_diversity_score(n_unique):
    return n_unique * 0.03
