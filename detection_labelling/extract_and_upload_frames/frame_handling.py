from collections import OrderedDict


class FrameHandler:
    """
    A class to handle frame processing for a video, identifying frames of interest
    and managing intermediate frames.

    Attributes:
        fps (int): Frames per second of the video.
        fps_of_interest (float): Number of frames per second to be marked as 'of interest'.
        n_intermediate (int): Number of previous intermediate frames to store.
        intermediate_step (float): Step size to calculate intermediate frames between frames of interest.
        intermediate_frames_idx (list): Indices of the most recent intermediate frames.
        intermediate_frames (dict): Stored index and frame for most recent intermediate frames.
    """

    def __init__(
        self,
        fps,
        fps_of_interest: float = 2.0,
        n_intermediate: int = 3,
        seconds_skip_after_save: int = 2,
        max_seconds_without_save: int = 10,
    ):
        """
        Initialize the FrameHandler.

        Args:
            fps (int): Frames per second of the video.
            fps_of_interest (int): Number of frames per second to be marked as 'of interest'.
            n_intermediate (int): Number of previous intermediate frames to store.
        """
        self.fps = fps
        self.fps_of_interest = fps_of_interest
        self.n_intermediate = n_intermediate

        self.current_frame = -1

        self.intermediate_step = (fps / fps_of_interest) / (n_intermediate + 1)
        self.intermediate_frames_idx = []
        self.intermediate_frames = OrderedDict()

        self.last_saved_frame_idx = -1
        self.seconds_skip_after_save = seconds_skip_after_save
        self.max_seconds_without_save = max_seconds_without_save

    def update_frame_counter(self):
        """
        Increment the frame counter by 1.

        This method keeps track of the current frame index being processed.
        It should be called whenever a new frame is processed to ensure the
        frame counter remains in sync with the frame processing workflow.
        """
        self.current_frame += 1

        return self.current_frame

    def is_frame_of_interest(self, idx):
        """
        Check if the frame is of interest.

        Args:
            idx (int): Frame index.

        Returns:
            bool: True if the frame is of interest, False otherwise.
        """
        # Check if the frame matches any of the frames of interest
        if isinstance(self.fps_of_interest, int):
            return any(
                (idx + int(self.fps // self.fps_of_interest) * i) % self.fps == 0
                for i in range(self.fps_of_interest)
            )
        else:
            return idx % int(self.fps // self.fps_of_interest) == 0

    def update_intermediate_frames_idx(self, idx):
        """
        Calculate the indices of following intermediate frames based on the current frame index.

        Args:
            idx (int): Current frame index.
        """
        self.intermediate_frames_idx = [
            idx + int(self.intermediate_step * (i + 1)) for i in range(self.n_intermediate)
        ]

    def update_intermediate_frames(self, idx, frame):
        """
        Store current frame and its index for later processing.

        Args:
            idx (int): The index of the frame.
            frame (Any): The frame data to store.
        """
        self.intermediate_frames[idx] = frame
        if len(self.intermediate_frames) > self.n_intermediate:
            self.intermediate_frames.popitem(last=False)

    def update_last_saved_frame_idx(self, idx):
        """
        Clear the list of stored intermediate frames.
        """
        self.last_saved_frame_idx = idx

        self.intermediate_frames = OrderedDict()

    def get_frames_since_last_save(self, idx):
        """
        Get the number of frames since the last save.
        """
        return idx - self.last_saved_frame_idx

    def _should_skip_frame_processing(self, idx) -> bool:
        """Check if frame processing should be skipped due to recent save."""
        frames_since_last_save = self.get_frames_since_last_save(idx)
        n_frames_threshold = self.fps * self.seconds_skip_after_save

        return frames_since_last_save < n_frames_threshold

    def is_long_time_since_last_save(self, idx: int) -> bool:
        """
        Checks if more than max_seconds_without_save have passed since the last save.
        """
        if self.last_saved_frame_idx == -1:
            return True
        frames_since_last_save = self.get_frames_since_last_save(idx)

        return frames_since_last_save > self.fps * self.max_seconds_without_save
