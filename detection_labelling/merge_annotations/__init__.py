# annotate_images/__init__.py
from .annotations_merging import merge_annotations
from .types import AnnMergingContext

__all__ = [
    # Image annotation
    "merge_annotations",
    # Data types
    "AnnMergingContext",
]