# annotate_images/__init__.py
from .image_annotation import annotate_images
from .types import ImageAnnotationContext

__all__ = [
    # Image annotation
    "annotate_images",
    # Data types
    "ImageAnnotationContext",
]