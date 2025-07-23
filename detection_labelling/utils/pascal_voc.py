import os
from pathlib import Path
from typing import Dict, List, Tuple
from xml.etree.ElementTree import parse

import cv2
import numpy as np
import supervision as sv
from supervision.dataset.formats.pascal_voc import (
    detections_from_xml_obj,
    detections_to_pascal_voc,
)


def save_as_pascal_voc(
    frame: np.ndarray,
    detections: sv.Detections,
    output_dir: Path,
    filename_stem: str,
    class_label: Dict[int, str],
    save_image: bool = True,
) -> None:
    image_filename = f"{filename_stem}.png"
    image_path = output_dir / "images" / image_filename

    if save_image:
        cv2.imwrite(str(image_path), frame)

    max_key = max(int(key) for key in class_label.keys())
    list_classes = [class_label.get(i, "none") for i in range(max_key + 1)]

    voc_xml = detections_to_pascal_voc(
        detections=detections,
        classes=list_classes,
        filename=image_filename,
        image_shape=frame.shape,
    )

    xml_filename = f"{filename_stem}.xml"
    xml_path = output_dir / "annotations_oob" / xml_filename

    with open(xml_path, "w", encoding="utf-8") as xml_file:
        xml_file.write(voc_xml)


def load_pascal_voc_annotation(
    image_path: str,
    annotations_directory_path: str,
    force_masks: bool = False,
) -> Tuple[sv.Detections, List[str]]:
    """
    Load a single PASCAL VOC XML annotation file.

    Args:
        annotation_path: Path to the XML annotation file
        image_path: Path to the corresponding image file
        classes: Existing list of class names (will be updated)
        force_masks: Whether to force mask loading

    Returns:
        Tuple of (Detections, updated_classes_list)
    """
    image_stem = Path(image_path).stem
    annotation_path = os.path.join(annotations_directory_path, f"{image_stem}.xml")

    if not os.path.exists(annotation_path):
        return sv.Detections.empty(), classes

    # Get image resolution
    image = cv2.imread(image_path)
    if image is None:
        return sv.Detections.empty(), classes

    resolution_wh = (image.shape[1], image.shape[0])

    # Parse XML
    tree = parse(annotation_path)
    root = tree.getroot()

    # Use supervision's internal function
    annotation, updated_classes = detections_from_xml_obj(
        root, classes, resolution_wh, force_masks
    )

    return annotation, updated_classes
