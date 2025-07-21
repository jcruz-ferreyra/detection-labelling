import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from xml.etree.ElementTree import Element, SubElement, parse

import cv2
from defusedxml.ElementTree import tostring
from defusedxml.minidom import parseString
import numpy as np
import supervision as sv
from supervision.dataset.formats.pascal_voc import detections_from_xml_obj
from supervision.dataset.formats.pascal_voc import detections_to_pascal_voc


# def object_to_pascal_voc(
#     xyxy: np.ndarray, name: str, polygon: Optional[np.ndarray] = None
# ) -> Element:
#     root = Element("object")

#     object_name = SubElement(root, "name")
#     object_name.text = name

#     # https://github.com/roboflow/supervision/issues/144
#     xyxy += 1

#     bndbox = SubElement(root, "bndbox")
#     xmin = SubElement(bndbox, "xmin")
#     xmin.text = str(int(xyxy[0]))
#     ymin = SubElement(bndbox, "ymin")
#     ymin.text = str(int(xyxy[1]))
#     xmax = SubElement(bndbox, "xmax")
#     xmax.text = str(int(xyxy[2]))
#     ymax = SubElement(bndbox, "ymax")
#     ymax.text = str(int(xyxy[3]))

#     if polygon is not None:
#         # https://github.com/roboflow/supervision/issues/144
#         polygon += 1
#         object_polygon = SubElement(root, "polygon")
#         for index, point in enumerate(polygon, start=1):
#             x_coordinate, y_coordinate = point
#             x = SubElement(object_polygon, f"x{index}")
#             x.text = str(x_coordinate)
#             y = SubElement(object_polygon, f"y{index}")
#             y.text = str(y_coordinate)

#     return root


# def detections_to_pascal_voc(
#     detections: sv.Detections,
#     classes: List[str],
#     filename: str,
#     image_shape: Tuple[int, int, int],
# ) -> str:
#     """
#     Converts Detections object to Pascal VOC XML format.

#     Args:
#         detections (Detections): A Detections object containing bounding boxes,
#             class ids, and other relevant information.
#         classes (List[str]): A list of class names corresponding to the
#             class ids in the Detections object.
#         filename (str): The name of the image file associated with the detections.
#         image_shape (Tuple[int, int, int]): The shape of the image
#             file associated with the detections.
#         min_image_area_percentage (float): Minimum detection area
#             relative to area of image associated with it.
#         max_image_area_percentage (float): Maximum detection area
#             relative to area of image associated with it.
#         approximation_percentage (float): The percentage of
#             polygon points to be removed from the input polygon, in the range [0, 1).
#     Returns:
#         str: An XML string in Pascal VOC format representing the detections.
#     """
#     height, width, depth = image_shape

#     # Create root element
#     annotation = Element("annotation")

#     # Add folder element
#     folder = SubElement(annotation, "folder")
#     folder.text = "VOC"

#     # Add filename element
#     file_name = SubElement(annotation, "filename")
#     file_name.text = filename

#     # Add source element
#     source = SubElement(annotation, "source")
#     database = SubElement(source, "database")
#     database.text = "roboflow.ai"

#     # Add size element
#     size = SubElement(annotation, "size")
#     w = SubElement(size, "width")
#     w.text = str(width)
#     h = SubElement(size, "height")
#     h.text = str(height)
#     d = SubElement(size, "depth")
#     d.text = str(depth)

#     # Add segmented element
#     segmented = SubElement(annotation, "segmented")
#     segmented.text = "0"

#     # Add object elements
#     for xyxy, _, _, class_id, _, _ in detections:
#         name = classes[class_id]
#         next_object = object_to_pascal_voc(xyxy=xyxy, name=name)
#         annotation.append(next_object)

#     # Generate XML string
#     xml_string = parseString(tostring(annotation)).toprettyxml(indent="  ")
#     return xml_string


def save_as_pascal_voc(
    frame: np.ndarray,
    detections: sv.Detections,
    output_dir: Path,
    filename_stem: str,
    class_label: Dict[int, str],
) -> None:
    image_filename = f"{filename_stem}.png"
    image_path = output_dir / "images" / image_filename
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
    xml_path = output_dir / "annotations" / xml_filename

    with open(xml_path, "w", encoding="utf-8") as xml_file:
        xml_file.write(voc_xml)


def load_single_pascal_voc_annotation(
    annotation_path: str,
    image_path: str,
    classes: List[str],
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
