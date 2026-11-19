"""Đọc ảnh và nhãn, biến đổi đồng thời ảnh và hộp bao."""

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

"x1, y1, x2, y2"
Box = tuple[float, float, float, float]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def image_to_tensor(image):
    """Đổi ảnh RGB sang tensor CHW và chuẩn hóa ImageNet."""
    import torch

    pixels = np.array(image, dtype=np.uint8)
    "[0,1]"
    tensor = torch.from_numpy(pixels).permute(2, 0, 1).to(torch.float32).div_(255)
    mean = tensor.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = tensor.new_tensor(IMAGENET_STD).view(3, 1, 1)
    "chuẩn hóa ảnh"
    return tensor.sub_(mean).div_(std)


def resize_sample(image: Image.Image, boxes: list[Box], size: int):
    """Resize về ảnh vuông và đổi tọa độ box theo từng trục."""
    scale_x = size / image.width
    scale_y = size / image.height
    resized_boxes = [
        (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)
        for x1, y1, x2, y2 in boxes
    ]
    return image.resize((size, size), Image.Resampling.BILINEAR), resized_boxes


def flip_sample(image: Image.Image, boxes: list[Box]):
    """Lật ngang ảnh và phản chiếu box theo trục dọc ở giữa ảnh."""
    width = image.width
    flipped_boxes = [(width - x2, y1, width - x1, y2) for x1, y1, x2, y2 in boxes]
    return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT), flipped_boxes


def load_annotations(json_path):
    """Đọc nhãn và nhóm box theo ID ảnh, kể cả ảnh không có box."""
    with Path(json_path).open(encoding="utf-8") as file:
        data = json.load(file)
    classes = data["classes"]
    "enumerate tạo (index, name)"
    class_indices = {name: index for index, name in enumerate(classes)}
    "dict"
    records = {
        image["id"]: {**image, "boxes": [], "labels": []}
        for image in data["images"]
    }
    for annotation in data["annotations"]:
        record = records[annotation["image_id"]]
        record["boxes"].append(tuple(map(float, annotation["bbox"])))
        "0,1,2..."
        record["labels"].append(class_indices[annotation["class"]])
    return classes, list(records.values())


def crop_sample(image, boxes, labels):
    """Cắt ngẫu nhiên và giữ các vật thể có tâm nằm trong vùng cắt."""
    if not boxes:
        return image, boxes, labels
    for _ in range(15):
        "chọn ngẫu nhiên tỉ lệ từ 0.5 đến 1"
        scale = random.uniform(0.5, 1.0)
        "round để làm tròn, đảm bảo vùng cắt >=1 pixel"
        width = max(1, round(image.width * scale))
        height = max(1, round(image.height * scale))
        "tọa độ ngang, dọc của góc trên trái"
        left = random.randint(0, image.width - width)
        top = random.randint(0, image.height - height)
        right, bottom = left + width, top + height
        selected_boxes, selected_labels = [], []

        for (x1, y1, x2, y2), label in zip(boxes, labels):
            "chỉ giữ vật thể nếu tâm hộp nằm trong vùng cắt"
            if left <= (x1 + x2) / 2 < right and top <= (y1 + y2) / 2 < bottom:
                "đổi sang hệ tọa độ mới"
                selected_boxes.append((
                    max(x1, left) - left, max(y1, top) - top,
                    min(x2, right) - left, min(y2, bottom) - top,
                ))
                selected_labels.append(label)
        if selected_boxes:
            return image.crop((left, top, right, bottom)), selected_boxes, selected_labels
    return image, boxes, labels


