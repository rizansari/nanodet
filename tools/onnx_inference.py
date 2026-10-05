#!/usr/bin/env python3
# Copyright 2021 RangiLyu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
ONNX Runtime inference for NanoDet.

The exported graph returns one tensor per image:
    [batch, num_points, num_classes + 4 * (reg_max + 1)]
Class scores are already passed through sigmoid. The remaining channels are
the distribution-focal box distances, decoded the same way as NanoDetPlusHead.

usage

    python tools/onnx_inference.py \
        -m models/nanodet.onnx \
        --source /path/to/images \
        --dest demo_output \
        -s 0.35 \
        --input_shape "320,320"

    python tools/onnx_inference.py \
        -m models/nanodet.onnx \
        -i /path/to/image.jpg \
        -o demo_output \
        -s 0.35 \
        --input_shape "320,320"

    python tools/onnx_inference.py \
        -m models/nanodet.onnx \
        --config config/nanodet-plus-m_320.yml \
        --source /path/to/images \
        --dest demo_output
"""

import argparse
import math
import os
import time

import cv2
import numpy as np
import onnxruntime

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_MODEL = os.path.join(REPO_ROOT, "models", "nanodet.onnx")

# COCO mean/std in BGR, matching config data.*.pipeline.normalize.
MEAN = np.array([103.53, 116.28, 123.675], dtype=np.float32)
STD = np.array([57.375, 57.12, 58.395], dtype=np.float32)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

COCO_CLASSES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic_light",
    "fire_hydrant",
    "stop_sign",
    "parking_meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports_ball",
    "kite",
    "baseball_bat",
    "baseball_glove",
    "skateboard",
    "surfboard",
    "tennis_racket",
    "bottle",
    "wine_glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot_dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted_plant",
    "bed",
    "dining_table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell_phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy_bear",
    "hair_drier",
    "toothbrush",
)


def make_parser():
    parser = argparse.ArgumentParser("onnxruntime inference sample")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Input your onnx model.",
    )
    parser.add_argument(
        "-i",
        "--image_path",
        "--source",
        type=str,
        required=True,
        help="Path to an input image or a folder of images.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        "--dest",
        type=str,
        default="demo_output",
        help="Folder where annotated images are saved.",
    )
    parser.add_argument(
        "-s",
        "--score_thr",
        type=float,
        default=0.35,
        help="Score threshold to filter the result.",
    )
    parser.add_argument(
        "--nms_thr",
        type=float,
        default=0.6,
        help="IoU threshold used by class-wise NMS.",
    )
    parser.add_argument(
        "--input_shape",
        type=str,
        default=None,
        help="Model input as height,width. Default is the ONNX spatial size, "
        "or 320,320 when that size is dynamic.",
    )
    parser.add_argument(
        "--strides",
        type=str,
        default=None,
        help="Feature strides used to decode center priors. "
        "Default is the config value, or 8,16,32,64.",
    )
    parser.add_argument(
        "--reg_max",
        type=int,
        default=None,
        help="Distribution focal regression max. "
        "Default is the config value, or 7.",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=None,
        help="Number of categories, excluding background. "
        "Default is the config value, or 80.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional NanoDet yml. Fills strides, reg_max, classes, "
        "normalize, keep_ratio, and input size when those flags are omitted.",
    )
    parser.add_argument(
        "--keep_ratio",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Keep aspect ratio and center the image. Default follows --config, "
        "otherwise stretch to the input size.",
    )
    return parser


def list_images(source_dir):
    paths = []
    for name in sorted(os.listdir(source_dir)):
        path = os.path.join(source_dir, name)
        ext = os.path.splitext(name)[1].lower()
        if os.path.isfile(path) and ext in IMAGE_EXTS:
            paths.append(path)
    return paths


def load_yaml_config(path):
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def apply_config(args):
    mean = MEAN.copy()
    std = STD.copy()
    class_names = COCO_CLASSES
    if args.config:
        cfg = load_yaml_config(args.config)
        head = cfg["model"]["arch"]["head"]
        val = cfg["data"]["val"]
        if args.num_classes is None:
            args.num_classes = int(head["num_classes"])
        if args.reg_max is None:
            args.reg_max = int(head["reg_max"])
        if args.strides is None:
            args.strides = ",".join(str(int(item)) for item in head["strides"])
        if args.input_shape is None:
            width, height = val["input_size"]
            args.input_shape = "{},{}".format(int(height), int(width))
        if args.keep_ratio is None:
            args.keep_ratio = bool(val.get("keep_ratio", False))
        normalize = val["pipeline"]["normalize"]
        mean = np.array(normalize[0], dtype=np.float32)
        std = np.array(normalize[1], dtype=np.float32)
        if cfg.get("class_names"):
            class_names = tuple(cfg["class_names"])
    if args.num_classes is None:
        args.num_classes = 80
    if args.reg_max is None:
        args.reg_max = 7
    if args.strides is None:
        args.strides = "8,16,32,64"
    return mean, std, class_names


def parse_hw(input_shape):
    try:
        height, width = tuple(map(int, input_shape.split(",")))
    except (AttributeError, TypeError, ValueError):
        raise ValueError(
            "Invalid --input_shape '{}'. Expected 'height,width', e.g. 320,320.".format(
                input_shape
            )
        )
    if height <= 0 or width <= 0:
        raise ValueError("--input_shape values must be positive integers.")
    return height, width


def parse_strides(strides):
    values = tuple(int(item) for item in strides.split(",") if item.strip())
    if not values or any(item <= 0 for item in values):
        raise ValueError("--strides must be a comma-separated list of positive integers.")
    return values


def create_session(model_path):
    """Prefer CoreML, and use CPU when that provider is missing or fails to load."""
    available = onnxruntime.get_available_providers()
    print("Available providers: {}".format(available))

    if "CoreMLExecutionProvider" in available:
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    else:
        print("CoreMLExecutionProvider is not available; using CPUExecutionProvider.")
        providers = ["CPUExecutionProvider"]

    try:
        session = onnxruntime.InferenceSession(model_path, providers=providers)
    except Exception as exc:
        if "CoreMLExecutionProvider" not in providers:
            raise
        print(
            "CoreMLExecutionProvider failed to load ({}); using CPUExecutionProvider.".format(
                exc
            )
        )
        session = onnxruntime.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )

    active = session.get_providers()
    if "CoreMLExecutionProvider" in providers and "CoreMLExecutionProvider" not in active:
        print("CoreMLExecutionProvider did not stay active; using {}.".format(active))
    else:
        print("ONNX Runtime providers: {}".format(active))
    return session


def image_model_input(session):
    inputs = session.get_inputs()
    for item in inputs:
        if item.name == "data":
            return item
    for item in inputs:
        shape = item.shape
        if isinstance(shape, (list, tuple)) and len(shape) == 4:
            return item
    return inputs[0]


def is_static_dim(dim):
    return isinstance(dim, int) and dim > 0


def resolve_input_hw(model_input, input_shape):
    shape = list(model_input.shape)
    if input_shape is not None:
        return parse_hw(input_shape)
    if len(shape) == 4 and is_static_dim(shape[2]) and is_static_dim(shape[3]):
        return int(shape[2]), int(shape[3])
    return 320, 320


def get_resize_matrix(raw_shape, dst_shape, keep_ratio):
    """Map raw (width, height) onto dst (width, height). Same as nanodet warp."""
    raw_w, raw_h = raw_shape
    dst_w, dst_h = dst_shape
    resize = np.eye(3, dtype=np.float32)
    if keep_ratio:
        center = np.eye(3, dtype=np.float32)
        center[0, 2] = -raw_w / 2
        center[1, 2] = -raw_h / 2
        if raw_w / raw_h < dst_w / dst_h:
            ratio = dst_h / raw_h
        else:
            ratio = dst_w / raw_w
        resize[0, 0] *= ratio
        resize[1, 1] *= ratio
        translate = np.eye(3, dtype=np.float32)
        translate[0, 2] = 0.5 * dst_w
        translate[1, 2] = 0.5 * dst_h
        return translate @ resize @ center
    resize[0, 0] *= dst_w / raw_w
    resize[1, 1] *= dst_h / raw_h
    return resize


def warp_boxes(boxes, matrix, width, height):
    count = len(boxes)
    if count == 0:
        return boxes
    xy = np.ones((count * 4, 3), dtype=np.float32)
    xy[:, :2] = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(count * 4, 2)
    xy = xy @ matrix.T
    xy = (xy[:, :2] / xy[:, 2:3]).reshape(count, 8)
    x = xy[:, [0, 2, 4, 6]]
    y = xy[:, [1, 3, 5, 7]]
    warped = np.stack((x.min(1), y.min(1), x.max(1), y.max(1)), axis=1)
    warped[:, [0, 2]] = warped[:, [0, 2]].clip(0, width)
    warped[:, [1, 3]] = warped[:, [1, 3]].clip(0, height)
    return warped.astype(np.float32)


def preprocess(origin_img, input_hw, keep_ratio, mean, std):
    dst_h, dst_w = input_hw
    src_h, src_w = origin_img.shape[:2]
    matrix = get_resize_matrix((src_w, src_h), (dst_w, dst_h), keep_ratio)
    resized = cv2.warpPerspective(origin_img, matrix, dsize=(dst_w, dst_h))
    blob = resized.astype(np.float32)
    blob = (blob - mean.reshape(1, 1, 3)) / std.reshape(1, 1, 3)
    blob = np.transpose(blob, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(blob), matrix


def center_priors(input_hw, strides):
    height, width = input_hw
    levels = []
    for stride in strides:
        feat_h = math.ceil(height / stride)
        feat_w = math.ceil(width / stride)
        ys = np.arange(feat_h, dtype=np.float32) * stride
        xs = np.arange(feat_w, dtype=np.float32) * stride
        grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
        stride_arr = np.full(grid_x.size, stride, dtype=np.float32)
        levels.append(
            np.stack(
                [grid_x.ravel(), grid_y.ravel(), stride_arr, stride_arr], axis=-1
            )
        )
    return np.concatenate(levels, axis=0)


def softmax(x, axis=-1):
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def distribution_project(reg_preds, reg_max):
    bins = reg_max + 1
    project = np.arange(bins, dtype=np.float32)
    dist = reg_preds.reshape(-1, 4, bins)
    dist = softmax(dist, axis=-1)
    return (dist * project).sum(axis=-1)


def distance2bbox(points, distance, max_shape):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    height, width = max_shape
    x1 = np.clip(x1, 0, width)
    y1 = np.clip(y1, 0, height)
    x2 = np.clip(x2, 0, width)
    y2 = np.clip(y2, 0, height)
    return np.stack([x1, y1, x2, y2], axis=-1)


def nms(boxes, scores, iou_thr):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        iou = inter / np.maximum(union, 1e-6)
        order = rest[iou <= iou_thr]
    return np.asarray(keep, dtype=np.int64)


def multiclass_nms(boxes, scores, score_thr, nms_thr, max_num=100):
    kept_boxes = []
    kept_scores = []
    kept_labels = []
    for label in range(scores.shape[1]):
        cls_scores = scores[:, label]
        mask = cls_scores > score_thr
        if not np.any(mask):
            continue
        cls_boxes = boxes[mask]
        cls_scores = cls_scores[mask]
        keep = nms(cls_boxes, cls_scores, nms_thr)
        kept_boxes.append(cls_boxes[keep])
        kept_scores.append(cls_scores[keep])
        kept_labels.append(np.full(keep.shape[0], label, dtype=np.int64))
    if not kept_boxes:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )
    boxes = np.concatenate(kept_boxes, axis=0)
    scores = np.concatenate(kept_scores, axis=0)
    labels = np.concatenate(kept_labels, axis=0)
    order = scores.argsort()[::-1][:max_num]
    return boxes[order], scores[order], labels[order]


def decode_predictions(preds, input_hw, strides, num_classes, reg_max, score_thr, nms_thr):
    if preds.ndim == 3:
        preds = preds[0]
    expected = sum(
        math.ceil(input_hw[0] / stride) * math.ceil(input_hw[1] / stride)
        for stride in strides
    )
    if preds.shape[0] != expected:
        raise ValueError(
            "Prediction has {} points, but strides {} on input {} produce {}. "
            "Pass matching --strides and --input_shape.".format(
                preds.shape[0], strides, input_hw, expected
            )
        )
    reg_channels = 4 * (reg_max + 1)
    if preds.shape[-1] != num_classes + reg_channels:
        raise ValueError(
            "Prediction channels are {}, expected num_classes {} + 4 * (reg_max + 1) "
            "= {}.".format(preds.shape[-1], num_classes, num_classes + reg_channels)
        )
    cls_scores = preds[:, :num_classes]
    reg_preds = preds[:, num_classes:]
    priors = center_priors(input_hw, strides)
    distance = distribution_project(reg_preds, reg_max) * priors[:, 2:3]
    boxes = distance2bbox(priors[:, :2], distance, input_hw)
    return multiclass_nms(boxes, cls_scores, score_thr, nms_thr)


def color_for(label):
    rng = np.random.default_rng(label + 1)
    color = rng.integers(64, 256, size=3)
    return int(color[0]), int(color[1]), int(color[2])


def draw_detections(image, boxes, scores, labels, class_names):
    canvas = image.copy()
    order = scores.argsort()
    for index in order:
        x0, y0, x1, y1 = [int(v) for v in boxes[index]]
        label = int(labels[index])
        score = float(scores[index])
        name = class_names[label] if label < len(class_names) else str(label)
        color = color_for(label)
        text = "{}:{:.1f}%".format(name, score * 100)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1
        text_w, text_h = cv2.getTextSize(text, font, scale, thickness)[0]
        text_y = y0 - 2 if y0 - text_h - 4 >= 0 else y0 + text_h + 2
        cv2.rectangle(
            canvas,
            (x0, text_y - text_h - 2),
            (x0 + text_w + 2, text_y + 2),
            color,
            -1,
        )
        cv2.putText(
            canvas, text, (x0 + 1, text_y), font, scale, (0, 0, 0), thickness, cv2.LINE_AA
        )
    return canvas


def annotate(
    session,
    input_name,
    origin_img,
    input_hw,
    score_thr,
    nms_thr,
    strides,
    num_classes,
    reg_max,
    class_names,
    keep_ratio,
    mean,
    std,
):
    blob, matrix = preprocess(origin_img, input_hw, keep_ratio, mean, std)
    output = session.run(None, {input_name: blob})
    boxes, scores, labels = decode_predictions(
        output[0], input_hw, strides, num_classes, reg_max, score_thr, nms_thr
    )
    if len(boxes):
        src_h, src_w = origin_img.shape[:2]
        boxes = warp_boxes(boxes, np.linalg.inv(matrix), src_w, src_h)
    return draw_detections(origin_img, boxes, scores, labels, class_names), len(boxes)


def process_images(session, image_paths, output_dir, runtime):
    input_name = image_model_input(session).name
    saved = 0
    detections = 0
    start = time.perf_counter()
    for image_path in image_paths:
        origin_img = cv2.imread(image_path)
        if origin_img is None:
            print("skip unreadable image: {}".format(image_path))
            continue
        annotated, count = annotate(
            session,
            input_name,
            origin_img,
            runtime["input_hw"],
            runtime["score_thr"],
            runtime["nms_thr"],
            runtime["strides"],
            runtime["num_classes"],
            runtime["reg_max"],
            runtime["class_names"],
            runtime["keep_ratio"],
            runtime["mean"],
            runtime["std"],
        )
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(output_path, annotated)
        saved += 1
        detections += count
        print("saved {} ({} detections)".format(output_path, count))
    elapsed = time.perf_counter() - start
    images_per_sec = saved / elapsed if elapsed > 0 else 0.0
    print(
        "Processed {} images in {:.3f} s ({:.2f} images/s, {} detections)".format(
            saved, elapsed, images_per_sec, detections
        )
    )


def main():
    args = make_parser().parse_args()
    mean, std, class_names = apply_config(args)
    if args.keep_ratio is None:
        args.keep_ratio = False
    if not os.path.isfile(args.model):
        raise SystemExit("ONNX model not found: {}".format(args.model))

    onnxruntime.set_default_logger_severity(3)
    session = create_session(args.model)
    model_input = image_model_input(session)
    input_hw = resolve_input_hw(model_input, args.input_shape)
    strides = parse_strides(args.strides)
    if args.num_classes <= 0 or args.reg_max < 0:
        raise SystemExit("--num_classes must be positive and --reg_max must be >= 0.")
    if len(class_names) < args.num_classes:
        class_names = class_names + tuple(
            "class_{}".format(i) for i in range(len(class_names), args.num_classes)
        )

    if os.path.isdir(args.image_path):
        image_paths = list_images(args.image_path)
        if not image_paths:
            raise SystemExit("No images found in {}".format(args.image_path))
    elif os.path.isfile(args.image_path):
        image_paths = [args.image_path]
    else:
        raise SystemExit("Input not found: {}".format(args.image_path))

    os.makedirs(args.output_dir, exist_ok=True)
    print("model: {}".format(os.path.abspath(args.model)))
    print("input: {}  shape: {}".format(model_input.name, input_hw))
    runtime = {
        "input_hw": input_hw,
        "score_thr": args.score_thr,
        "nms_thr": args.nms_thr,
        "strides": strides,
        "num_classes": args.num_classes,
        "reg_max": args.reg_max,
        "class_names": class_names,
        "keep_ratio": args.keep_ratio,
        "mean": mean,
        "std": std,
    }
    process_images(session, image_paths, args.output_dir, runtime)


if __name__ == "__main__":
    main()
