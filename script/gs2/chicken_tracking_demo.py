import json
import os
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pycocotools.mask as mask_util
import supervision as sv
import torch
from PIL import Image
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from tqdm import tqdm
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images

from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--gs2-repo-path",
        default="/home/prince/proj/chicken-behaviour-classifier/Grounded-SAM-2",
        type=str,
        help="Path to Grounded-SAM-2 repo",
    )
    parser.add_argument(
        "-i",
        "--video-dir",
        type=str,
        required=True,
        help="Directory of JPEG frames named <frame_index>.jpg",
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Text queries (lowercase, ending with a dot)",
    )
    parser.add_argument(
        "-m",
        "--mask-file",
        type=str,
        help="Path to .json file containing masks",
    )
    parser.add_argument("--fps", type=int, default=25, help="FPS of output video")
    parser.add_argument(
        "--size",
        type=str,
        choices=("large", "base_plus", "small", "tiny"),
        default="large",
        help="SAM2 model size",
    )
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=0.25,
        help="Box threshold for GroundingDINO",
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.3,
        help="Text threshold for GroundingDINO",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="CUDA GPU index to use (e.g. 0 or 1). Defaults to 0.",
    )
    parser.add_argument(
        "-o",
        "--save-dir",
        type=str,
        default="./tracking_results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--save-masks", action="store_true", help="Save mask JSON files"
    )
    parser.add_argument(
        "--save-images", action="store_true", help="Save annotated JPGs"
    )
    parser.add_argument(
        "--create-video", action="store_true", help="Create annotated video"
    )

    return parser.parse_args()


def resolve_device(gpu_id: int):
    """Safely select GPU device or fall back to CPU."""
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if gpu_id < num_gpus:
            print(f"[INFO] Using GPU {gpu_id} / {num_gpus} available GPUs")
            return torch.device(f"cuda:{gpu_id}")
        else:
            print(
                f"[WARN] Requested GPU {gpu_id} but only {num_gpus} GPUs available. Falling back to CPU."
            )
    else:
        print("[WARN] CUDA not available, running on CPU.")

    return torch.device("cpu")


def single_mask_to_rle(mask):
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


# -----------------------
# Begin main script logic
# -----------------------
args = parse_args()
if args.create_video:
    assert args.create_video and args.save_masks, (
        "Must invoke --create-video MUST be invoked together with --save-images"
    )
device = resolve_device(args.gpu_id)
print(f"Input device resolved to: {device}")

if device.type == "cuda":
    torch.cuda.set_device(device)  # or torch.cuda.set_device(args.gpu_id)
    print("[DEBUG] torch.cuda.current_device() =", torch.cuda.current_device())


gs2_repo_path = Path(args.gs2_repo_path)
assert gs2_repo_path.exists(), "Grounded-SAM-2 repo path not found."

# Mixed precision setup
torch.autocast(
    device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.bfloat16
).__enter__()

if device.type == "cuda" and torch.cuda.get_device_properties(device.index).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Load SAM2
# sam2_checkpoint = f"./checkpoints/sam2.1_hiera_{args.size}.pt"
sam2_checkpoint = Path(
    gs2_repo_path, f"checkpoints/sam2.1_hiera_{args.size}.pt"
).as_posix()
model_cfg_size = args.size[0].split("_")[0].lower()
if "plus" in args.size:
    model_cfg_size += "+"
model_cfg = f"configs/sam2.1/sam2.1_hiera_{model_cfg_size}.yaml"

video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
image_predictor = SAM2ImagePredictor(sam2_image_model)

# Load GroundingDINO on selected device
model_id = "IDEA-Research/grounding-dino-tiny"
processor = AutoProcessor.from_pretrained(model_id)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(
    device
)


# setup the input image and text prompt for SAM 2 and Grounding DINO
# VERY important: text queries need to be lowercased + end with a dot
text = args.text  # "chicken.bird."

# `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
video_dir = args.video_dir  #  "/home/nclow23/src/chicken/tmp/b1_20241019_2001"

# scan all the JPEG frame names in this directory
frame_names = [
    p
    for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# init video predictor state
inference_state = video_predictor.init_state(
    video_path=video_dir, offload_video_to_cpu=False
)

ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = (
    1  # give a unique id to each object we interact with (it can be any integers)
)


###
# Step 2: Prompt Grounding DINO and SAM image predictor to get the box and mask for specific frame
###

if args.mask_file is None:
    # prompt grounding dino to get the box coordinates on specific frame
    # img_path = os.path.join(video_dir, frame_names[ann_frame_idx])
    img_path = Path(video_dir, frame_names[ann_frame_idx]).resolve().as_posix()
    image = Image.open(img_path)

    # run Grounding DINO on the image
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        # box_threshold=args.box_threshold,
        threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        target_sizes=[image.size[::-1]],
    )

    # prompt SAM image predictor to get the mask for the object
    image_predictor.set_image(np.array(image.convert("RGB")))

    # process the detection results
    input_boxes = results[0]["boxes"].cpu().numpy()  # (K, 4) or (0, 4)
    dino_scores = results[0]["scores"].cpu().numpy()  # numpy array of scores
    labels = results[0]["labels"]  # e.g., ['chicken', 'bird', ...]
    OBJECTS = [str(l) for l in labels]  # keep class names as-is

    # Early exit if no detections (avoid SAM2 call altogether)
    if input_boxes.shape[0] == 0:
        print(
            f"[WARN] No detections for text='{text}' at thresholds "
            f"box={args.box_threshold}, text={args.text_threshold}."
        )
        masks = np.zeros((0, image.size[1], image.size[0]), dtype=np.uint8)  # (0, H, W)
        scores = np.zeros((0,), dtype=np.float32)
        logits = np.zeros((0, 1, image.size[1], image.size[0]), dtype=np.float32)
        PROMPT_TYPE_FOR_VIDEO = "box"  # we still use box prompts downstream
    else:
        # Run SAM2 per box to keep batch=1 (avoid assertion)
        all_masks = []
        all_scores = []
        all_logits = []

        for k in range(input_boxes.shape[0]):
            box_k = np.asarray(input_boxes[k], dtype=np.float32)[None, ...]  # (1, 4)

            m_k, s_k, l_k = image_predictor.predict(
                point_coords=None,  # batch=1
                point_labels=None,
                box=box_k,  # (1, 4)
                multimask_output=False,
            )

            # m_k: (1, 1, H, W) or (1, H, W) depending on predictor version
            # Standardize to (H, W) per detection for downstream code
            if m_k.ndim == 4:  # (B=1, 1, H, W) -> (H, W)
                m_k = m_k.squeeze(0).squeeze(0)
            elif m_k.ndim == 3:  # (B=1, H, W) -> (H, W)
                m_k = m_k.squeeze(0)

            all_masks.append(m_k)
            all_scores.append(np.array(s_k).reshape(-1))  # to (<=1,) per detection
            all_logits.append(l_k)  # keep original shape; you use logits later

        # Stack masks to (K, H, W)
        masks = np.stack(all_masks, axis=0)
        # Concatenate scores/logits across detections
        scores = (
            np.concatenate(all_scores, axis=0)
            if len(all_scores) > 0
            else np.zeros((0,), dtype=np.float32)
        )
        # If logits have shape (1, 1, H, W) per detection, stack to (K, 1, H, W)
        try:
            logits = np.concatenate(all_logits, axis=0)
        except Exception:
            # Fallback: stack if concatenate fails due to shape diff
            logits = np.stack([np.asarray(l) for l in all_logits], axis=0)

        PROMPT_TYPE_FOR_VIDEO = "box"  # or "point" as needed

else:
    with open(args.mask_file, "r", encoding="utf-8") as f:
        annotations = json.load(f)["annotations"]
    masks = [mask_util.decode(annotation["segmentation"]) for annotation in annotations]
    # masks = np.concatenate(masks, axis=0)
    PROMPT_TYPE_FOR_VIDEO = "mask"
    OBJECTS = [str(_) for _ in range(annotations)]

###
# Step 3: Register each object's positive points to video predictor with seperate add_new_points call
###

assert PROMPT_TYPE_FOR_VIDEO in [
    "point",
    "box",
    "mask",
], "SAM 2 video predictor only support point/box/mask prompt"

# If you are using point prompts, we uniformly sample positive points based on the mask
if PROMPT_TYPE_FOR_VIDEO == "point":
    # sample the positive points from mask for each objects
    all_sample_points = sample_points_from_masks(masks=masks, num_points=10)

    for object_id, (label, points) in enumerate(
        zip(OBJECTS, all_sample_points), start=1
    ):
        labels = np.ones((points.shape[0]), dtype=np.int32)
        _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=object_id,
            points=points,
            labels=labels,
        )
# Using box prompt
elif PROMPT_TYPE_FOR_VIDEO == "box":
    for object_id, (label, box) in enumerate(zip(OBJECTS, input_boxes), start=1):
        _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=object_id,
            box=box,
        )
# Using mask prompt is a more straightforward way
elif PROMPT_TYPE_FOR_VIDEO == "mask":
    for object_id, (label, mask) in enumerate(zip(OBJECTS, masks), start=1):
        labels = np.ones((1), dtype=np.int32)
        _, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=object_id,
            mask=mask,
        )
else:
    raise NotImplementedError(
        "SAM 2 video predictor only support point/box/mask prompts"
    )


###
# Step 4: Propagate the video predictor to get the segmentation results for each frame
###
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(
    inference_state
):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

###
# Step 5: Visualize the segment results across the video and save them
###

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

# load the first image to get the dimensions of the video
output_video_path = f"{args.save_dir}/{os.path.basename(os.path.normpath(args.video_dir))}_{args.size}.mp4"

first_image_path = os.path.join(video_dir, frame_names[0])
first_image = cv2.imread(first_image_path)
height, width, _ = first_image.shape

# create a video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # codec for saving the video
video_writer = cv2.VideoWriter(output_video_path, fourcc, args.fps, (width, height))

all_results = {}

ID_TO_OBJECTS = dict(enumerate(OBJECTS, start=1))
for frame_idx, segments in tqdm(video_segments.items(), desc="Annotating frames"):
    img_path = os.path.join(video_dir, frame_names[frame_idx])
    img = cv2.imread(img_path)

    object_ids = list(segments.keys())
    masks = list(segments.values())
    masks = np.concatenate(masks, axis=0)

    input_boxes = sv.mask_to_xyxy(masks)

    if args.save_masks:
        frame_dict = {}

        for idx, obj_id in enumerate(object_ids):
            mask = masks[idx]
            mask_rle = single_mask_to_rle(mask)
            bbox = sv.mask_to_xyxy(mask[None, ...])[0].tolist()

            class_name = ID_TO_OBJECTS[obj_id]
            score = dino_scores[obj_id - 1]

            frame_dict[obj_id] = {
                "class_name": class_name,
                "score": float(score),
                "segmentation": mask_rle,
                "bbox": bbox,
            }

        all_results[frame_idx] = frame_dict

    detections = sv.Detections(
        xyxy=input_boxes,  # (n, 4)
        mask=masks,  # (n, h, w)
        class_id=np.array(object_ids, dtype=np.int32),
    )
    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
    label_annotator = sv.LabelAnnotator()
    annotated_frame = label_annotator.annotate(
        annotated_frame,
        detections=detections,
        labels=[ID_TO_OBJECTS[i] for i in object_ids],
    )
    mask_annotator = sv.MaskAnnotator()
    annotated_frame = mask_annotator.annotate(
        scene=annotated_frame, detections=detections
    )
    video_writer.write(annotated_frame)
    if args.save_images:
        cv2.imwrite(
            os.path.join(args.save_dir, f"annotated_frame_{frame_idx:05d}.jpg"),
            annotated_frame,
        )

output_json_path = os.path.join(
    args.save_dir,
    f"{os.path.basename(os.path.normpath(args.video_dir))}_{args.size}.json",
)
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=4)

print(f"[INFO] Saved masks JSON at: {output_json_path}")


# ---- Build MultiIndex DataFrame: (frame, object_id) ----
records = []
mi_tuples = []

# Sort for deterministic ordering
for frame_idx in sorted(all_results.keys()):
    obj_map = all_results[frame_idx]
    for obj_id in sorted(obj_map.keys()):
        det = obj_map[obj_id]
        rle = det["segmentation"]  # {'size': [H, W], 'counts': str}
        size = rle.get("size", None)  # [H, W]
        counts = rle.get("counts", None)  # str (RLE)
        bbox = det.get("bbox", None)  # [x1, y1, x2, y2]
        cls = det.get("class_name", None)
        score = det.get("score", None)

        # Ensure empty string classes become NaN
        if isinstance(cls, str) and cls.strip() == "":
            cls = np.nan

        records.append([size, counts, bbox, cls, score])
        mi_tuples.append((int(frame_idx), int(obj_id)))

df = pd.DataFrame(
    records,
    index=pd.MultiIndex.from_tuples(mi_tuples, names=["frame", "object_id"]),
    columns=["size", "counts", "bbox", "class", "score"],
)

# ---- Save to Parquet ----
parquet_path = os.path.join(
    args.save_dir,
    f"{os.path.basename(os.path.normpath(args.video_dir))}_{args.size}.parquet",
)
df.to_parquet(parquet_path, index=True, engine="pyarrow")
print(f"[INFO] Saved annotations DataFrame (parquet) at: {parquet_path}")


video_writer.release()
print(f"Video saved at {output_video_path}")

if args.create_video:
    create_video_from_images(args.save_dir, output_video_path)
