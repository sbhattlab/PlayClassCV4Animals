import json
import os
from argparse import ArgumentParser

import cv2
import numpy as np
import pycocotools.mask as mask_util
import supervision as sv
import torch
from PIL import Image
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from tqdm import tqdm
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from utils.track_utils import sample_points_from_masks

# from utils.video_utils import create_video_from_images


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "-i",
        "--video_dir",
        type=str,
        required=True,
        help="a directory of JPEG frames with filenames like `<frame_index>.jpg",
    )
    parser.add_argument(
        "--text",
        type=str,
        help="text queries. Need to be lowercased + end with a dot",
        required=True,
    )
    parser.add_argument(
        "-m", "--mask_file", type=str, help="Path to .json file containing masks"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=25,
        help="frame rate of the output video",
    )
    parser.add_argument(
        "--size",
        type=str,
        choices=("large", "base_plus", "small", "tiny"),
        default="large",
        help="SAM2 model size to be used",
    )
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=0.25,
        help="box threshold for grounding DINO",
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.3,
        help="text threshold for grounding DINO",
    )
    parser.add_argument(
        "-o",
        "--save-dir",
        type=str,
        default="./tracking_results",
        help="directory to save the tracking results",
    )
    parser.add_argument(
        "--save-masks",
        action="store_true",
        help="save the masks for each object",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="remove the images after creating the video",
    )

    return parser.parse_args()


def single_mask_to_rle(mask):
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


args = parse_args()

###
# Step 1: Environment settings and model initialization
###
# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# init sam image predictor and video predictor model
sam2_checkpoint = f"./checkpoints/sam2.1_hiera_{args.size}.pt"
model_cfg_size = args.size[0].split("_")[0].lower()
if "plus" in args.size:
    model_cfg_size += "+"
model_cfg = f"configs/sam2.1/sam2.1_hiera_{model_cfg_size}.yaml"

video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
image_predictor = SAM2ImagePredictor(sam2_image_model)


# init grounding dino model from huggingface
model_id = "IDEA-Research/grounding-dino-tiny"
device = "cuda" if torch.cuda.is_available() else "cpu"
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
    img_path = os.path.join(video_dir, frame_names[ann_frame_idx])
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
    input_boxes = results[0]["boxes"].cpu().numpy()
    OBJECTS = [str(_) for _ in range(len(results[0]["labels"]))]

    # prompt SAM 2 image predictor to get the mask for the object
    masks, scores, logits = image_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    # convert the mask shape to (n, H, W)
    if masks.ndim == 3:
        masks = masks[None]
        scores = scores[None]
        logits = logits[None]
    elif masks.ndim == 4:
        masks = masks.squeeze(1)

    PROMPT_TYPE_FOR_VIDEO = "box"  # or "point"
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

ID_TO_OBJECTS = dict(enumerate(OBJECTS, start=1))
for frame_idx, segments in tqdm(video_segments.items(), desc="Annotating frames"):
    img_path = os.path.join(video_dir, frame_names[frame_idx])
    img = cv2.imread(img_path)

    object_ids = list(segments.keys())
    masks = list(segments.values())
    masks = np.concatenate(masks, axis=0)

    input_boxes = sv.mask_to_xyxy(masks)

    if args.save_masks:
        # convert mask into rle format
        mask_rles = [single_mask_to_rle(mask) for mask in masks]

        results = {
            "image_path": img_path,
            "annotations": [
                {"segmentation": mask_rle, "bbox": box}
                for mask_rle, box in zip(mask_rles, input_boxes.tolist())
            ],
        }

        with open(
            os.path.join(args.save_dir, f"{frame_idx:05d}.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(results, f, indent=4)

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
video_writer.release()
print(f"Video saved at {output_video_path}")

# create_video_from_images(args.save_dir, output_video_path)
