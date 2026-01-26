import torch
from PIL import Image

from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = AutoModelForZeroShotObjectDetection.from_pretrained(
    "IDEA-Research/grounding-dino-tiny"
).to(device)
processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")

im = Image.open("data/test-img/0001.jpg")
inputs = processor(images=im, text="chicken.", return_tensors="pt").to(device)

with torch.no_grad():
    out = model(**inputs)

print("OK, ran GroundingDINO on", device)
