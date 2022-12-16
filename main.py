import torch
import open_clip
from PIL import Image
import json
from open_clip import tokenizer
import numpy as np

model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')

images = []
images.append(Image.open("a anime girl.png"))
images.append(Image.open("a manga girl.png"))
images = [preprocess(x) for x in images]

image_input = torch.tensor(np.stack(images))

with open("tags.json","r") as f:
    tags=json.load(f)

text_descriptions = [f"This is {label}." for label in tags["cool type"]]
text_tokens = tokenizer.tokenize(text_descriptions)

with torch.no_grad():
    image_features = model.encode_image(image_input).float()
    text_features = model.encode_text(text_tokens).float()
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

text_probs = (100.0*image_features @ text_features.T).softmax(dim=-1)
top_probs, top_labels = text_probs.cpu().topk(1, dim=-1)

print([f'image {i}: {tags["cool type"][top_labels[i][0]]},{top_probs[i][0]}' for i in range(len(images))])

