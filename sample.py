import torch
import open_clip
from PIL import Image
import json
from open_clip import tokenizer
import numpy as np
import itertools
from functools import reduce

model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')

cool_tags=[]
images = []
images.append(Image.open("a anime girl.png"))
cool_tags.append([])
images.append(Image.open("a manga girl.png"))
cool_tags.append([])

TH=0.5

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

for i,tag in enumerate(cool_tags):
    if(top_probs[i]>TH):
        tag.append(tags["cool type"][top_labels[i][0]])

text_descriptions = [f"{label} would be in this picture." for label in tags["number of girls"]]
text_tokens = tokenizer.tokenize(text_descriptions)

with torch.no_grad():
    text_features = model.encode_text(text_tokens).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)

text_probs = (100.0*image_features @ text_features.T).softmax(dim=-1)
top_probs, top_labels = text_probs.cpu().topk(1, dim=-1)

print([f'image {i}: {tags["number of girls"][top_labels[i][0]]},{top_probs[i][0]}' for i in range(len(images))])

for i,tag in enumerate(cool_tags):
    if(top_probs[i]>TH):
        tag.append(tags["number of girls"][top_labels[i][0]])

for key, attribute in tags["attributes with no color"].items():
    text_descriptions = [f"This is a girl with {label}." for label in attribute]
    text_tags = [f"{label}" for label in attribute]

    text_tokens = tokenizer.tokenize(text_descriptions)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0*image_features @ text_features.T).softmax(dim=-1)
    top_probs, top_labels = text_probs.cpu().topk(1, dim=-1)

    print([f'image {i}: {text_tags[top_labels[i][0]]},{top_probs[i][0]}' for i in range(len(images))])

    for i,tag in enumerate(cool_tags):
        if (top_probs[i] > TH):
            tag.append(text_tags[top_labels[i][0]])


for key, attribute in tags["attributes with color"].items():
    text_descriptions = [f"This is a girl with {color} {label}." for label in attribute for color in tags["color"]]
    text_tags = [f"{color} {label}" for label in attribute for color in tags["color"]]

    text_tokens = tokenizer.tokenize(text_descriptions)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0*image_features @ text_features.T).softmax(dim=-1)
    top_probs, top_labels = text_probs.cpu().topk(1, dim=-1)

    print([f'image {i}: {text_tags[top_labels[i][0]]},{top_probs[i][0]}' for i in range(len(images))])

    for i,tag in enumerate(cool_tags):
        if (top_probs[i] > TH):
            tag.append(text_tags[top_labels[i][0]])

text_descriptions = [f"This is a girl {label}." for label in tags["actions"]]

text_tokens = tokenizer.tokenize(text_descriptions)

with torch.no_grad():
    text_features = model.encode_text(text_tokens).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)

text_probs = (100.0*image_features @ text_features.T).softmax(dim=-1)
top_probs, top_labels = text_probs.cpu().topk(1, dim=-1)

print([f'image {i}: {text_tags[top_labels[i][0]]},{top_probs[i][0]}' for i in range(len(images))])

for i,tag in enumerate(cool_tags):
    if (top_probs[i] > TH):
        tag.append(tags["actions"][top_labels[i][0]])

text_descriptions = [f"This is a girl in {label}." for label in tags["scenes"]]

text_tokens = tokenizer.tokenize(text_descriptions)

with torch.no_grad():
    text_features = model.encode_text(text_tokens).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)

text_probs = (100.0*image_features @ text_features.T).softmax(dim=-1)
top_probs, top_labels = text_probs.cpu().topk(1, dim=-1)

print([f'image {i}: {text_tags[top_labels[i][0]]},{top_probs[i][0]}' for i in range(len(images))])

for i,tag in enumerate(cool_tags):
    if (top_probs[i] > TH):
        tag.append(tags["scenes"][top_labels[i][0]])

print(cool_tags)

print([reduce(lambda a,b:a+", "+b,ct[1:],ct[0]) for ct in cool_tags])
