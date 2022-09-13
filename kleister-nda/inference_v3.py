from PIL import Image, ImageDraw, ImageFont
import json
import os
import datasets
from datasets import  load_dataset, Features
import pandas as pd
from PIL import Image
from transformers import LayoutLMv3Processor, logging, LayoutLMv3ForTokenClassification
from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D
import torch
import time
import argparse

processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)

def get_labels(path):
    with open(path, "r") as f:
        labels = f.read().splitlines()
    if "O" not in labels:
        labels = ["O"] + labels
    return labels

labels = get_labels("data_seglevel/labels.txt")

id2label = {v: k for v, k in enumerate(labels)}
label2id = {k: v for v, k in enumerate(labels)}


device = torch.device('cpu')
model = torch.load('models/LayoutLMv3_seglevel_kleister-nda.pt', map_location = device)
model.to(device)

def normalize_box(box, width, height):
    return [
        int(1000 * (box[0] / width)),
        int(1000 * (box[1] / height)),
        int(1000 * (box[2] / width)),
        int(1000 * (box[3] / height)),
    ]

def unnormalize_box(bbox, width, height):
     return [
         width * (bbox[0] / 1000),
         height * (bbox[1] / 1000),
         width * (bbox[2] / 1000),
         height * (bbox[3] / 1000),
     ]

def iob_to_label(label):
    if not label:
        return 'o'
    return label


def inference(img, annotation):
    image = Image.open(img)
    image = image.convert("RGB")
    with open(annotation) as f:
        data = json.load(f)
    words = []
    actual_boxes= []

    for el in data['form']:
        for w in el['words']:
            words.append(w['text'])
            actual_boxes.append(w['box'])
    width, height = image.size
    boxes = []

    for box in actual_boxes:
        boxes.append(normalize_box(box, width, height))
    
    encoded_inputs = processor(image, words, boxes=boxes,
                               padding="max_length", truncation=True, return_tensors="pt")
    for k,v in encoded_inputs.items():
        encoded_inputs[k] = v.to(device)

    outputs = model(**encoded_inputs)

    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    token_boxes = encoded_inputs.bbox.squeeze().tolist()
    token_words = encoded_inputs.input_ids.squeeze().tolist()

    width, height = image.size

    true_predictions = [id2label[prediction] for prediction in predictions]
    true_boxes = [unnormalize_box(box, width, height) for box in token_boxes]
    draw = ImageDraw.Draw(image)

    font = ImageFont.load_default()

    label2color = {'term':'blue', 'party':'green', 'effective_date':'orange', 'jurisdiction':'violet', 'o': 'grey'}


    for prediction, box in zip(true_predictions, true_boxes):
        predicted_label = iob_to_label(prediction[2:])
        if predicted_label != 'o':
            draw.rectangle(box, outline=label2color[predicted_label.lower()])
            draw.text((box[0] + 10, box[1] - 10), text=predicted_label, fill=label2color[predicted_label.lower()], font=font)
        image.save(f'predictionV3_{img.split("/")[-1]}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''inference with LayoutLMv3 finetuned on kleister-nda''')
   
    parser.add_argument('--image', required=True, help='')
    parser.add_argument('--annotation', required=True,  help='')
    args = parser.parse_args()
    start = time.time()
    inference(args.image, args.annotation)
    end = time.time()
    print(f'time: {end - start}')