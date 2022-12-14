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
dataset = load_dataset("nielsr/funsd-layoutlmv3")

features = dataset["train"].features
column_names = dataset["train"].column_names
image_column_name = "image"
text_column_name = "tokens"
boxes_column_name = "bboxes"
label_column_name = "ner_tags"

def get_label_list(labels):
    unique_labels = set()
    for label in labels:
        unique_labels = unique_labels | set(label)
    label_list = list(unique_labels)
    label_list.sort()
    return label_list

if isinstance(features[label_column_name].feature, ClassLabel):
    label_list = features[label_column_name].feature.names
    # No need to convert the labels since they are already ints.
    id2label = {k: v for k,v in enumerate(label_list)}
    label2id = {v: k for k,v in enumerate(label_list)}
else:
    label_list = get_label_list(dataset["train"][label_column_name])
    id2label = {k: v for k,v in enumerate(label_list)}
    label2id = {v: k for k,v in enumerate(label_list)}
num_labels = len(label_list)


device = torch.device('cpu')
model = LayoutLMv3ForTokenClassification.from_pretrained('Sennodipoi/LayoutLMv3-FUNSD-ft')
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

def get_line_bbox(bboxs):
    x = [bboxs[i][j] for i in range(len(bboxs)) for j in range(0, len(bboxs[i]), 2)]
    y = [bboxs[i][j] for i in range(len(bboxs)) for j in range(1, len(bboxs[i]), 2)]

    x0, y0, x1, y1 = min(x), min(y), max(x), max(y)

    assert x1 >= x0 and y1 >= y0
    bbox = [x0, y0, x1, y1]
    return bbox

def inference(img, annotation):
    image = Image.open(img)
    image = image.convert("RGB")
    with open(annotation) as f:
        data = json.load(f)
    words = []
    actual_boxes= []

    for el in data['form']:
        tmp_box = []
        for w in el['words']:
            words.append(w['text'])
            tmp_box.append(w['box'])
        for b in tmp_box:
            actual_boxes.append(get_line_bbox(tmp_box))
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
    print(len(predictions), len(token_boxes))
    true_predictions = [id2label[prediction] for prediction in predictions]
    true_boxes = [unnormalize_box(box, width, height) for box in token_boxes]
    draw = ImageDraw.Draw(image)
    print(len(true_predictions), len(true_boxes))
    print(true_predictions)
    font = ImageFont.load_default()

    label2color = {'question':'blue', 'answer':'green', 'header':'orange', 'o':'violet'}

    
    for prediction, box in zip(true_predictions, true_boxes):
        predicted_label = iob_to_label(prediction[2:])
        draw.rectangle(box, outline=label2color[predicted_label.lower()])
        draw.text((box[0] + 10, box[1] - 10), text=predicted_label, fill=label2color[predicted_label.lower()], font=font)
    image.save(f'predictionV3_{img.split("/")[-1]}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''inference with LayoutLMv3 finetuned on FUNSD''')
   
    parser.add_argument('--image', required=True, help='')
    parser.add_argument('--annotation', required=True,  help='')
    args = parser.parse_args()
    start = time.time()
    inference(args.image, args.annotation)
    end = time.time()
    print(f'time: {end - start}')