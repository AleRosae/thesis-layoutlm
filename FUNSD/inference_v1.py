from PIL import Image, ImageDraw, ImageFont
from torch.nn import CrossEntropyLoss
import json
from transformers import LayoutLMTokenizer
from funsd import FunsdDataset, InputFeatures
from transformers import LayoutLMForTokenClassification, AutoModel,LayoutLMv2ForTokenClassification
import pandas as pd
import os
import torch
import argparse
import time


model = LayoutLMForTokenClassification.from_pretrained("Sennodipoi/LayoutLMv1-FUNSD-ft") 
device = torch.device('cpu')
def get_labels(path):
    with open(path, "r") as f:
        labels = f.read().splitlines()
    if "O" not in labels:
        labels = ["O"] + labels
    return labels
labels = get_labels("data/labels.txt")
num_labels = len(labels)
label_map = {i: label for i, label in enumerate(labels)}

def normalize_box(box, width, height):
    return [
        int(1000 * (box[0] / width)),
        int(1000 * (box[1] / height)),
        int(1000 * (box[2] / width)),
        int(1000 * (box[3] / height)),
    ]

def convert_example_to_features(image, words, boxes, actual_boxes, tokenizer, args, cls_token_box=[0, 0, 0, 0],
                                 sep_token_box=[1000, 1000, 1000, 1000],
                                 pad_token_box=[0, 0, 0, 0]):
      width, height = image.size

      tokens = []
      token_boxes = []
      actual_bboxes = [] # we use an extra b because actual_boxes is already used
      token_actual_boxes = []
      for word, box, actual_bbox in zip(words, boxes, actual_boxes):
          word_tokens = tokenizer.tokenize(word)
          tokens.extend(word_tokens)
          token_boxes.extend([box] * len(word_tokens))
          actual_bboxes.extend([actual_bbox] * len(word_tokens))
          token_actual_boxes.extend([actual_bbox] * len(word_tokens))

      # Truncation: account for [CLS] and [SEP] with "- 2". 
      special_tokens_count = 2 
      if len(tokens) > args.max_seq_length - special_tokens_count:
          tokens = tokens[: (args.max_seq_length - special_tokens_count)]
          token_boxes = token_boxes[: (args.max_seq_length - special_tokens_count)]
          actual_bboxes = actual_bboxes[: (args.max_seq_length - special_tokens_count)]
          token_actual_boxes = token_actual_boxes[: (args.max_seq_length - special_tokens_count)]

      # add [SEP] token, with corresponding token boxes and actual boxes
      tokens += [tokenizer.sep_token]
      token_boxes += [sep_token_box]
      actual_bboxes += [[0, 0, width, height]]
      token_actual_boxes += [[0, 0, width, height]]
      
      segment_ids = [0] * len(tokens)

      # next: [CLS] token
      tokens = [tokenizer.cls_token] + tokens
      token_boxes = [cls_token_box] + token_boxes
      actual_bboxes = [[0, 0, width, height]] + actual_bboxes
      token_actual_boxes = [[0, 0, width, height]] + token_actual_boxes
      segment_ids = [1] + segment_ids

      input_ids = tokenizer.convert_tokens_to_ids(tokens)

      # The mask has 1 for real tokens and 0 for padding tokens. Only real
      # tokens are attended to.
      input_mask = [1] * len(input_ids)

      # Zero-pad up to the sequence length.
      padding_length = args.max_seq_length - len(input_ids)
      input_ids += [tokenizer.pad_token_id] * padding_length
      input_mask += [0] * padding_length
      segment_ids += [tokenizer.pad_token_id] * padding_length
      token_boxes += [pad_token_box] * padding_length
      token_actual_boxes += [pad_token_box] * padding_length

      assert len(input_ids) == args.max_seq_length
      assert len(input_mask) == args.max_seq_length
      assert len(segment_ids) == args.max_seq_length
      #assert len(label_ids) == args.max_seq_length
      assert len(token_boxes) == args.max_seq_length
      assert len(token_actual_boxes) == args.max_seq_length
      
      return input_ids, input_mask, segment_ids, token_boxes, token_actual_boxes

def iob_to_label(label):
  if label != 'O':
    return label[2:]
  else:
    return "O"
    
def inference(img, annotation):
  
  tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
  
  args = {'local_rank': -1,
        'overwrite_cache': True,
        'data_dir': '/content/data',
        'model_name_or_path':'microsoft/layoutlm-base-uncased',
        'max_seq_length': 512,
        'model_type': 'layoutlm',}
    
  class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

  args = AttrDict(args)

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

  input_ids, input_mask, segment_ids, token_boxes, token_actual_boxes = convert_example_to_features(image=image, 
    words=words, boxes=boxes, actual_boxes=actual_boxes, tokenizer=tokenizer, args=args)
  
  input_ids = torch.tensor(input_ids, device=device).unsqueeze(0)
  attention_mask = torch.tensor(input_mask, device=device).unsqueeze(0)
  token_type_ids = torch.tensor(segment_ids, device=device).unsqueeze(0)
  bbox = torch.tensor(token_boxes, device=device).unsqueeze(0)

  outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids)
  token_predictions = outputs.logits.argmax(-1).squeeze().tolist() # the predictions are at the token level

  word_level_predictions = [] # let's turn them into word level predictions
  final_boxes = []
  new_words = [] #prendi anche lista delle parole

  for id, token_pred, box in zip(input_ids.squeeze().tolist(), token_predictions, token_actual_boxes):
    if (tokenizer.decode([id]).startswith("##")) or (id in [tokenizer.cls_token_id, 
                                                            tokenizer.sep_token_id, 
                                                            tokenizer.pad_token_id]):
      # skip prediction + bounding box

      continue
    else:
      word_level_predictions.append(token_pred)
      final_boxes.append(box)
      new_words.append(id)
  
  draw = ImageDraw.Draw(image)

  font = ImageFont.load_default()

  label2color = {'question':'blue', 'answer':'green', 'header':'orange', 'o':'violet'}

  for prediction, box in zip(word_level_predictions, final_boxes):
      predicted_label = iob_to_label(label_map[prediction]).lower()
      draw.rectangle(box, outline=label2color[predicted_label])
      draw.text((box[0] + 10, box[1] - 10), text=predicted_label, fill=label2color[predicted_label], font=font)
  image.save(f'predictionV1_{img.split("/")[-1]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''inference with LayoutLMv1 finetuned on FUNSD''')
   
    parser.add_argument('--image', required=True, help='')
    parser.add_argument('--annotation', required=True,  help='')
    args = parser.parse_args()
    start = time.time()
    inference(args.image, args.annotation)
    end = time.time()
    print(f'time: {end - start}')