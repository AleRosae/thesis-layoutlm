import json
import os
import datasets
from datasets import  DatasetDict, Dataset, Features, Sequence, ClassLabel, Value, Array2D, Array3D
import pandas as pd
from PIL import Image
from transformers import LayoutLMv2Processor, logging, LayoutLMv2Tokenizer
from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D


logging.set_verbosity_error()

with open('data/labels.txt', 'r') as fp:
    lines = fp.readlines()
    label2id =  {k[:-1].upper():v for v, k in enumerate(lines)}

def normalize_box(box, width, height):

    return [
        int(1000 * (box[0] / width)),
        int(1000 * (box[1] / height)),
        int(1000 * (box[2] / width)),
        int(1000 * (box[3] / height)),
    ]

def create_DS(path):
  train = []
  for f in os.listdir(f'{path}/annotations'):
    if '.json' in f:
        with open(f'{path}/annotations/{f}') as fp:
          annotation = json.load(fp)
    else:
        continue

    id = 0
    tmp = {}
    tmp['id'] = str(id)
    tmp['words'] = []
    tmp['bboxes'] = []
    tmp['ner_tags'] = []
    tmp['idxs'] = []
    for ann in annotation['form']:
      image = f"{path}/images/{annotation['form'][0]['image']}"
      tmp['image_path'] = annotation['form'][0]['image']
      img = Image.open(image)
      width, height = img.size
      tmp['size'] = str(width) + " " + str(height)
  
      words, label = ann["words"], ann["label"]
      words = [w for w in words if w["text"].strip() != ""]
  
      if label == "other":
        for w in words:
            tmp['words'].append(w['text'])
            tmp['bboxes'].append(normalize_box(w['box'], width, height))
            tmp['ner_tags'].append('O')
            tmp['idxs'].append(w['idx'])
      else:
          if len(words) == 1:
              tmp['words'].append(words[0]["text"])
              tmp['bboxes'].append(normalize_box(words[0]['box'], width, height))
              tmp['ner_tags'].append("S-"+label.upper())
              tmp['idxs'].append(words[0]['idx'])

          else:
              tmp['words'].append(words[0]["text"])
              tmp['bboxes'].append(normalize_box(words[0]['box'], width, height))
              tmp['ner_tags'].append("B-"+label.upper())
              tmp['idxs'].append(words[0]['idx'])
              for w in words[1:-1]:
                  tmp['words'].append(w["text"])
                  tmp['bboxes'].append(normalize_box(w['box'], width, height))
                  tmp['ner_tags'].append("I-"+label.upper())
                  tmp['idxs'].append(w['idx'])

              tmp['words'].append(words[-1]["text"])
              tmp['bboxes'].append(normalize_box(words[-1]['box'], width, height))
              tmp['ner_tags'].append("E-"+label.upper())
              tmp['idxs'].append(words[-1]['idx'])
    
    new_words = tmp['words'].copy()
    new_boxes = tmp['bboxes'].copy()
    new_tags = tmp['bboxes'].copy()
    for w, b, t, i in sorted(zip(tmp['words'], tmp['bboxes'], tmp['ner_tags'], tmp['idxs'])):
      new_words[i] = w
      new_boxes[i] = b
      new_tags[i] = t
    
    tmp['words'] = new_words
    tmp['bboxes'] = new_boxes
    tmp['ner_tags'] = new_tags
    
    train.append(tmp)
    id += 1
    
  df = pd.DataFrame(data=train)
  return df


train = create_DS('data/training_data')
test = create_DS('data/testing_data')
pred = create_DS('data_pred/testing_data')

def convert_txt(df, path, split):
    with open(f'{path}/{split}.txt.tmp', 'w', encoding="utf-8") as f, open(
        f'{path}/{split}_box.txt.tmp', 'w', encoding="utf-8") as fb, open(
            f'{path}/{split}_image.txt.tmp', 'w', encoding="utf-8") as fi:

        for index, row in df.iterrows():
            for w, l in zip(row['words'], row['ner_tags']):
                f.write(f'{w}\t{l}\n')
            for w, b in zip(row['words'], row['bboxes']):
                b = [str(x) for x in b]
                b = " ".join(b)
                fb.write(f'{w}\t{b}\n')
            for w, b in zip(row['words'], row['bboxes']):
                b = [str(x) for x in b]
                b = " ".join(b)
                fi.write(f'{w}\t{b}\t{row["size"]}\t{row["image_path"]}\n')
            
            f.write('\n')
            fb.write('\n')
            fi.write('\n')

        
if __name__ == "__main__":
    convert_txt(train, 'data', 'train')
    convert_txt(test, 'data', 'test')
    convert_txt(pred, 'data_pred', 'test')