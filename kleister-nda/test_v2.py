import json
import os
import datasets
from datasets import  Features, Sequence, ClassLabel, Value, Array2D, Array3D, load_metric
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from transformers import LayoutLMv2Processor, logging
from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import pandas as pd
import argparse
from seqeval.metrics import classification_report
from seqeval.scheme import IOBES, BILOU
from sklearn.metrics import precision_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

logging.set_verbosity_error()
processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")

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
    if ".json" in f:
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
    tmp['document'] = annotation['form'][0]['image'].replace('.jpg', "").split("_")[0]

    for ann in annotation['form']:
      image = f"{path}/images/{annotation['form'][0]['image']}"
      tmp['image_path'] = image
      img = Image.open(image)
      width, height = img.size
  
      words, label = ann["words"], ann["label"]
      words = [w for w in words if w["text"].strip() != ""]
  
      if label == "other":
        for w in words:
            tmp['words'].append(w['text'])
            tmp['bboxes'].append(normalize_box(w['box'], width, height))
            tmp['ner_tags'].append(label2id['O'])
            tmp['idxs'].append(w['idx'])
      else:
          if len(words) == 1:
              tmp['words'].append(words[0]["text"])
              tmp['bboxes'].append(normalize_box(words[0]['box'], width, height))
              tmp['ner_tags'].append(label2id["S-"+label.upper()])
              tmp['idxs'].append(words[0]['idx'])

          else:
              tmp['words'].append(words[0]["text"])
              tmp['bboxes'].append(normalize_box(words[0]['box'], width, height))
              tmp['ner_tags'].append(label2id["B-"+label.upper()])
              tmp['idxs'].append(words[0]['idx'])
              for w in words[1:-1]:
                  tmp['words'].append(w["text"])
                  tmp['bboxes'].append(normalize_box(w['box'], width, height))
                  tmp['ner_tags'].append(label2id["I-"+label.upper()])
                  tmp['idxs'].append(w['idx'])

              tmp['words'].append(words[-1]["text"])
              tmp['bboxes'].append(normalize_box(words[-1]['box'], width, height))
              tmp['ner_tags'].append(label2id["E-"+label.upper()])
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

print('preparing dataset...')

t_pred = create_DS('data_pred/testing_data')

print(t_pred.keys())

print(f'testing split: {len(t_pred)}')

documents = t_pred['document'].tolist()
features= Features({'bboxes': Sequence(feature=Sequence(feature=Value(dtype='int64', id=None), length=-1, id=None), length=-1, id=None),
 'id': Value(dtype='string', id=None),
 'image_path': Value(dtype='string', id=None),
 'ner_tags': Sequence(feature=ClassLabel(num_classes=len(label2id), names=list(label2id.keys()),
       id=None), length=-1, id=None),
 'words': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None)})

pred_ds = datasets.Dataset.from_pandas(t_pred, features = features)

labels = pred_ds.features['ner_tags'].feature.names
id2label = {v: k for v, k in enumerate(labels)}
label2id = {k: v for v, k in enumerate(labels)}
label_other = label2id['O']
print(label2id)

# we need to define custom features
features = Features({
    'image': Array3D(dtype="int64", shape=(3, 224, 224)),
    'input_ids': Sequence(feature=Value(dtype='int64')),
    'attention_mask': Sequence(Value(dtype='int64')),
    'token_type_ids': Sequence(Value(dtype='int64')),
    'bbox': Array2D(dtype="int64", shape=(512, 4)),
    'labels': Sequence(feature=Value(dtype='int64')),
})

def preprocess_data(examples):
  images = [Image.open(path).convert("RGB") for path in examples['image_path']]
  words = examples['words']
  boxes = examples['bboxes']
  word_labels = examples['ner_tags']
  

  encoded_inputs = processor(images, words, boxes=boxes, word_labels=word_labels,
                             padding="max_length", truncation=True)
  return encoded_inputs


pred_dataset = pred_ds.map(preprocess_data, batched=True, remove_columns=pred_ds.column_names,
                                      features=features, batch_size = 8)

pred_dataset.set_format(type='torch', device='cuda')

pred_dataloader = DataLoader(pred_dataset, batch_size=8)

def do_eval_pred(model, dataloader_pred, device):
  # put model in evaluation mode
  model.eval()
  df = {'document': [], 'labels': []}
  final_predictions = []
  final_true = []
  for index, batch in enumerate(tqdm(dataloader_pred, desc="Evaluating on testing set")):
      with torch.no_grad():
        input_ids = batch['input_ids'].to(device)
        bbox = batch['bbox'].to(device)
        image = batch['image'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)
        #words = [processor.tokenizer.decode([el]) for el in batch['input_ids'].squeeze().tolist()]

        # forward pass
        outputs = model(input_ids=input_ids, bbox=bbox, image=image, attention_mask=attention_mask, 
                        token_type_ids=token_type_ids, labels=labels)


        eval_loss = outputs.loss
        # predictions
        predictions = outputs.logits.argmax(dim=2)
        # Remove ignored index (special tokens)
        true_predictions = [
            final_predictions.append([id2label[p.item()] for (p, l) in zip(prediction, label) if l != -100])
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            final_true.append([id2label[l.item()] for (p, l) in zip(prediction, label) if l != -100])
            for prediction, label in zip(predictions, labels)
        ]
        prec_pred = [[id2label[p.item()] for (p, l) in zip(prediction, label)]
                        for prediction, label in zip(predictions, labels)]

        inferences = []

        #for prediction, word in zip(prec_pred[0], words):
         #   if prediction != 'O' and word != '[PAD]':
          #      inferences.append((word, prediction))
        #if len(inferences) > 0:     
         #   df['document'].append(documents[index])   
          #  df['labels'].append(inferences)
          
  final_score = classification_report(final_predictions, final_true, output_dict=True, mode='strict', scheme=IOBES)
  print(classification_report(final_predictions, final_true, output_dict=False, mode='strict', scheme=IOBES))
  #print(classification_report(final_predictions, final_true, output_dict=False))
  #df = pd.DataFrame(df)
  #df.to_csv('out_v2.csv', index = False)

  final_score['eval_loss'] = eval_loss.item()
  print(f"Validation loss: {final_score['eval_loss']}")
  return final_score



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''test LayoutLM2 for label detection task on the 
                            patra dataset''')
    parser.add_argument('--model', default='models/LayoutLMv2_kleister-nda.pt', 
                    help='path to the model for testing. Defailt is models/LayoutLMv2_kleister-nda.pt')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size for mapping function. Default is 2')
    parser.add_argument('--run', type=str, default=1, help='run id')
    args = parser.parse_args()

    device = torch.device('cuda')

    model = torch.load(args.model, map_location=device)
    print('Evaluating on the test split...')
    test = do_eval_pred(model, pred_dataloader, device)
    df = pd.DataFrame([test])
    df.to_csv('results/LayoutLMv2_kleister-nda-TEST_run{run}.csv', index=False)