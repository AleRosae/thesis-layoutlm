import json
import os
import datasets
from datasets import  DatasetDict, Dataset, Features, Sequence, ClassLabel, Value, Array2D, Array3D, load_metric
import pandas as pd
from PIL import Image
from transformers import LayoutLMv2Processor, logging, LayoutLMv2ForTokenClassification, AdamW, LayoutLMForTokenClassification
from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import argparse
from seqeval.metrics import classification_report
from seqeval.scheme import IOBES


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
train = create_DS('data/training_data')
test = create_DS('data/testing_data')

print(f'trainig split: {len(train)}')
print(f'validation split: {len(test)}')


features= Features({'bboxes': Sequence(feature=Sequence(feature=Value(dtype='int64', id=None), length=-1, id=None), length=-1, id=None),
 'id': Value(dtype='string', id=None),
 'image_path': Value(dtype='string', id=None),
 'ner_tags': Sequence(feature=ClassLabel(num_classes=len(label2id), names=list(label2id.keys()),
       id=None), length=-1, id=None),
 'words': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None)})

train_ds = datasets.Dataset.from_pandas(train, features = features)
test_ds = datasets.Dataset.from_pandas(test, features = features)

labels = train_ds.features['ner_tags'].feature.names
id2label = {v: k for v, k in enumerate(labels)}
label2id = {k: v for v, k in enumerate(labels)}
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

train_dataset = train_ds.map(preprocess_data, batched=True, remove_columns=train_ds.column_names,
                                      features=features, batch_size = 8)
test_dataset = test_ds.map(preprocess_data, batched=True, remove_columns=test_ds.column_names,
                                      features=features, batch_size = 8)

train_dataset.set_format(type="torch", device='cuda')
test_dataset.set_format(type="torch", device='cuda')

batch_size = 8

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
eval_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


def do_eval(model, dataloader_eval):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # put model in evaluation mode
    model.eval()
    final_predictions = []
    final_true = []
    for batch in tqdm(dataloader_eval, desc="Evaluating"):
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            bbox = batch['bbox'].to(device)
            image = batch['image'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)

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

    
    final_score = classification_report(final_true, final_predictions, output_dict=True, mode='strict', scheme=IOBES)
    print(classification_report(final_true, final_predictions, output_dict=False, mode='strict', scheme=IOBES))
    final_score['eval_loss'] = eval_loss.item()
    print(f"Validation loss: {final_score['eval_loss']}")
    return final_score

def train_layoutLM(model, epochs, dataloader_train, dataloader_eval, optimizer, early_stop_arg, run):
  #args for early stop
  last_loss = 1000
  patience = early_stop_arg['patience']
  trigger_times = 0
  accumulation_steps = 8

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)

  global_step = 1
  num_train_epochs = epochs
  final_results = []
  #put the model in training mode
  model.train() 
  for epoch in range(num_train_epochs):  
    print("Epoch:", epoch)
    for index, batch in enumerate(tqdm(dataloader_train,  desc=f'training {epoch} / {num_train_epochs}')):
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(**batch) 
        loss = outputs.loss 

        loss.backward()

        optimizer.step()

        global_step += 1


    print(f"Loss after {epoch} epochs: {loss.item()}")    
    eval_results = do_eval(model, dataloader_eval) 
    current_loss = eval_results['eval_loss']
    #implementing early stopping
    if current_loss > last_loss:
      trigger_times += 1
      print(f'Validation loss did not decrease from {last_loss}.')
      print('Trigger Times:', trigger_times)
      df = pd.DataFrame(final_results)
      df.to_csv(f'results/v2/log_layoutLMv2_kleister-nda_run{run}.csv', index = False)

      if trigger_times >= patience:
          print(f'Early stopping because validation loss did not decrease after {trigger_times} epochs.')
          print(f'Returning best model named: {best_model}')
          best_model = torch.load(best_model)
          df = pd.DataFrame(final_results)
          df.to_csv(f'results/v2/log_layoutLMv2_kleister-nda_run{run}.csv', index = False)
          return best_model

    else:
      print(f'Validation loss decresed. Saving checkpoint...')
      best_model = f'models/checkpointLMv2_epoch{epoch}.pt'
      for ckpt in os.listdir('models'):
          if 'checkpointLMv2_epoch' in ckpt:
              os.remove(f'models/{ckpt}') #avoid too many checkpoints
      torch.save(model, best_model)
      trigger_times = 0
      last_loss = current_loss

    tmp = eval_results
    tmp['epoch'] =  epoch
    tmp['train_loss'] =  loss.item()
    final_results.append(tmp)
  df = pd.DataFrame(final_results)
  df.to_csv(f'results/v2/log_layoutLMv2_kleister-nda_run{run}.csv', index = False)
  best_model = torch.load(best_model)
  return best_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''Train LayoutLM2 for label detection task on the 
                            patra dataset''')

    parser.add_argument('--batch_size', type=int, default=2, help='batch size for mapping function. Default is 2')
    parser.add_argument('--lr', type=float, default = 5e-5, help='Learning rate for training. Default is 4e-5')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs. Default is 25')
    parser.add_argument('--patience', type=int, default=12, help='Patience. Default is 5')
    parser.add_argument('--run', type=str, default=1, help='run id')

    args = parser.parse_args()

    base_model = LayoutLMv2ForTokenClassification.from_pretrained('microsoft/layoutlmv2-base-uncased',
                                                          num_labels=len(labels))
    early_stop_arg = {'patience': args.patience}
    print('starting training...')
    model = train_layoutLM(base_model, args.epochs, train_dataloader, eval_dataloader, 
              AdamW(base_model.parameters(), lr=args.lr), early_stop_arg, args.run)
    torch.save(model, 'models/LayoutLMv2_kleister-nda.pt')

                                                        