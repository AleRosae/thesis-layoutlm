from torch.nn import CrossEntropyLoss
from transformers import LayoutLMTokenizer
import argparse
import os
from funsd import FunsdDataset, InputFeatures
import pre_process
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import LayoutLMForTokenClassification, AdamW, logging
import torch
import numpy as np
from seqeval.metrics import classification_report
from seqeval.scheme import IOBES
from datasets import load_metric
import pandas as pd
from tqdm import tqdm
import json
logging.set_verbosity_error()

if 'labels.txt' not in os.listdir('data_pred'):
    pre_process.main(data_dir = 'data_pred/testing_data/annotations', data_split='test', output_dir='data_pred',
                    model_name_or_path='microsoft/layoutlm-base-uncased', max_len=510)
    with open('data_pred/test.txt', 'r') as fp:
        lines = fp.readlines()
        labels = []
        for l in lines:
            if l != "\n":
                labels.append(l.split('\t')[1])

    #labels = list(set(labels))
    #with open('data_pred/labels.txt', 'w') as fp:
     #   for l in labels:
      #      fp.write(l)


def create_DS(path):
  train = []
  for f in os.listdir(f'{path}/annotations'):

    with open(f'{path}/annotations/{f}') as fp:
      annotation = json.load(fp)

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
      words, label = ann["words"], ann["label"]
      words = [w for w in words if w["text"].strip() != ""]
  
      if label == "other":
        for w in words:
            tmp['words'].append(w['text'])

            tmp['idxs'].append(w['idx'])
      else:
          if len(words) == 1:
              tmp['words'].append(words[0]["text"])

              tmp['idxs'].append(words[0]['idx'])

          else:
              tmp['words'].append(words[0]["text"])

              tmp['idxs'].append(words[0]['idx'])
              for w in words[1:-1]:
                  tmp['words'].append(w["text"])

                  tmp['idxs'].append(w['idx'])

              tmp['words'].append(words[-1]["text"])

              tmp['idxs'].append(words[-1]['idx'])

    new_words = tmp['words'].copy()
    new_boxes = tmp['bboxes'].copy()
    new_tags = tmp['ner_tags'].copy()
    for w, b, t, i in zip(tmp['words'], tmp['bboxes'], tmp['ner_tags'], tmp['idxs']):
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

print(f'testing split: {len(t_pred)}')

documents = t_pred['document'].tolist()

def get_labels(path):
    with open(path, "r") as f:
        labels = f.read().splitlines()
    if "O" not in labels:
        labels = ["O"] + labels
    return labels

labels = get_labels("data/labels.txt")
num_labels = len(labels)
label_map = {i: label for i, label in enumerate(labels)}
print(label_map)
pad_token_label_id = CrossEntropyLoss().ignore_index

args = {'local_rank': -1,
        'overwrite_cache': True,
        'data_dir': 'data_pred',
        'model_name_or_path':'microsoft/layoutlm-base-uncased',
        'max_seq_length': 512,
        'model_type': 'layoutlm',}

# class to turn the keys of a dict into attributes (thanks Stackoverflow)
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

args = AttrDict(args)

tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")

test_dataset = FunsdDataset(args, tokenizer, labels, pad_token_label_id, mode="test")
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset,
                             sampler=test_sampler,
                            batch_size=16)

def do_eval(model, dataloader_eval, device):
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    tot_words = [[] for i in range (len(dataloader_eval))]
    df = {'document': [], 'labels': []}

    # put model in evaluation mode
    model.eval()
    for index, batch in enumerate(tqdm(dataloader_eval, desc="Evaluating")):
        with torch.no_grad():
            input_ids = batch[0].to(device)
            bbox = batch[4].to(device)
            attention_mask = batch[1].to(device)
            token_type_ids = batch[2].to(device)
            labels = batch[3].to(device)

            # forward pass
            outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            labels=labels)

            #words = [tokenizer.deco#de([el]) for el in batch[0].squeeze().tolist()]
            #tot_words[index].append(#words)
            # get the loss and logits
            eval_loss = outputs.loss
            logits = outputs.logits

            nb_eval_steps += 1

            # compute the predictions
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, labels.detach().cpu().numpy(), axis=0
                )

    # compute average evaluation loss
    preds = np.argmax(preds, axis=2)



    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    pred_tot =  [[] for _ in range(out_label_ids.shape[0])]
    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id: #escludi pad 
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])



    results = classification_report(out_label_list, preds_list, output_dict=True, mode='strict', scheme=IOBES)
    print(classification_report(out_label_list, preds_list, output_dict=False, mode='strict', scheme=IOBES))
    #print(classification_report(out_label_list, preds_list, output_dict=False))


    #results = classification_report(out_label_list, preds_list, output_dict=True)
    results['eval_loss'] = eval_loss.item()
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''test LayoutLM1 for label detection task on the 
                            kleister-nda dataset''')
    parser.add_argument('--model', default='models/LayoutLMv1_kleister-nda.pt', 
                    help='path to the model for testing. Defailt is models/LayoutLMv1_kleister-nda.pt')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size for mapping function. Default is 2')
    parser.add_argument('--run', type=str, default="1", help='run id')
    args = parser.parse_args()
    device = torch.device('cuda')
    model = torch.load(args.model, device)    
    test = do_eval(model, test_dataloader, device)
    df = pd.DataFrame([test])
    df.to_csv(f'results/v1/layoutLMv1_kleister-nda-TEST_run{args.run}.csv', index=False)
