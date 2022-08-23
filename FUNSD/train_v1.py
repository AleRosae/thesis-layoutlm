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
import ast

logging.set_verbosity_error()

if 'labels.txt' not in os.listdir('data'):
    print('preprocessing data...')
    pre_process.main(data_dir = 'data/training_data/annotations', data_split='train', output_dir='data',
                    model_name_or_path='microsoft/layoutlm-base-uncased', max_len=510)

    pre_process.main(data_dir = 'data/testing_data/annotations', data_split='test', output_dir='data',
                    model_name_or_path='microsoft/layoutlm-base-uncased', max_len=510)

    with open('data/train.txt', 'r') as fp:
        lines = fp.readlines()
        labels = []
        for l in lines:
            if l != "\n":
                labels.append(l.split('\t')[1])

    labels = list(set(labels))
    with open('data/labels.txt', 'w') as fp:
        for l in labels:
            fp.write(l)

print('creating dataset...')

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
        'data_dir': 'data',
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


batch_size = 14
# the LayoutLM authors already defined a specific FunsdDataset, so we are going to use this here
train_dataset = FunsdDataset(args, tokenizer, labels, pad_token_label_id, mode="train")
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset,
                              shuffle=True,
                              batch_size=batch_size)

eval_dataset = FunsdDataset(args, tokenizer, labels, pad_token_label_id, mode="test")
eval_sampler = SequentialSampler(eval_dataset)
eval_dataloader = DataLoader(eval_dataset,
                             shuffle=True,
                            batch_size=batch_size)

print(f'train split: {len(train_dataloader)}')
print(f'validation split: {len(eval_dataloader)}')



def train_layoutLM(model, epochs, dataloader_train, dataloader_eval, optimizer, early_stop_arg, run, test_mode):
  #args for early stop
  last_loss = 1000
  last_f1 = 0
  patience = early_stop_arg['patience']
  trigger_times = 0

  #device = "cuda" if torch.cuda.is_available() else "cpu"
  model.to(device)

  final_results = []
  steps = -1
  num_epochs = epochs
  model.train()
  for epoch in range(1, num_epochs):  # loop over the dataset multiple times
    for batch in tqdm(dataloader_train, desc=f'training {epoch} / {num_epochs}'):
        # get the inputs;
        input_ids = batch[0].to(device)
        bbox = batch[4].to(device)
        attention_mask = batch[1].to(device)
        token_type_ids = batch[2].to(device)
        labels = batch[3].to(device)

        optimizer.zero_grad()
        # forward pass
        outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids,
                        labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        steps += 1
        
    print(f"Loss after {epochs} epochs: {loss.item()}")    
    eval_results = do_eval(model, dataloader_eval) 
    #print(f'Validation results: {eval_results}')
    current_loss = eval_results['eval_loss']
    current_f1 = eval_results['micro avg']['f1-score']
    print(f'Validaiton loss: {current_loss}')
    #implementing early stopping
    if test_mode == 'val_loss':
        if current_loss > last_loss:
            trigger_times += 1
            print(f'Validation loss did not decrease from {last_loss}.')
            print('Trigger Times:', trigger_times)

            if trigger_times >= patience or epoch == num_epochs - 1:
                print(f'Early stopping because validation loss did not decrease after {trigger_times} epochs.')
                print(f'Returning best model named: {best_model}')
                best_model = torch.load(best_model)
                df = pd.DataFrame(final_results)
                df.to_csv(f'results/v1/log_v1_FUNSD_{test_mode}_run{run}.csv', index = False)
                return best_model

        else:
            print(f'Validation loss decresed from {last_loss}. Saving checkpoint...')
            best_model = f'models/checkpointLM1_epoch{epoch}.pt'
            for ckpt in os.listdir('models'):
                if 'checkpointLM1_epoch' in ckpt:
                    os.remove(f'models/{ckpt}') #avoid too many checkpoints
            torch.save(model, best_model)
            trigger_times = 0
            last_loss = current_loss
    elif test_mode == 'f1_score':
        if current_f1 < last_f1:
            trigger_times += 1
            print(f'f1 score did not increase from {last_f1}.')
            print('Trigger Times:', trigger_times)

            if trigger_times >= patience or epoch == num_epochs - 1:
                print(f'Early stopping because f1_score did not increase after {trigger_times} epochs.')
                print(f'Returning best model named: {best_model}')
                best_model = torch.load(best_model)
                df = pd.DataFrame(final_results)
                df.to_csv(f'results/v1/log_v1_FUNSD_{test_mode}_run{run}.csv', index = False)
                return best_model

        else:
            print(f'F1 score incresead from {last_f1}. Saving checkpoint...')
            best_model = f'models/checkpointLM1_epoch{epoch}.pt'
            for ckpt in os.listdir('models'):
                if 'checkpointLM1_epoch' in ckpt:
                    os.remove(f'models/{ckpt}') #avoid too many checkpoints
            torch.save(model, best_model)
            trigger_times = 0
            last_loss = current_loss
        
        
    tmp = eval_results
    tmp['epoch'] =  epoch
    tmp['train_loss'] =  loss.item()
    final_results.append(tmp)
  df = pd.DataFrame(final_results)
  df.to_csv(f'results/v1/log_v1_FUNSD_{test_mode}_run{run}.csv', index = False)
  best_model = torch.load(best_model)
  return best_model

def do_eval(model, dataloader_eval):
  eval_loss = 0.0
  nb_eval_steps = 0
  preds = None
  out_label_ids = None

  # put model in evaluation mode
  model.eval()
  for batch in tqdm(dataloader_eval, desc="Evaluating"):
      with torch.no_grad():
          input_ids = batch[0].to(device)
          bbox = batch[4].to(device)
          attention_mask = batch[1].to(device)
          token_type_ids = batch[2].to(device)
          labels = batch[3].to(device)

          # forward pass
          outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids,
                          labels=labels)
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

  for i in range(out_label_ids.shape[0]):
      for j in range(out_label_ids.shape[1]):
          if out_label_ids[i, j] != pad_token_label_id: #escludi pad e other 
              out_label_list[i].append(label_map[out_label_ids[i][j]])
              preds_list[i].append(label_map[preds[i][j]])

  strict = classification_report(out_label_list, preds_list,  output_dict=True, mode='strict', scheme=IOBES)
  print('strict: ', classification_report(out_label_list, preds_list, output_dict=False, mode='strict', scheme=IOBES))

  strict['eval_loss'] = eval_loss.item()
  return strict

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = LayoutLMForTokenClassification.from_pretrained("microsoft/layoutlm-base-uncased", num_labels=num_labels)
    metric = load_metric("seqeval")
    parser = argparse.ArgumentParser(description='''Train LayoutLM1 for label detection task on the 
                            kelister dataset''')

    parser.add_argument('--batch_size', type=int, default=4, help='batch size for mapping function. Default is 4')
    parser.add_argument('--lr', type=float, default = 5e-5, help='Learning rate for training. Default is 4e-5')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs. Default is 100')
    parser.add_argument('--patience', type=int, default=25, help='Patience. Default is 25')
    parser.add_argument('--run', type=str, default="1", help='run id')
    parser.add_argument('--test_mode', type=str, default='val_loss', help='Mode of testing. val_loss for using validaiton loss, f1_score for using f1_score')

    args = parser.parse_args()
    early_stop_arg = {'patience': args.patience}
    model = train_layoutLM(base_model, args.epochs, train_dataloader, eval_dataloader, 
              AdamW(base_model.parameters(), lr=args.lr), early_stop_arg, args.run, args.test_mode)
    torch.save(model, f'models/LayoutLMv1_FUNSD.pt')
