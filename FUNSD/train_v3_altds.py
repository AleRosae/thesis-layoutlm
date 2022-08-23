import json
import os
import datasets
from datasets import  DatasetDict, Dataset, Features, Sequence, ClassLabel, Value, Array2D, Array3D, load_metric
import pandas as pd
from PIL import Image
from transformers import LayoutLMv3Processor, logging, LayoutLMv3ForTokenClassification, AdamW, LayoutLMForTokenClassification, AutoProcessor
from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D, load_dataset
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import pandas as pd
import argparse
from seqeval.metrics import classification_report
from seqeval.scheme import IOBES


logging.set_verbosity_error()
processor =  AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)


def normalize_box(box, width, height):

    return [
        int(1000 * (box[0] / width)),
        int(1000 * (box[1] / height)),
        int(1000 * (box[2] / width)),
        int(1000 * (box[3] / height)),
    ]

dataset = load_dataset("nielsr/funsd-layoutlmv3")

print('preparing dataset...')
features = dataset["train"].features
column_names = dataset["train"].column_names
image_column_name = "image"
text_column_name = "tokens"
boxes_column_name = "bboxes"
label_column_name = "ner_tags"

# In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
# unique labels.
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
print(label_list, id2label, label2id)

def prepare_examples(examples):
  images = examples[image_column_name]
  words = examples[text_column_name]
  boxes = examples[boxes_column_name]
  word_labels = examples[label_column_name]

  encoding = processor(images, words, boxes=boxes, word_labels=word_labels,
                       truncation=True, padding="max_length")

  return encoding

features = Features({
    'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
    'input_ids': Sequence(feature=Value(dtype='int64')),
    'attention_mask': Sequence(Value(dtype='int64')),
    'bbox': Array2D(dtype="int64", shape=(512, 4)),
    'labels': Sequence(feature=Value(dtype='int64')),
})

train_dataset = dataset["train"].map(
    prepare_examples,
    batched=True,
    remove_columns=column_names,
    features=features,
)
test_dataset = dataset["test"].map(
    prepare_examples,
    batched=True,
    remove_columns=column_names,
    features=features,
)

train_dataset.set_format(type="torch", device='cuda')
test_dataset.set_format(type="torch", device='cuda')

train_dataloader = DataLoader(train_dataset, batch_size=6, shuffle=True)
eval_dataloader = DataLoader(test_dataset, batch_size=6, shuffle=True)


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
            image = batch['pixel_values'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # forward pass
            outputs = model(**batch) 
            
            eval_loss = outputs.loss
            # predictions
            predictions = outputs.logits.argmax(dim=2)
            
            # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p.item()] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l.item()] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        for p, l in zip(true_predictions, true_labels):
            final_predictions.append(p)
            final_true.append(l)

    final_score = classification_report(final_true, final_predictions, output_dict=True)
    print(classification_report(final_true, final_predictions, output_dict=False))
    final_score['eval_loss'] = eval_loss.item()
    print(f"Validation loss: {final_score['eval_loss']}")
    return final_score

def train_layoutLM(model, epochs, dataloader_train, dataloader_eval, optimizer, early_stop_arg, run):
  #args for early stop
  last_loss = 1000
  patience = early_stop_arg['patience']
  trigger_times = 0
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)

  global_step = 0
  num_train_epochs = epochs
  final_results = []
  #put the model in training mode
  model.train() 
  for epoch in range(num_train_epochs):  
    for batch in tqdm(dataloader_train, desc=f'training {epoch}/{num_train_epochs}'):
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

      if trigger_times >= patience:
          print(f'Early stopping because validation loss did not decrease after {trigger_times} epochs.')
          print(f'Returning best model named: {best_model}')
          best_model = torch.load(best_model)
          df = pd.DataFrame(final_results)
          df.to_csv(f'results/v3/log_altv3_FUNSD_run{run}.csv', index = False)
          return best_model

    else:
      print(f'Validation loss decresed. Saving checkpoint...')
      best_model = f'models/checkpointLMv3_epoch{epoch}.pt'
      for ckpt in os.listdir('models'):
          if 'checkpointLMv3_epoch' in ckpt:
              os.remove(f'models/{ckpt}') #avoid too many checkpoints
      torch.save(model, best_model)
      trigger_times = 0
      last_loss = current_loss

    tmp = eval_results
    tmp['epoch'] =  epoch
    tmp['train_loss'] =  loss.item()
    final_results.append(tmp)
    
  df = pd.DataFrame(final_results)
  df.to_csv(f'results/v3/log_v3alt_FUNSD_run{run}.csv', index = False)
  best_model = torch.load(best_model)
  return best_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''Train LayoutLM3 for label detection task on the 
                            patra dataset''')

    parser.add_argument('--batch_size', type=int, default=2, help='batch size for mapping function. Default is 2')
    parser.add_argument('--lr', type=float, default = 1e-5, help='Learning rate for training. Default is 3e-5')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs. Default is 25')
    parser.add_argument('--patience', type=int, default=12, help='Patience. Default is 5')
    parser.add_argument('--run', type=str, default=1, help='run id')
    args = parser.parse_args()

    base_model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base",
                                                          num_labels = num_labels)
    early_stop_arg = {'patience': args.patience}
    print('starting training...')
    model = train_layoutLM(base_model, args.epochs, train_dataloader, eval_dataloader, 
              AdamW(base_model.parameters(), lr=args.lr), early_stop_arg, args.run)
    torch.save(model, 'models/LayoutLMv3alt_FUNSD.pt')

                                                        