import csv
import pandas as pd
import os
from tqdm import tqdm
from pdf2image import convert_from_path

train, val, test = [], [], []
labels_train, labels_val, labels_test = [], [], []
with open('train\expected.tsv', 'r', encoding='utf-8') as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t', quoting=csv.QUOTE_NONE)
    for item in reader:
        labels_train.append(item)

with open('train\in.tsv', 'r', encoding='utf-8') as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t', quoting=csv.QUOTE_NONE)
    for index, item in enumerate(reader):
        tmp = {'file': item[0], 'keys': item[1], 'labels': labels_train[index]}
        train.append(tmp)

with open('dev-0\expected.tsv', 'r', encoding='utf-8') as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t', quoting=csv.QUOTE_NONE)
    for item in reader:
        labels_val.append(item)

with open('dev-0\in.tsv', 'r', encoding='utf-8') as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t', quoting=csv.QUOTE_NONE)
    for index, item in enumerate(reader):
        tmp = {'file': item[0], 'keys': item[1], 'labels': labels_val[index]}
        val.append(tmp)


df_train = pd.DataFrame(train)
df_val = pd.DataFrame(val)


def create_imgs(doc, split):
    script_dir = os.path.abspath(os.path.dirname(__file__))
    poppler_path = os.path.join(script_dir, "poppler-21.03.0", "Library", "bin")

    pdf_path = f'documents/{doc}'
    directory, filename = os.path.split(pdf_path)
    images_path = os.path.join(f'dataset\{split}\{filename.split(".")[0]}', 'images')
    os.makedirs(images_path)

    imgs = convert_from_path(pdf_path=pdf_path, dpi=200, thread_count=2, poppler_path=poppler_path)
    for page_num, img in enumerate(imgs):
        img.save(os.path.join(images_path, f'{doc}_{str(page_num+1).zfill(4)}.png'))

if __name__ == '__main__':
    for el in tqdm(df_val['file'].tolist()[:5]):
        create_imgs(el, 'val')