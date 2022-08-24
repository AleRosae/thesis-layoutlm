import json
import gzip
import base64
import os
import editdistance
import argparse
from tqdm import tqdm
import shutil
import csv
import pandas as pd
from num2words import num2words
from datetime import datetime
import re

parser = argparse.ArgumentParser(description='''Merge annotations into LayoutLM and Librarian format''')
parser.add_argument("--directory", type=str, default='dataset', 
  help='''directory of dataset''')
args = parser.parse_args()

def decode_words_list(encoded):
    encoded = gzip.decompress(base64.standard_b64decode(encoded))
    res = []
    index = 0
    while index < len(encoded):
        index_old = index
        index = index + encoded[index:].find(b"\x00")
        text = bytes(encoded[index_old:index]).decode("utf-8")
        index += 1  # skip byte 0
        element_index = int.from_bytes(encoded[index : index + 4], "little")
        index += 4
        font_id = int.from_bytes(encoded[index : index + 4], "little")
        index += 4
        bbox = [int.from_bytes(encoded[i : i + 4], "little") for i in range(index, index + 16, 4)]
        index += 16
        res.append(
            {
                "text": text,
                "index": element_index,
                "font_id": font_id,
                "bbox": bbox,
            }
        )
    return res


def fuzzy(s1,s2):
    s1 = s1.replace('.', "").replace(',', "")
    s2 = s2.replace('.', "").replace(',', "")
    res = (editdistance.eval(s1.lower(),s2.lower())/((len(s1)+len(s2))/2)) 

    return res < 0.28

def subfinder(words_list, answer_list):  
    matches = []
    start_indices = []
    end_indices = []
    for idx, i in enumerate(range(len(words_list))):
        if len(words_list[i:i+len(answer_list)])==len(answer_list) and all(fuzzy(words_list[i+j],answer_list[j]) for j in range(len(answer_list))):
            matches.append(answer_list)
            start_indices.append(idx)
            end_indices.append(idx + len(answer_list) - 1)
    if matches:
      return matches[0], start_indices[0], end_indices[0]
    else:
      return None, 0, 0

def convert_term(term, test=None):
    if test == None:
        number, units = term.split('_')
        num = num2words(int(float(number)))
        new_1 = f'{num}_{units}'
        new_2 = f'{num}_({number})_{units}' # _ per normalizzare con lo split delle altre label
        return new_1, new_2
    else:
        units = "months" if "M" in term else 'years'
        number = re.search('\d', term)[0]
        print(number)
        num = num2words(int(float(number)))
        new_1 = f'{num}_{units}'
        new_2 = f'{num}_({number})_{units}' # _ per normalizzare con lo split delle altre label
        return new_1, new_2

def convert_date(date):
    datetime_obj= datetime.strptime(date, '%Y-%m-%d')
    a = datetime_obj.strftime('%B %d, %Y')
    b = datetime_obj.strftime('%Y/%m/%d')
    c = datetime_obj.strftime('%m/%d/%Y')
    d = datetime_obj.strftime('%Y-%m-%d')
    e = datetime_obj.strftime('%m-%d-%Y')
    f = datetime_obj.strftime('%m-%d-%Y')
    return a, b, c, d, e

def create_annotations(split):
    if split == 'train':
        f_path = 'train'
        out_path = 'training_data'
        terms_test = None
    elif split == 'val':
        f_path = 'dev-0'
        out_path = 'testing_data'
        terms_test = None
    elif split == 'test':
        f_path = 'test-A'
        out_path = 'test'
        terms_test = True #per far andare la verisone alt di convert term
    ds = []
    labels_tsv = []
    
    with open(f'{f_path}\expected.tsv', 'r', encoding='utf-8') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t', quoting=csv.QUOTE_NONE)
        for item in reader:
            labels_tsv.append(item)

    with open(f'{f_path}\in.tsv', 'r', encoding='utf-8') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t', quoting=csv.QUOTE_NONE)
        for index, item in enumerate(reader):
            tmp = {'file': item[0], 'keys': item[1], 'labels': labels_tsv[index]}
            ds.append(tmp)


    for item in tqdm(ds, desc=f'Processing {split} split...'):
        file, keys, labels = item['file'], item['keys'], item['labels']
        Lib_annotations = []
        imgs_names = []
        mapping = []
        id = 1
        name_f = file.split('.')[0]

        try:
            with open(f'kleister-needle/{name_f}/{name_f}.json', 'r', encoding='utf-8-sig') as fp:
                n_data = json.load(fp)
        except FileNotFoundError:
            print(f'Not found: {file}')
            continue
        
        for el in labels[0].split(' '):
            label, value = el.split('=')
            if label == 'term':
                values = convert_term(value, terms_test)
                tmp_labels = {'label': label, 'value': values[0]}
                mapping.append(tmp_labels)
                tmp_labels = {'label': label, 'value': values[1]}
                mapping.append(tmp_labels)
            elif label == 'effective_date':
                values = convert_date(value)
                for v in values:
                    tmp_labels = {'label': label, 'value': v}
                    mapping.append(tmp_labels)
            elif label == 'party':
                values = [value]
                v = value.split('_')
                v[-1] = ',' + v[-1]
                values.append("_".join(v))
                values.append(value.replace("&", "and"))
                values.append(value.replace('and', '&'))
                

                for v in values:
                    tmp_labels = {'label': label, 'value': v}
                    mapping.append(tmp_labels)
            
            tmp_labels = {'label': label, 'value': value}
            mapping.append(tmp_labels)

        for index, page in enumerate(n_data['words']):
            text = []
            boxes = []
            decoded = decode_words_list(page)
            for el in decoded:
                text.append(el['text'].replace('\n', " "))
                boxes.append(el['bbox'])
            end_slice = 512 if len(decoded) > 512 else len(decoded) #controllo se text > 512 e nel caso splitto in vari json
            add_id = 0 if len(decoded) > 512 else ""
            tmp_text = text.copy()
            tmp_boxes = boxes.copy()
            for slide in range(0, len(decoded), 450):
                form = {} #FUNSD format
                form['form'] = []
                indexes = []
                text = tmp_text[slide:end_slice]
                boxes = tmp_boxes[slide:end_slice]
                end_slice = len(decoded)
                if type(add_id) == int:
                    add_id += 1

                for item in mapping:
                    label = item['label']

                    value = item['value'].split('_') if label != 'effective_date' else item['value'].split(" ")
                    result = subfinder(text, value)
                    if result[0]:
                        start = result[1]
                        end = result[2]
                        entity = text[start:end+1]

                        bboxes = boxes[start:end+1]
                        label_words = [{'text': w, 'box': bboxes[idx], 'idx': start + idx} for idx, w in enumerate(entity)]

                        tmp_lm = {'image': f'{name_f}_{str(index).zfill(5)}.jpg', 'label':label, 
                                'text':" ".join(value), 'words': label_words}

                        indexes.append({'start':start, 'end':end})
                        if tmp_lm not in form['form']:
                            form['form'].append(tmp_lm)


                        for n, b in enumerate(bboxes): #general file per librarian
                            tmp_Lib = {'label': label, "page": index+1,
                            "bbox": [b], 'beginWord': start+n, 'endWord': start+n}
                            if tmp_Lib not in Lib_annotations:
                                Lib_annotations.append(tmp_Lib) 

                        for idx_el in range(start, end+1):
                            text[idx_el] = 'Niente_Vuoto'
                    else:
                        pass

                labeled_text = ['O' for w in text]
                for idx in indexes:
                    s = idx['start']
                    e = idx['end']
                    new = ['L' for i in range(s, e +1)]
                    labeled_text[s:e+1] = new
                

                if len(indexes) > 0: #così salto tutte le parti di pagina che non hanno label eccetto other
                    true_text = []
                    true_boxes = []
                    true_idxs = []
                    for i, w in enumerate(text):
                        if labeled_text[i] == 'O':
                            true_text.append(w)
                            true_boxes.append(boxes[i])
                            true_idxs.append(i)

                    unlabel_words = [{'text': w, 'box': true_boxes[idx], 'idx': true_idxs[idx]} for idx, w in enumerate(true_text)]
                    tmp_others = {'image': f'{name_f}_{str(index).zfill(5)}.jpg', 'text': " ".join(true_text), 
                    'label': 'other', 'words': unlabel_words}
                    form['form'].append(tmp_others) 
                    imgs_names.append(f'{name_f}_{str(index).zfill(5)}.jpg')      
                    n_annotation = f'data/{out_path}/annotations/{name_f}_{str(index).zfill(5)}{add_id}.json' if type(add_id) == str else f'data/{out_path}/annotations/{name_f}_{str(index).zfill(5)}_{add_id}.json'
                    with open(n_annotation, 'w') as fp:
                            json.dump(form, fp)
                else:
                   pass
    
        for el in os.listdir(f'kleister-needle/{name_f}/images'):
            if el in imgs_names:
                shutil.copy(f'kleister-needle/{name_f}/images/{el}', f'data/{out_path}/images')

        for el in Lib_annotations:
            el['id'] = id
            id += 1
        n_data['annotations'] = Lib_annotations
        with open(f'kleister-needle/{name_f}/{name_f}.json', 'w',  encoding='utf-8-sig') as fp:
            json.dump(n_data, fp, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    if not os.path.exists('data/test'):
        os.makedirs('data/test/images')
        os.makedirs('data/test/annotations')
    create_annotations('test')  
    if  not os.path.exists('data/training_data'):
        os.makedirs('data/training_data/images')
        os.makedirs('data/training_data/annotations')
    create_annotations('train')     
    if not os.path.exists('data/testing_data'):
        os.makedirs('data/testing_data/images')
        os.makedirs('data/testing_data/annotations')
    create_annotations('val')    
       

