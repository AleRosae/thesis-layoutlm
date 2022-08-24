import csv
import json
import os
import shutil
import gzip
import base64

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
if  not os.path.exists('data/test/'):
    os.makedirs('data/test/ocr')
    os.makedirs('data/test/images')

files = []
with open('test-A\in.tsv', 'r', encoding='utf-8') as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t', quoting=csv.QUOTE_NONE)
    for item in reader:
        files.append(item[0])

for file in files:
    name_f = file.split('.')[0]
    with open(f'kleister-needle/{name_f}/{name_f}.json', 'r', encoding='utf-8-sig') as fp:
        n_data = json.load(fp)
    for index, page in enumerate(n_data['words']):
        form = {} #FUNSD format
        form['form'] = []
        text = []
        indexes = []
        boxes = []
        decoded = decode_words_list(page)
        for el in decoded:
            text.append(el['text'])
            boxes.append(el['bbox'])

        end_slice = 512 if len(decoded) > 512 else len(decoded) #controllo se text > 512 e nel caso splitto in vari json
        add_id = 0 if len(decoded) > 512 else ""
        tmp_text = text.copy()
        tmp_boxes = boxes.copy()
        for slide in range(0, len(decoded), 500):
            text = tmp_text[slide:end_slice]
            boxes = tmp_boxes[slide:end_slice]
            end_slice = len(decoded)
            if type(add_id) == int:
                add_id += 1

            lm = {'image': f'{name_f}_{str(index).zfill(5)}', 
            'words': text, 'boxes': boxes}
            form['form'].append(lm)
            n_annotation = f'data/test/ocr/{name_f}_{str(index).zfill(5)}{add_id}.json' if type(add_id) == str else f'data/test/ocr/{name_f}_{str(index).zfill(5)}_{add_id}.json'

            with open(n_annotation, 'w') as fp:
                    json.dump(form, fp)

    for el in os.listdir(f'kleister-needle/{name_f}/images'):
        shutil.copy(f'kleister-needle/{name_f}/images/{el}', f'data/test/images')