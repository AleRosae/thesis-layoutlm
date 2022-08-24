import os
import json

def get_line_bbox(bboxs):
    x = [bboxs[i][j] for i in range(len(bboxs)) for j in range(0, len(bboxs[i]), 2)]
    y = [bboxs[i][j] for i in range(len(bboxs)) for j in range(1, len(bboxs[i]), 2)]

    x0, y0, x1, y1 = min(x), min(y), max(x), max(y)

    assert x1 >= x0 and y1 >= y0
    bbox = [x0, y0, x1, y1]
    return bbox

os.makedirs('data_seglevel/test/annotations', exist_ok=True)
os.makedirs('data_seglevel/testing_data/annotations', exist_ok=True)
os.makedirs('data_seglevel/training_data/annotations', exist_ok=True)

for dir in os.listdir('data'):
    for ann in os.listdir(f'data/{dir}/annotations'):
        path = os.path.join(f'data/{dir}/annotations', ann)
        with open(path, 'r') as fp:
            data = json.load(fp)
            new_data = data.copy()
        
        for index, el in enumerate(data['form']):
            if el['label'] != 'other':
                bboxes = [w['box'] for w in el['words']]
                new_bboxes = get_line_bbox(bboxes)
                for w in new_data['form'][index]['words']:
                    w['box'] = new_bboxes
        new_path = os.path.join(f'data_seglevel/{dir}/annotations', ann)
        with open(new_path, 'w') as nfp:
            json.dump(new_data, nfp)
            

