import pandas as pd
import ast
import numpy as np
import copy

def mean_scores(csvs):
    results = {'EFFECTIVE_DATE': [], 'JURISDICTION': [], 'PARTY': [], 'TERM': [],
                'micro avg': [],'macro avg': [], 'weighted avg': []}
    new_results = copy.deepcopy(results)

    dfs = [pd.read_csv(el) for el in csvs]
    for df in dfs:
        min_val = df['eval_loss'].min()
        df_best = df[df['eval_loss'] == min_val]
        for k in results.keys():
            d = df_best[k].tolist()
            df_tmp = ast.literal_eval(d[0])
            results[k].append(df_tmp)
                
    for k, v in results.items():
        prec = []
        recall = []
        f1 = []
        for el in results[k]:
            prec.append(el['precision'])
            recall.append(el['recall'])
            f1.append(el['f1-score'])
        print(f'Variance on {k} (f1): {np.var(f1)}')
        tmp = {'precision': np.mean(prec), 'recall':np.mean(recall), 'f1': np.mean(f1)}
        new_results[k] = tmp

    for k in new_results.keys():
        print(f'{str(k)} = Precision: {new_results[k]["precision"]} | Recall: {new_results[k]["recall"]} | F1: {new_results[k]["f1"]}')
    return new_results

print('Means for LayoytLMv1:')
mean_scores(['v1/layoutLMv1_kleister-nda-TEST_run1.csv', 'v1/layoutLMv1_kleister-nda-TEST_run2.csv', 
            'v1/layoutLMv1_kleister-nda-TEST_run3.csv'])

print('------------------------')
print('Means for LayoytLMv2:')
mean_scores(['v2/LayoutLMv2_kleister-nda-TEST_run1.csv', 'v2/LayoutLMv2_kleister-nda-TEST_run2.csv', 
            'v2/LayoutLMv2_kleister-nda-TEST_run3.csv'])

print('------------------------')
print('Means for LayoytLMv3:')
mean_scores(['v3/LayoutLMv3_kleister-nda-TEST_run1.csv', 'v3/LayoutLMv3_kleister-nda-TEST_run2.csv', 
            'v3/LayoutLMv3_kleister-nda-TEST_run3.csv'])
print('------------------------')
print('Means for LayoytLMv3seglevel:')
mean_scores(['v3/LayoutLMv3_altds_kleister-nda-TEST_run1.csv', 'v3/LayoutLMv3_altds_kleister-nda-TEST_run2.csv'])