import pandas as pd
import json

df = pd.read_csv('./arr_results.csv', index_col=0)
col = df['1.00']

sorted_idx = col.sort_values(ascending=True).index.tolist()
point25 = len(sorted_idx) // 4
point125 = len(sorted_idx) // 8
print(f'total images: {len(sorted_idx)}, 25%: {point25}')
print("Sorted indices for column 1.00:", sorted_idx)


arr_reference = {}
arr_reference['all'] = sorted_idx
arr_reference['best25'] = sorted_idx[:point25]
arr_reference['others'] = sorted_idx[:-point25]
arr_reference['worst25'] = sorted_idx[-point25:]
arr_reference['best125'] = sorted_idx[:point125]
arr_reference['others125'] = sorted_idx[:-point125]
arr_reference['worst125'] = sorted_idx[-point125:]

with open("arr_reference.json", "w") as f:
    json.dump(arr_reference, f, indent=2)