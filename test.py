import pandas as pd

splits = {'train': 'train.jsonl', 'validation': 'validation.jsonl', 'test': 'test.jsonl'}
df = pd.read_json("hf://datasets/AdamLucek/youtube-titles/" + splits["train"], lines=True)

df.to_csv('dataset.csv')
