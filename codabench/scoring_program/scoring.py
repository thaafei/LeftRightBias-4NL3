import json
import os
import pandas as pd
from sklearn.metrics import f1_score

reference_dir = os.path.join('/app/input/', 'ref')
prediction_dir = os.path.join('/app/input/', 'res')
score_dir = '/app/output/'
# reference_dir = os.path.join("./app/reference_data/")
# prediction_dir = os.path.join("./app/output/")


print('Reading prediction')

prediction = pd.read_csv(os.path.join(prediction_dir, 'prediction.csv'), header=None, dtype=str).iloc[:,0]
truth = pd.read_csv(os.path.join(reference_dir, 'testing_label.csv'), header=0, dtype=str).iloc[:,0]

with open(os.path.join(prediction_dir, 'metadata.json')) as f:
    duration = json.load(f).get('duration', -1)

print('Checking F1 Score')
f1 = f1_score(truth, prediction, average="micro")


print('Scores:')
scores = {
    'f1_score': f1,
    'duration': duration
}
print(scores)

with open(os.path.join(score_dir, 'scores.json'), 'w') as score_file:
    score_file.write(json.dumps(scores))
