import os
import sys
import time
import json
import pandas as pd

input_dir = '/app/input_data/'
output_dir = '/app/output/'
program_dir = '/app/program'
submission_dir = '/app/ingested_program'

sys.path.append(program_dir)
sys.path.append(submission_dir)


def get_training_data():
    X_train = pd.read_csv(os.path.join(input_dir, "training_data.csv"), keep_default_na=False, dtype=str)
    y_train = pd.read_csv(os.path.join(input_dir, 'training_label.csv'), keep_default_na=False, dtype=str)

    # Display the first 5 rows
    print("Sample Training Data")
    print(X_train.head())
    print(y_train.head())
    return X_train, y_train


def get_prediction_data():
    pred = pd.read_csv(os.path.join(input_dir, 'testing_data.csv'), keep_default_na=False, dtype=str)
    print("Sample Prediction Data")
    print(pred.head())
    return pred


def main():
    from model import Model
    print('Reading Data')
    X_train, y_train = get_training_data()
    X_test = get_prediction_data()
    print('-' * 10)
    print('Starting')
    start = time.time()
    m = Model()
    print('-' * 10)
    print('Training Model')

    m.fit(X_train, y_train)
    print('-' * 10)
    print('Running Prediction')
    prediction = m.predict(X_test)
    duration = time.time() - start
    print('-' * 10)
    print(f'Completed Prediction. Total duration: {duration}')

    pd.DataFrame(prediction).to_csv(
        os.path.join(output_dir, "prediction.csv"),
        index=False,
        header=False
    )
    with open(os.path.join(output_dir, 'metadata.json'), 'w+') as f:
        json.dump({'duration': duration}, f)
    print()
    print('Ingestion Program finished. Moving on to scoring')


if __name__ == '__main__':
    main()
