import json

import os

import numpy as np

if __name__ == '__main__':
    file = '/scratch/partial_datasets/niudt/project/llarva_v2/close_jar_initial_tests/train_164971_nov21.json'
    file_json = json.load(open(file))
    s = 1
    idx = 1289
    batch_size = 32

    tmp = file_json#[idx*batch_size : (idx + 1)*batch_size]
    questions = []
    answers = []
    for ann in tmp:
        s = 1
        assert len(ann['image'])==16
        ques =  len(ann['conversations'][0]['value'])
        questions.append(ques)
        answer = len(ann['conversations'][1]['value'])
        answers.append(answer)

    questions = np.array(questions)
    answers = np.array(answers)
    s = 1
