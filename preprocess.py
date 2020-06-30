import numpy as np
import base64
import os

if __name__ == '__main__':
    root_dir = os.path.join('data', 'features')
    if not os.path.isdir(root_dir):
        os.makedirs(root_dir)
    with open('./data/train/train.tsv', 'r') as f:
        i = 0
        for line in f:
            if i > 0:
                columns = line.split('\t')
                features = np.frombuffer(base64.b64decode(columns[5]), dtype=np.float32).reshape(-1, 2048)
                boxes = np.frombuffer(base64.b64decode(columns[4]), dtype=np.float32).reshape(-1, 4)
                class_labels = np.frombuffer(base64.b64decode(columns[6]), dtype=np.int64).reshape(-1)
                np.savez_compressed(os.path.join(root_dir, '{}.npz'.format(columns[0])),
                                    features=features, boxes=boxes, class_labels=class_labels)
            i += 1
