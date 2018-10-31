from collections import Counter

import numpy as np
import tabulate
from sklearn import metrics

HEADERS_TRAIN = ['phase', 'epoch', 'batch_index', 'average_loss', 'accuracy', 'f1', 'precision', 'recall', 'conf_matrix']
HEADERS_TEST = ['phase', 'accuracy', 'f1', 'precision', 'recall', 'conf_matrix']

def evaluator_binary_classif_one(y_true, y_pred):
    """ evaluator for one prediction training or testing (binary)
    """
    return [metrics.accuracy_score(y_true, y_pred), metrics.f1_score(y_true, y_pred), metrics.precision_score(y_true, y_pred), metrics.recall_score(y_true, y_pred), metrics.confusion_matrix(y_true, y_pred)]

class Evaluator():
    def __init__(self, dataset, path, phase = 'training'):
        self.metrics = []
        self.phase = phase
        self.epoch = 0
        self.batch_id = -1
        self.error_track_train = Counter()
        self.dataset = dataset
    def new_epoch(self):
        self.y_pred = []
        self.y_true = []
        self.losses = []
        self.epoch += 1
        self.batch_id = -1
    def update_all(self, targets, predictions, indexes, loss = None):
        if (targets == predictions).sum() != len(targets):
            self.error_track_train += Counter(list(indexes[((targets == predictions) == 0).nonzero()[:,0]].flatten().numpy()))
        self.batch_id += 1
        self.y_true += list(targets.flatten().numpy())
        self.y_pred += list(predictions.flatten().numpy())
        if loss:
            self.losses.append(float(loss))
    def __str__(self):
        if self.phase == 'training':
            res = [self.phase, self.epoch, self.batch_id, np.mean(self.losses)]
        else:
            res = [self.phase]
        res += evaluator_binary_classif_one(self.y_true, self.y_pred)
        self.y_pred = []
        self.y_true = []
        self.losses = []
        self.metrics.append(res)
        return tabulate.tabulate(self.metrics, headers = HEADERS_TEST if self.phase == 'testing' else HEADERS_TRAIN)
    def save(self):
        error_track_train_to_save = {}
        for key, value in self.error_track_train.items():
            error_track_train_to_save[self.dataset.files[key]] = value
        f = open('error_track_train.txt','w')
        f.write(str(error_track_train_to_save))
        f.close()
