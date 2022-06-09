#!/usr/bin/env python
# -*- coding: utf-8 -*-
#==============================================================================

from keras import backend as K
from keras.callbacks import Callback
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score


class FScoreValues:
    def __init__(self, fscore, precision, recall):
        self.fscore = fscore
        self.precision = precision
        self.recall = recall

    def __str(self):
        return ("%.4f\t%.4f\t%.4f" % (self.fscore, self.precision, self.recall))
    def __repr__(self):
        return ("%.4f\t%.4f\t%.4f" % (self.fscore, self.precision, self.recall))

    def isImprovedBy(self, fscore_values_val):
        return fscore_values_val.fscore > self.fscore




class Fscore(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
 
    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.model.validation_data[0]))).round()
        val_targ = self.model.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print (" — val_f1: %f — val_precision: %f — val_recall %f" %(_val_f1, _val_precision, _val_recall))
        conf_matrix = confusion_matrix(val_targ, val_predict)
        print(conf_matrix)
        return


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def prepare_imgs_to_sklearn(binarized_imgs_cat, labels_target_cat, axis=3):
    binarized_imgs_2D = np.argmax(binarized_imgs_cat, axis=axis)

    if (len(labels_target_cat.shape) == 4):
        labels_target_2D = np.argmax(labels_target_cat, axis=axis)
    else:
        labels_target_2D = np.argmax(labels_target_cat, axis=axis)
    
    num_pixels = np.ma.size(binarized_imgs_2D)
    assert(num_pixels == np.ma.size(labels_target_2D))

    binarized_imgs_1D = binarized_imgs_2D.ravel()
    labels_target_1D = labels_target_2D.ravel()

    return [binarized_imgs_1D, labels_target_1D]


def macroFScore(binarized_imgs, labels_target):
    return f1_score(y_pred=binarized_imgs, y_true=labels_target, average='macro')
    
def FScoreAllClasses(binarized_imgs, labels_target, num_classes = None):
    macro_fscore_val_source_classes = f1_score(y_pred=binarized_imgs, y_true=labels_target, average=None)

    if num_classes is not None:
        if len(macro_fscore_val_source_classes) < num_classes:
            macro_fscore_val_source_classes_aux = []
            idx_macrofscore = 0
            for idx_class in range(num_classes):
                if idx_class in macro_fscore_val_source_classes:
                    macro_fscore_val_source_classes_aux.append(macro_fscore_val_source_classes[idx_macrofscore])
                    idx_macrofscore = idx_macrofscore+1
                else:
                    macro_fscore_val_source_classes_aux.append(0.)
            macro_fscore_val_source_classes = macro_fscore_val_source_classes_aux
    return macro_fscore_val_source_classes

def macroRecall(binarized_imgs, labels_target):
    return recall_score(y_pred=binarized_imgs, y_true=labels_target, average='macro')
    
def macroPrecision(binarized_imgs, labels_target):
    return precision_score(y_pred=binarized_imgs, y_true=labels_target, average='macro')

def accuracy(prediction, labels):
    return accuracy_score(y_pred=prediction, y_true=labels)