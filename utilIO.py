# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import cv2
import numpy as np
import util
import utilConst
from collections import Counter


# -----------------------------------------------------------------------------
def load_array_of_files(basepath, folders, truncate=False):
    X = []
    for folder in folders:
        full_path = os.path.join(basepath, folder)
        array_of_files = util.list_files(full_path, ext='png')

        for fname_x in array_of_files:
            X.append(fname_x)

    if truncate:
        X = X[:10]

    return np.asarray(X)


def load_folds_names_list(path_dataset, dbnames):

    train_folders = []
    test_folders = []

    for dbname in dbnames:
        train_dbname, test_dbname = load_folds_names(path_dataset, dbname)
        train_folders.append(train_dbname)
        test_folders.append(test_dbname)

    return train_folders, test_folders


# ----------------------------------------------------------------------------
def load_folds_names(path_dataset, dbname):
    assert dbname in utilConst.ARRAY_DBS

    train_folds = []
    test_folds = []


    EINSIELDELN_TEXT_train = ['Einsiedeln/Text/train/ein_GR']
    EINSIELDELN_TEXT_test  = ['Einsiedeln/Text/test/ein_GR']

    EINSIELDELN_NOTES_train = ['Einsiedeln/Notes/train/ein_GR']
    EINSIELDELN_NOTES_test  = ['Einsiedeln/Notes/test/ein_GR']

    EINSIELDELN_STAFF_train = ['Einsiedeln/Staff/train/ein_GR']
    EINSIELDELN_STAFF_test  = ['Einsiedeln/Staff/test/ein_GR']

    EINSIELDELN_INK_train = ['Einsiedeln/Ink/train/ein_GR']
    EINSIELDELN_INK_test  = ['Einsiedeln/Ink/test/ein_GR']

    SALZINNES_TEXT_train = ['Salzinnes/Text/train/sal_GR']
    SALZINNES_TEXT_test = ['Salzinnes/Text/test/sal_GR']

    SALZINNES_NOTES_train = ['Salzinnes/Notes/train/sal_GR']
    SALZINNES_NOTES_test = ['Salzinnes/Notes/test/sal_GR']

    SALZINNES_STAFF_train = ['Salzinnes/Staff/train/sal_GR']
    SALZINNES_STAFF_test = ['Salzinnes/Staff/test/sal_GR']

    SALZINNES_INK_train = ['Salzinnes/Ink/train/sal_GR']
    SALZINNES_INK_test = ['Salzinnes/Ink/test/sal_GR']

    SALZINNES_RECORTED_TEXT_train = ['Salzinnes-recorted/Text/train/sal_GR']
    SALZINNES_RECORTED_TEXT_test = ['Salzinnes-recorted/Text/test/sal_GR']

    SALZINNES_RECORTED_NOTES_train = ['Salzinnes-recorted/Notes/train/sal_GR']
    SALZINNES_RECORTED_NOTES_test = ['Salzinnes-recorted/Notes/test/sal_GR']

    SALZINNES_RECORTED_STAFF_train = ['Salzinnes-recorted/Staff/train/sal_GR']
    SALZINNES_RECORTED_STAFF_test = ['Salzinnes-recorted/Staff/test/sal_GR']

    SALZINNES_RECORTED_INK_train = ['Salzinnes-recorted/Ink/train/sal_GR']
    SALZINNES_RECORTED_INK_test = ['Salzinnes-recorted/Ink/test/sal_GR']

    B_59_850_INK_train = ['b-59-850/Ink/train/cap_GR']
    B_59_850_INK_test =  ['b-59-850/Ink/test/cap_GR']

    B_59_850_STAFF_train = ['b-59-850/Staff/train/cap_GR']
    B_59_850_STAFF_test =  ['b-59-850/Staff/test/cap_GR']

    B_59_850_TEXT_train = ['b-59-850/Text/train/cap_GR']
    B_59_850_TEXT_test =  ['b-59-850/Text/test/cap_GR']

    B_59_850_NOTES_train = ['b-59-850/Notes/train/cap_GR']
    B_59_850_NOTES_test =  ['b-59-850/Notes/test/cap_GR']

    if dbname == 'ein-text':
        train_folds = EINSIELDELN_TEXT_train
        test_folds = EINSIELDELN_TEXT_test
    elif dbname == 'ein-notes':
        train_folds = EINSIELDELN_NOTES_train
        test_folds = EINSIELDELN_NOTES_test
    elif dbname == 'ein-staff':
        train_folds = EINSIELDELN_STAFF_train
        test_folds = EINSIELDELN_STAFF_test
    elif dbname == 'ein-ink':
        train_folds = EINSIELDELN_INK_train
        test_folds = EINSIELDELN_INK_test
    elif dbname == 'sal-text':
        train_folds = SALZINNES_TEXT_train
        test_folds = SALZINNES_TEXT_test
    elif dbname == 'sal-notes':
        train_folds = SALZINNES_NOTES_train
        test_folds = SALZINNES_NOTES_test
    elif dbname == 'sal-staff':
        train_folds = SALZINNES_STAFF_train
        test_folds = SALZINNES_STAFF_test
    elif dbname == 'sal-ink':
        train_folds = SALZINNES_INK_train
        test_folds = SALZINNES_INK_test
    elif dbname == 'sal-rec-text':
        train_folds = SALZINNES_RECORTED_TEXT_train
        test_folds = SALZINNES_RECORTED_TEXT_test
    elif dbname == 'sal-rec-notes':
        train_folds = SALZINNES_RECORTED_NOTES_train
        test_folds = SALZINNES_RECORTED_NOTES_test
    elif dbname == 'sal-rec-staff':
        train_folds = SALZINNES_RECORTED_STAFF_train
        test_folds = SALZINNES_RECORTED_STAFF_test
    elif dbname == 'sal-rec-ink':
        train_folds = SALZINNES_RECORTED_INK_train
        test_folds = SALZINNES_RECORTED_INK_test
    elif dbname == 'b-59-850-ink':
        train_folds = B_59_850_INK_train
        test_folds = B_59_850_INK_test
    elif dbname == 'b-59-850-notes':
        train_folds = B_59_850_NOTES_train
        test_folds = B_59_850_NOTES_test
    elif dbname == 'b-59-850-text':
        train_folds = B_59_850_TEXT_train
        test_folds = B_59_850_TEXT_test
    elif dbname == 'b-59-850-staff':
        train_folds = B_59_850_STAFF_train
        test_folds = B_59_850_STAFF_test
    else:
        raise Exception('Unknown database name')

    
    train_folds_path = []
    for train_fold in train_folds:
        train_folds_path.append(path_dataset + "/" + train_fold)

    test_folds_path = []
    for test_fold in test_folds:
        test_folds_path.append(path_dataset + "/" + test_fold)

    return train_folds_path, test_folds_path


def getGTFileNameFromGRFileName(gr_file_name):
    return gr_file_name.replace(utilConst.X_SUFIX, utilConst.Y_SUFIX)