# -*- coding: utf-8 -*-
from __future__ import print_function
import sys, os, re
import pandas as pd
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import tensorflow as tf
import shutil
import utilConst
import keras
import progressbar
import utilMetrics
from Results import Results
from CustomJson import CustomJson
import GTJSONReaderMuret
from collections import Counter

import sklearn.metrics
from sklearn.metrics import precision_recall_fscore_support

import os, sys
from os import listdir
from os.path import isfile, join, exists

import cv2


def readListPathsWithinlistFiles(list_pathfile):
    
    content = readListStringFiles(list_pathfile)
    list_items = content.split("\n")

    for idx in range(len(list_items)):
        if list_items[idx] == '':
            list_items.pop(idx)

    return list_items


def readListStringFiles(list_pathfile):
    content_files = ""
    for path_file in list_pathfile:
        content_files += readStringFile(path_file)

    return content_files

def readStringFile(path_file):
    assert type(path_file) == str

    f = open(path_file)
    
    content = f.read()
    f.close()
    
    assert type(content) == str

    return content

def saveImage(img, path):
    dir_output = os.path.dirname(path)
    mkdirp(dir_output)

    cv2.imwrite(path, img)

#-------------------------------------------------------------
# Create a progress bar object with a message
#-------------------------------------------------------------
def createProgressBar(msg, max_steps):
    return progressbar.ProgressBar(
                            maxval=max_steps, 
                            widgets=
                                    [
                                        msg,
                                        ' ',
                                        progressbar.Bar('=', '[', ']'), 
                                        ' ', 
                                        progressbar.Percentage(), 
                                        ' ', 
                                        progressbar.Timer(), 
                                        ' ', 
                                        progressbar.ETA()])


def init():
    np.set_printoptions(threshold=sys.maxsize)
    sys.setrecursionlimit(40000)
    random.seed(42)                             # For reproducibility
    np.random.seed(42)
    #tf.compat.v1.set_random_seed(42)
    tf.random.set_seed(42)

def print_stats(var_name, var):
    print(' - {}: shape {} - min {:.2f} - max {:.2f} - mean {:.2f} - std {:.2f}'.format(
            var_name, var.shape, np.min(var), np.max(var), np.mean(var), np.std(var)))

def mkdirp(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)

def deleteFolder(directory):
    if os.path.isdir(directory):
        shutil.rmtree(directory, ignore_errors=True)

# ----------------------------------------------------------------------------
# Return the list of files in folder
def list_dirs(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, f))]


# ----------------------------------------------------------------------------
# Return the list of files in folder
# ext param is optional. For example: 'jpg' or 'jpg|jpeg|bmp|png'
def list_files(directory, ext=None):
    return [os.path.join(directory, f) for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f)) and ( ext==None or re.match('([\w_-]+\.(?:' + ext + '))', f) )]

def listFiles (path_dir):
    assert type(path_dir) == str
    if (exists(path_dir) == False):
        path_dir = "../" + path_dir
    
    list_files = ([f for f in listdir(path_dir) if isfile(join(path_dir, f))])
    list_files.sort()
    return list_files

def listFilesRecursive(path_dir):
    
    try:
        listOfFile = os.listdir(path_dir)
    except Exception:
        pathdir_exec = os.path.dirname(os.path.abspath(__file__))
        path_dir = pathdir_exec + "/" + path_dir
        listOfFile = os.listdir(path_dir)

    list_files = list()
    
    for entry in listOfFile:
        fullPath = os.path.join(path_dir, entry)
        if os.path.isdir(fullPath):
            list_files = list_files + listFilesRecursive(fullPath)
        else:
            list_files.append(fullPath)
    
    list_files.sort()            
    return list_files


def list_files_all_paths_folders(directories, ext=None):
    all_files = []
    for list_directories in directories:
        files_dir = list_files_all_paths(list_directories, ext)
        all_files.append(files_dir)

    return all_files


def list_files_all_paths(directories, ext=None):
    all_files = []
    for directory in directories:
        files = list_files(directory, ext)
        for f in files:
            all_files.append(f)
    
    all_files.sort()
    return all_files


def sample_is_selected(gr_sample, gt_sample, sample_filter):

    if sample_filter == utilConst.FILTER_WITH_GT_INFO:
        if np.amax(gt_sample) > 0:
            return True
        else:
            return False
    elif sample_filter == utilConst.FILTER_ENTROPY:
        assert(False) # to be implemented
        return True
    elif sample_filter == utilConst.FILTER_WITHOUT:
        return True
    else:
        return True

def generate_chunks(gr_img, gt_img, windows_shape, sample_filter=None, number_samples=None):

    if gt_img is not None:
        assert(gr_img.shape[0] == gt_img.shape[0] and gr_img.shape[1] == gt_img.shape[1])

    min_row = 0
    min_col = 0
    max_row = gr_img.shape[0] - windows_shape[0]
    max_col = gr_img.shape[1] - windows_shape[1]

    gr_chunks = []
    gt_chunks = []

    if (number_samples is not None):

        for i in range(number_samples):
            row = random.randint(min_row, max_row) 
            col = random.randint(min_col, max_col) 

            gr_sample = gr_img[row:row+windows_shape[0], col:col+windows_shape[1]]
            gt_sample = gt_img[row:row+windows_shape[0], col:col+windows_shape[1]]

            if (sample_is_selected(gr_sample, gt_sample, sample_filter)):
                gr_sample.reshape(1, windows_shape[0], windows_shape[1], 1)
                gr_chunks.append(gr_sample)

                #gt_chunk_categorical = keras.utils.to_categorical(gt_sample, 2)
                gt_chunks.append(gt_sample)
            else:
                i-=1
    else:
        for row in range(min_row, max_row, windows_shape[0]):
            for col in range(min_col, max_col, windows_shape[1]):
                
                row = min(row, max_row)
                col = min(col, max_col)
                
                gr_sample = gr_img[row:row+windows_shape[0], col:col+windows_shape[1]]

                if (gt_img is not None):
                    gt_sample = gt_img[row:row+windows_shape[0], col:col+windows_shape[1]]

                    if (sample_is_selected(gr_sample, gt_sample, sample_filter)):
                        gr_sample.reshape(1, windows_shape[0], windows_shape[1], 1)
                        gr_chunks.append(gr_sample)

                        #gt_chunk_categorical = keras.utils.to_categorical(gt_sample, 2)
                        gt_chunks.append(gt_sample)
                    else:
                        i-=1
                else:
                    gr_sample.reshape(1, windows_shape[0], windows_shape[1], 1)
                    gr_chunks.append(gr_sample)

    return gr_chunks, gt_chunks



def calculate_number_samples_per_file(list_files, total_number_samples):
    num_samples_per_file = {}

    if total_number_samples is not None:
        shape_per_file = {}
        total_px = 0
        for fname_gr in list_files:
            gr_img = cv2.imread(fname_gr, cv2.IMREAD_GRAYSCALE)
            shape_per_file[fname_gr] = gr_img.shape
            total_px += gr_img.shape[0] * gr_img.shape[1]

    total_samples_computed = 0
    for fname_gr in list_files:
        if total_number_samples is not None:
            shape_file = shape_per_file[fname_gr]

            px_file = shape_file[0]*shape_file[1]
            num_samples_file = int((float(px_file) / total_px) * total_number_samples)
        
            num_samples_per_file[fname_gr] = num_samples_file
            total_samples_computed += num_samples_file
        else:
            num_samples_per_file[fname_gr] = None

    print (num_samples_per_file)
    print ("Total samples: " + str(total_samples_computed))
    return num_samples_per_file


def resizeImage(img, height, width, interpolation = cv2.INTER_LINEAR):
    img2 = img.copy().astype('float32')
    return cv2.resize(img2,(width,height), interpolation=interpolation)


def loadImageAndGTData(fname_gr, fname_gt, windows_shape, considered_classes, vertical_reduction_regions = 0.2):
    js = CustomJson()
    js.loadJson(fname_gt)
    gtjson = GTJSONReaderMuret.GTJSONReaderMuret()
    gtjson.load(js)

    bboxes = gtjson.getListBoundingBoxesPerClass(considered_classes)
    
    gr_img = 1.0 - (cv2.imread(fname_gr, cv2.IMREAD_GRAYSCALE) / 255.)
    gt_img = gtjson.generateGT(considered_classes, gr_img.shape, vertical_reduction_regions = vertical_reduction_regions)

    gr_img = resizeImage(gr_img, windows_shape[0], windows_shape[1])
    gt_img = resizeImage(gt_img, windows_shape[0], windows_shape[1])

    return gr_img, gt_img, bboxes

def create_generator(list_files, list_files_json, windows_shape, sample_filter, batch_size, considered_classes):

    assert(len(list_files) > 0)
    if (type(list_files[0]) is list):
        list_files_use = [item for sublist in list_files for item in sublist]
        list_files_use_json = [item for sublist in list_files_json for item in sublist]
    else:
        list_files_use = list_files
        list_files_use_json = list_files_json

    all_indexes = [idx for idx in range(len(list_files_use))] 
        
    while(True):
        
        random.shuffle(all_indexes)
        gr_chunks = []
        gt_chunks = []

        for index_selected in all_indexes:
            fname_gr = list_files[index_selected]
            fname_gt = list_files_json[index_selected]
            
            gr_img, gt_img, bboxes = loadImageAndGTData(fname_gr, fname_gt, windows_shape, considered_classes)

            gr_chunks.append(gr_img)
            gt_chunks.append(gt_img)

            #print("Number of samples extracted: " + str(len(gr_chunks)))
            assert(len(gr_chunks) == len(gt_chunks))

            if len(gr_chunks) == batch_size:
                gr_chunks_arr = np.asarray(gr_chunks)
                gt_chunks_arr = np.asarray(gt_chunks)

                yield gr_chunks_arr, gt_chunks_arr
                gr_chunks = []
                gt_chunks = []



def computeBestThreshold(prediction_imgs, gt_imgs, out_fnames, verbose, fixed_threshold=None):

    th_precision = 10
    min_th = 0
    max_th = th_precision
    if fixed_threshold is not None:
        assert(fixed_threshold>=0 and fixed_threshold<=1)
        min_th = int(fixed_threshold*th_precision)
        max_th = int(fixed_threshold*th_precision) + 1

    if verbose is True:
        print(len(prediction_imgs))
    min_prob_pred = 1.
    max_prob_pred = 0.
    for idx_img in range(len(prediction_imgs)):
        pred_img = prediction_imgs[idx_img]
        gt_img = gt_imgs[idx_img]
        
        min_prob_pred = min(min_prob_pred, np.amin(pred_img))
        max_prob_pred = max(max_prob_pred, np.amax(gt_img))
        
    if min_prob_pred == 0 and max_prob_pred == 0:
        return 0., 0., 0., 0., 0., 0., 0., 0.

    best_precision = -100.
    best_recall = -100.
    best_fscore = -100.
    best_threshold = -100.

    best_pseudo_precision = -100.
    best_pseudo_recall = -100.
    best_pseudo_fscore = -100.
    best_pseudo_threshold = -100.

    print("Selecting threshold...")
    for th in range(min_th, max_th):
        th /= float(th_precision)

        if fixed_threshold is None:
            if th < min_prob_pred or th > max_prob_pred:
                continue

        fscore, precision, recall, pseudo_fscore, pseudo_precision, pseudo_recall = utilMetrics.evaluateMetrics(prediction_imgs, gt_imgs, 1, verbose, th)

        print (
                "P: %.3f" % precision, 
                "R: %.3f" % recall,
                "F1: %.3f" % fscore,
                "pseudo-P: %.3f" % pseudo_precision, 
                "pseudo-R: %.3f" % pseudo_recall,
                "pseudo-F1: %.3f" % pseudo_fscore,
                "Th: %.3f" % th
            )

        if fscore > best_fscore:
            best_fscore = fscore
            best_precision = precision
            best_recall = recall
            best_threshold = th
    
        if pseudo_fscore > best_pseudo_fscore:
            best_pseudo_fscore = pseudo_fscore
            best_pseudo_precision = pseudo_precision
            best_pseudo_recall = pseudo_recall
            best_pseudo_threshold = th

        if pseudo_fscore < best_pseudo_fscore:
            break

    idx = 0
    if out_fnames is not None:
        for prediction_img in prediction_imgs:
            saveImage((prediction_img > best_threshold)*255, out_fnames[idx])
            idx += 1

        idx = 0
        for prediction_img in prediction_imgs:
            saveImage((prediction_img > best_pseudo_threshold)*255, out_fnames[idx].replace(".png", "_pseudo.png"))
            idx += 1

    print ("Best threshold selected: " + str(best_pseudo_threshold))
    return best_pseudo_threshold, best_pseudo_precision, best_pseudo_recall, best_pseudo_fscore, best_threshold, best_precision, best_recall, best_fscore


def computeBestThresholdSklearn(prediction_imgs, gt_imgs, out_fnames):

    th_prec = 10
    
    pred_1D = np.asarray(prediction_imgs).flatten()
    gt_1D = np.asarray(gt_imgs).flatten()

    min_prob_pred = np.amin(pred_1D)
    max_prob_pred = np.amax(pred_1D)
        
    if min_prob_pred == 0 and max_prob_pred == 0:
        return 0., 0., 0., 0.

    best_precision = 0.
    best_recall = 0.
    best_fscore = 0.
    best_threshold = 0.
    print("Selecting threshold...")
    for th in range(0,th_prec):
        th /= float(th_prec)

        if th < min_prob_pred or th > max_prob_pred:
            continue

        (precision, recall, fscore, support) = sklearn.metrics.precision_recall_fscore_support(1*(pred_1D > th), gt_1D, pos_label = 1, average='binary')
        print (
                "Precision: %.3f" % precision, 
                "Recall: %.3f" % recall,
                "Fscore: %.3f" % fscore,
                "Threshold: %.3f" % th
            )

        if fscore > best_fscore:
            best_fscore = fscore
            best_precision = precision
            best_recall = recall
            best_threshold = th
    
    idx = 0
    for prediction_img in prediction_imgs:
        saveImage((prediction_img > best_threshold)*255, out_fnames[idx])
        idx += 1

    return best_threshold, best_precision, best_recall, best_fscore


def dumpChunksIntoImage(model, gr_chunks, rows_cols, prediction_img, row_overlapping, col_overlapping, batch_size, windows_shape_3D):
    #print(type(gr_chunks))
    gr_chunks = np.asarray(gr_chunks)
    #print(gr_chunks.shape)
    gr_chunks = gr_chunks.reshape(gr_chunks.shape[0], gr_chunks.shape[1], gr_chunks.shape[2], 1)
    predictions_chunks = model.predict(gr_chunks, batch_size=batch_size)
    #print (str(np.amin(predictions_chunks)) + "-" + str(np.amax(predictions_chunks)))
    idx = 0
    for row_col in rows_cols:
        row = row_col[0]
        col = row_col[1]
        prediction_chunk = predictions_chunks[idx]
        #print ( "\t" + str(np.amin(prediction_chunk)) + "-" + str(np.amax(prediction_chunk)))
        #print(str(np.amin(prediction_chunk[:,:,0])) + " - " + str(np.amax(prediction_chunk[:,:,0])) + " || " + str(np.amin(prediction_chunk[:,:,1])) + " - " + str(np.amax(prediction_chunk[:,:,1])))
        
        #prediction_chunk = np.argmax(prediction_chunk, axis=2)
        assert(prediction_chunk.shape == windows_shape_3D)
        pred_chunk = prediction_chunk[row_overlapping//2:-(row_overlapping//2),col_overlapping//2:-(col_overlapping//2),0]
        #print ( "\t\t" + str(np.amin(pred_chunk)) + "-" + str(np.amax(pred_chunk)))

        prediction_img[row+row_overlapping//2:row+windows_shape_3D[0]-row_overlapping//2, col+col_overlapping//2:col+windows_shape_3D[1]-col_overlapping//2] = pred_chunk
        idx += 1


def evaluateDomainModelOneDomain(accuracy_accumulative_domain, model, list_files, gt_domain_db, windows_shape, batch_size, verbose, fixed_threshold=None):
    assert(len(list_files) > 0)
    if (type(list_files[0]) is list):
        list_files_use = [item for sublist in list_files for item in sublist]
    else:
        list_files_use = list_files

    accuracy_domain = tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=fixed_threshold)

    for fname_gr in list_files_use:
        gr_img = 1.0 - (cv2.imread(fname_gr, cv2.IMREAD_GRAYSCALE) / 255.)

        min_row = 0
        min_col = 0
        max_row = gr_img.shape[0] - windows_shape[0]
        max_col = gr_img.shape[1] - windows_shape[1]
        
        overlapping = 0.2
        row_overlapping = int(windows_shape[0] * overlapping)
        col_overlapping = int(windows_shape[1] * overlapping)

        gr_chunks = []
        cont_sample_batch = 0
        for row in range(min_row, max_row, windows_shape[0]-row_overlapping):
            for col in range(min_col, max_col, windows_shape[1] - col_overlapping):
                row = min(row, max_row)
                col = min(col, max_col)

                gr_sample = gr_img[row:row+windows_shape[0], col:col+windows_shape[1]]
                cont_sample_batch+=1
                gr_chunks.append(gr_sample)

                if cont_sample_batch >= batch_size:
                    gr_chunks_array = np.asarray(gr_chunks)
                    gr_chunks_array = gr_chunks_array.reshape(gr_chunks_array.shape[0], gr_chunks_array.shape[1], gr_chunks_array.shape[2], 1)

                    gr_domain = gr_chunks_array
                    gt_domain = gt_domain_db[0:len(gr_chunks_array)]

                    logits_domain = model(tf.convert_to_tensor(gr_domain, dtype=tf.float32))
                    accuracy_domain.update_state(gt_domain, logits_domain)
                    accuracy_accumulative_domain.update_state(gt_domain, logits_domain)

                    gr_chunks = []
                    cont_sample_batch = 0

    return accuracy_domain, accuracy_accumulative_domain

def evaluateDomainModel(model, list_files_source, list_files_target, windows_shape, batch_size, verbose, fixed_threshold=None):
    print ("Validating domain classifier...")
    print ("source")
    print(list_files_source)
    print ("target")
    print(list_files_source)
    windows_shape_3D = (windows_shape[0], windows_shape[1], 1)

    gt_domain_db1 = np.zeros((windows_shape[0], windows_shape[1]), dtype=np.float32)
    gt_domain_db2 = np.ones((windows_shape[0], windows_shape[1]), dtype=np.float32)
    
    gt_domains_db1 = []
    gt_domains_db2 = []
    for _ in range(batch_size):
        gt_domains_db1.append(gt_domain_db1)
    for _ in range(batch_size):
        gt_domains_db2.append(gt_domain_db2)

    gt_domains_db1 = np.asarray(gt_domains_db1)
    gt_domains_db2 = np.asarray(gt_domains_db2)
    
    accuracy_domain = tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=fixed_threshold)

    print("Predicting domain for source...")
    accuracy_source, accuracy_accumulative_domain = evaluateDomainModelOneDomain(accuracy_domain, model, list_files_source, gt_domains_db1, windows_shape, batch_size, verbose, fixed_threshold)
    print("Predicting domain for target...")
    accuracy_target, accuracy_accumulative_domain = evaluateDomainModelOneDomain(accuracy_domain, model, list_files_target, gt_domains_db2, windows_shape, batch_size, verbose, fixed_threshold)

    print('-'*80)
    print (
                "Acc domain source: %.3f" % accuracy_source.result().numpy(), 
                "Acc domain target: %.3f" % accuracy_target.result().numpy(),
                "Acc avg: %.3f" % accuracy_accumulative_domain.result().numpy()
            )
    print('-'*80)

    return accuracy_source, accuracy_target, accuracy_accumulative_domain



def printResults(dbnames, results):
    for i in range(len(dbnames)):
        dbname = dbnames[i]
        results_dbname = results[i]
        print (dbname)
        print(results_dbname)



def evaluateAutoModelListFolders(
                            db_name_source,
                            db_name_target,
                            db_name_test,
                            sae_model, 
                            dann_model, 
                            list_files, 
                            windows_shape, 
                            batch_size, 
                            verbose, 
                            norm_histogram_source,
                            norm_histogram_target, 
                            threshold_correlation = 0.25,
                            fixed_threshold_sae=None, fixed_threshold_dann=None):
    
    results_autodann = []
    results_autodann_ideal = []
    
    for list_files_i in list_files:
        results_autodann_i, results_autodann_ideal_i = evaluateAutoModel(
                            db_name_source,
                            db_name_target,
                            db_name_test,
                            sae_model, 
                            dann_model, 
                            list_files_i, 
                            windows_shape, 
                            batch_size, 
                            verbose, 
                            norm_histogram_source,
                            norm_histogram_target,
                            threshold_correlation,
                            fixed_threshold_sae, 
                            fixed_threshold_dann)
        results_autodann.append(results_autodann_i)
        results_autodann_ideal.append(results_autodann_ideal_i)

    return results_autodann, results_autodann_ideal


def evaluateModelListFolders(name_experiment, model, list_files, list_files_json, windows_shape, batch_size, verbose, considered_classes, fixed_threshold=None):
    list_results = []
    results = evaluateModel(name_experiment, model, list_files, list_files_json, windows_shape, batch_size, verbose, considered_classes, fixed_threshold)
    
    list_results.append(results)
    return list_results


def evaluateModel(name_experiment, model, list_files, list_files_json, windows_shape, batch_size, verbose, considered_classes, fixed_threshold=None):
    windows_shape_3D = (windows_shape[0], windows_shape[1], 1)

    prediction_imgs = []
    gt_imgs = []
    out_fnames = []

    gr_chunks = []

    threshold = 0.5
    if fixed_threshold is not None:
        threshold = fixed_threshold

    for idx in range(len(list_files)):
        fname_gr = list_files[idx]
        fname_gt = list_files_json[idx]
        
        js = CustomJson()
        js.loadJson(fname_gt)
        gtjson = GTJSONReaderMuret.GTJSONReaderMuret()
        gtjson.load(js)
        
        
        gr_img = 1.0 - (cv2.imread(fname_gr, cv2.IMREAD_GRAYSCALE) / 255.)
        gt_img = gtjson.generateGT(considered_classes, gr_img.shape, vertical_reduction_regions = 0.2)

        gr_img_resized = resizeImage(gr_img, windows_shape[0], windows_shape[1])

        
        overlapping = 0.2
        row_overlapping = int(windows_shape[0] * overlapping)
        col_overlapping = int(windows_shape[1] * overlapping)

        cont_sample_batch = 0
        
        list_gr_img_resized = []
        list_gr_img_resized.append(gr_img_resized)
        gr_chunks = np.asarray(list_gr_img_resized)
        prediction_img = model.predict(gr_chunks)[0,:,:,0]

        out_fname = "output" + "/" + str(name_experiment) + "/" + fname_gr
        dir_output = os.path.dirname(out_fname)
        mkdirp(dir_output)

        min_prob_pred = np.amin(prediction_img)
        max_prob_pred = np.amax(prediction_img)

        print(str(min_prob_pred) + "-" + str(max_prob_pred))
        
        saveImage(prediction_img*255, out_fname.replace(".png", "_prob.png").replace(".JPG", "_prob.png").replace(".png", "_prob.png"))
        saveImage((prediction_img>=threshold)*255, out_fname.replace(".png", "_th.png").replace(".JPG", "_th.png").replace(".png", "_th.png"))

        prediction_img_normal_size = resizeImage((prediction_img>=threshold)*1, gr_img.shape[0], gr_img.shape[1])
        prediction_imgs.append(((prediction_img_normal_size>=threshold)*1).astype('int16'))
        gt_imgs.append(((gt_img>0)*1).astype('int16'))

        out_fnames.append(out_fname)
        
        out_gt_fname = out_fname.replace(".png", "_gt.png").replace(".JPG", "_gt.png").replace(".png", "_gt.png")
        if os.path.isfile(out_gt_fname) == False:
            saveImage(gt_img*255, out_gt_fname)
        
        out_gr_fname = out_fname.replace(".png", "_gr.png").replace(".JPG", "_gr.png").replace(".png", "_gr.png")
        if os.path.isfile(out_gr_fname) == False:
            saveImage(gr_img*255, out_gr_fname)

    min_prob_pred = np.amin(prediction_imgs[0])
    max_prob_pred = np.amax(prediction_imgs[0])
    print(str(min_prob_pred) + "-" + str(max_prob_pred))


    
    pred_1D = np.asarray(prediction_imgs)
    gt_1D = np.asarray(gt_imgs)

    pred_1D = np.concatenate([x.ravel() for x in pred_1D])
    gt_1D = np.concatenate([x.ravel() for x in gt_1D])


    (precision, recall, fscore, support) = sklearn.metrics.precision_recall_fscore_support(y_pred=pred_1D*1, y_true=gt_1D*1, pos_label = 1, average='binary')
    
    pseudo_fscore = fscore
    pseudo_precision = precision
    pseudo_recall = recall
    pseudo_threshold = threshold

    #pseudo_threshold, pseudo_precision, pseudo_recall, pseudo_fscore, threshold, precision, recall, fscore = computeBestThreshold(prediction_imgs, gt_imgs, out_fnames, verbose, fixed_threshold)

    results = Results(precision, recall, fscore, threshold, pseudo_precision, pseudo_recall, pseudo_fscore, pseudo_threshold)
    
    print('-'*80)
    print (results)
    print('-'*80)

    return results


def getPrecision(num_decimal):
    precision = 1.
    for _ in range(num_decimal):
        precision /= 10.

    return precision

def getHistogram(image, num_decimal):

    tuple_prediction = tuple(image.reshape(1,-1)[0])

    if num_decimal is not None:
        tuple_prediction_round = []
        for num in tuple_prediction:
            if num > 0.01:
                tuple_prediction_round.append(round(num, num_decimal))
            
        #tuple_prediction_round = [round(num, num_decimal) for num in tuple_prediction]
        tuple_prediction = tuple_prediction_round

        precision = getPrecision(num_decimal)
        
        value = 0.
        value = round(value, num_decimal)
        while value <= 1:
            tuple_prediction.append(value)
            value += precision
            value = round(value, num_decimal)
    
    histogram_prediction = Counter(tuple_prediction)

    probs_bins = []
    for i in range (int(1./precision) + 1):
        probs_bins.append(round(i*precision, num_decimal))
    histogram_list = [histogram_prediction[prob_bin] for prob_bin in probs_bins]

    return histogram_list


def getHistogramTuple(tuple_prediction, num_decimal):

    if num_decimal is not None:
        tuple_prediction_round = []
        for num in tuple_prediction:
            tuple_prediction_round.append(round(num, num_decimal))
            
        #tuple_prediction_round = [round(num, num_decimal) for num in tuple_prediction]
        tuple_prediction = tuple_prediction_round

        precision = getPrecision(num_decimal)
        
        value = 0.
        value = round(value, num_decimal)
        while value <= 1:
            tuple_prediction.append(value)
            value += precision
            value = round(value, num_decimal)
    
    histogram_prediction = Counter(tuple_prediction)

    probs_bins = []
    for i in range (int(1./precision) + 1):
        probs_bins.append(round(i*precision, num_decimal))
    histogram_list = [histogram_prediction[prob_bin]-1 for prob_bin in probs_bins]

    return histogram_list


def predictCompleteImage_AutoDANN(
                                img, 
                                gt_img,
                                windows_shape, 
                                batch_size, 
                                model_sae, 
                                model_dann, 
                                norm_histogram_source,
                                norm_histogram_target, 
                                threshold_correlation,
                                fixed_threshold_sae=None, 
                                fixed_threshold_dann=None,
                                num_decimal = 1):
    finalImg = np.zeros(img.shape, dtype=float)
    finalImg_bin = np.zeros(img.shape, dtype=float)
    finalImg_selection = np.zeros(img.shape, dtype=float)
    finalImg_ideal = np.zeros(img.shape, dtype=float)
    finalImg_ideal_bin = np.zeros(img.shape, dtype=float)
    finalImg_selection_ideal = np.zeros(img.shape, dtype=float)

    min_row = 0
    min_col = 0
    max_row = img.shape[0] - windows_shape[0]
    max_col = img.shape[1] - windows_shape[1]
    
    overlapping = 0.2
    row_overlapping = int(windows_shape[0] * overlapping)
    col_overlapping = int(windows_shape[1] * overlapping)

    number_sae_samples = 0
    number_dann_samples = 0
    number_sae_samples_ideal = 0
    number_dann_samples_ideal = 0

    print("Predicting chunks and generating image for histogram generation...")

    tp_selected = 0
    fp_selected = 0
    tn_selected = 0
    fn_selected = 0

    for row in range(min_row, max_row, windows_shape[0]-row_overlapping):
        for col in range(min_col, max_col, windows_shape[1] - col_overlapping):
            
            row = min(row, max_row)
            col = min(col, max_col)
            gr_sample = img[row:row+windows_shape[0], col:col+windows_shape[1]]
            gt_sample = gt_img[row:row+windows_shape[0], col:col+windows_shape[1]]

            roi = gr_sample.reshape(1, windows_shape[0], windows_shape[1], 1)
            roi = roi.astype('float32')

            sae_prediction = model_sae.predict(roi)
            dann_prediction = model_dann.predict(roi)

            gt_sample_arr = gt_sample.reshape(1, windows_shape[0], windows_shape[1])

            pseudo_threshold_sae, pseudo_precision_sae, pseudo_recall_sae, pseudo_fscore_sae, threshold_sae, precision_sae, recall_sae, fscore_sae = computeBestThreshold(sae_prediction[:,:,:,0], gt_sample_arr, None, False, fixed_threshold_sae)
            pseudo_threshold_dann, pseudo_precision_dann, pseudo_recall_dann, pseudo_fscore_dann, threshold_dann, precision_dann, recall_dann, fscore_dann = computeBestThreshold(dann_prediction[:,:,:,0], gt_sample_arr, None, False, fixed_threshold_dann)

            histogram_sample = getHistogramBins(sae_prediction[0,:,:,0], num_decimal)
            norm_histogram_sample = normalizeHistogram(histogram_sample)

            selection = -1
            correlation = np.corrcoef(norm_histogram_source, norm_histogram_sample)[0, 1]
            print("Source: " + str(norm_histogram_source))
            print("Sample: " + str(norm_histogram_sample))
            print("Pearson's correlation: " + str(correlation))

            if threshold_correlation == -1 or (threshold_correlation < 1 and (correlation >= threshold_correlation)):
                prediction = sae_prediction
                selection = 0
                number_sae_samples += 1
                fixed_threshold_selected = fixed_threshold_sae
                print("SAE selected")
            else:
                prediction = dann_prediction
                selection = 1
                number_dann_samples += 1
                fixed_threshold_selected = fixed_threshold_dann
                print("DANN selected")

            if pseudo_fscore_sae == pseudo_fscore_dann:
                prediction_ideal = prediction
                selection_ideal = selection
                fixed_threshold_ideal_selected = fixed_threshold_selected

                if (selection == 0):
                    number_sae_samples_ideal += 1
                else:
                    number_dann_samples_ideal += 1

                print("Selected the same that normal (ideal but SAE == DANN)")
            elif pseudo_fscore_sae > pseudo_fscore_dann:
                prediction_ideal = sae_prediction
                selection_ideal = 0
                number_sae_samples_ideal += 1
                fixed_threshold_ideal_selected = fixed_threshold_sae
                print("SAE selected (ideal)")
            else: 
                prediction_ideal = dann_prediction
                selection_ideal = 1
                number_dann_samples_ideal += 1
                fixed_threshold_ideal_selected = fixed_threshold_dann
                print("DANN selected (ideal)")

            if selection_ideal == 0: #GT = SAE
                if selection == 0: #Pred = SAE
                    tn_selected += 1
                else: #Pred = DANN
                    fp_selected += 1
            elif selection_ideal == 1: #GT = DANN
                if selection == 0: #Pred = SAE
                    fn_selected += 1
                else: #Pred = DANN
                    tp_selected += 1

            prediction = prediction[:,2:prediction.shape[1]-2,2:prediction.shape[2]-2,:]
            prediction_ideal = prediction_ideal[:,2:prediction_ideal.shape[1]-2,2:prediction_ideal.shape[2]-2,:]

            sample_prediction = prediction[0].reshape(windows_shape[0]-4, windows_shape[1]-4)
            sample_prediction_ideal = prediction_ideal[0].reshape(windows_shape[0]-4, windows_shape[1]-4)

            finalImg[row+2:(row + windows_shape[0]-2), col+2:(col + windows_shape[1]-2)] = sample_prediction
            finalImg_bin[row+2:(row + windows_shape[0]-2), col+2:(col + windows_shape[1]-2)] = sample_prediction > fixed_threshold_selected

            finalImg_selection[row+2:(row + windows_shape[0]-2), col+2:(col + windows_shape[1]-2)] = selection

            finalImg_ideal[row+2:(row + windows_shape[0]-2), col+2:(col + windows_shape[1]-2)] = sample_prediction_ideal
            finalImg_ideal_bin[row+2:(row + windows_shape[0]-2), col+2:(col + windows_shape[1]-2)] = sample_prediction_ideal > fixed_threshold_ideal_selected
            finalImg_selection_ideal[row+2:(row + windows_shape[0]-2), col+2:(col + windows_shape[1]-2)] = selection_ideal


    print("Counting number of selected samples (SAE or DANN)")
    print ("SAE: " + str(number_sae_samples))
    print ("DANN: " + str(number_dann_samples))
    print ('*'*80)
    print("Counting number of selected samples (SAE or DANN) IDEAL")
    print ("SAE: " + str(number_sae_samples_ideal))
    print ("DANN: " + str(number_dann_samples_ideal))
    print ('*'*80)
    print ("Summary selection (Positive = DANN, Negative = SAE):")
    print ("TP:"+ str(tp_selected) + ";TN:" + str(tn_selected) + ";FP:" + str(fp_selected) + ";FN:" + str(fn_selected))
    print ('*'*80)

    return finalImg, finalImg_bin, finalImg_ideal, finalImg_ideal_bin, finalImg_selection, finalImg_selection_ideal

def predictCompleteImage(img, windows_shape, batch_size, model):

    finalImg = np.zeros(img.shape, dtype=float)
    min_row = 0
    min_col = 0
    max_row = img.shape[0] - windows_shape[0]
    max_col = img.shape[1] - windows_shape[1]
    
    overlapping = 0.2
    row_overlapping = int(windows_shape[0] * overlapping)
    col_overlapping = int(windows_shape[1] * overlapping)

    print("Predicting chunks and generating image for histogram generation...")

    for row in range(min_row, max_row, windows_shape[0]-row_overlapping):
        for col in range(min_col, max_col, windows_shape[1] - col_overlapping):
            
            row = min(row, max_row)
            col = min(col, max_col)
            gr_sample = img[row:row+windows_shape[0], col:col+windows_shape[1]]
            
            roi = gr_sample.reshape(1, windows_shape[0], windows_shape[1], 1)
            roi = roi.astype('float32')

            prediction = model.predict(roi)
            prediction = prediction[:,2:prediction.shape[1]-2,2:prediction.shape[2]-2,:]

            sample_prediction = prediction[0].reshape(windows_shape[0]-4, windows_shape[1]-4)
            finalImg[row+2:(row + windows_shape[0]-2), col+2:(col + windows_shape[1]-2)] = sample_prediction

    return finalImg


def getHistogramBins(sample_image, num_decimal):
    tuple_sample = tuple(sample_image.reshape(1,-1)[0])

    if num_decimal is not None:
        tuple_sample_round = []
        for num in tuple_sample:
            if num > 0.01:
                tuple_sample_round.append(round(float(num), num_decimal))
            
        tuple_sample = tuple_sample_round
        precision = getPrecision(num_decimal)
        
        value = 0.
        value = round(value, num_decimal)
        for i in range(int(1/precision)+1):
            value = round(i*precision, num_decimal)
            tuple_sample.append(value)
            
    histogram_prediction = Counter(tuple_sample)

    return histogram_prediction

def getHistogramDomainListFolders(model, list_files, config, num_decimal=None):
    
    list_files_all = [item for sublist in list_files for item in sublist]

    histogram_domain, histogram_files =  getHistogramDomain(list_files_all, model, config, num_decimal)
    
    print('*'*80)
    print("Histogram per file")
    print('*'*80)
    for key in histogram_files:
        print(key)
        print(str(histogram_files[key]).replace(" ", "").replace(",",";").replace("[", "").replace("]", ""))

    return histogram_domain, histogram_files


def normalizeHistogram(histogram):
    number_pixels = sum(histogram)
    normalized_histogram = [histogram[number] / float(number_pixels) for number in range(len(histogram))]
    if sum(normalized_histogram) != 1.0:
        normalized_histogram[0] -= sum(normalized_histogram) - 1.0

    return normalized_histogram

def getHistogramDomain(array_files, model, config, num_decimal=None):

    histogram_domain = None
    histogram_files = {}

    for fname in array_files:
        print('Processing image', fname)

        #fname_gt = fname.replace(utilConst.X_SUFIX, utilConst.Y_SUFIX)
        img = 1.0 - (cv2.imread(fname, cv2.IMREAD_GRAYSCALE) / 255.)

        if img.shape[0] < config.window or img.shape[1] < config.window:
            new_rows = config.window if img.shape[0] < config.window else img.shape[0]
            new_cols = config.window if img.shape[1] < config.window else img.shape[1]
            img = cv2.resize(img, (new_cols, new_rows), interpolation = cv2.INTER_CUBIC)
            
        finalImg = predictCompleteImage(img, (config.window, config.window), config.batch, model)
        #saveImage(finalImg*255, "prueba.png")

        histogram_domain_fname = getHistogramBins(finalImg, num_decimal)
        print(str(histogram_domain_fname))

        if histogram_domain is None:
            histogram_domain = histogram_domain_fname.copy()
        else:
            histogram_domain = histogram_domain + histogram_domain_fname

        items_histogram = sorted(histogram_domain_fname.items())
        list_values_histogram = []
        str_prob = ""
        str_value = ""
        for prob, value in items_histogram:
            str_prob += str(prob) + "\t"
            str_value += str(value-1) + "\t"
            list_values_histogram.append(value)

        print(str_prob)
        print(str_value)
        histogram_files[fname] = list_values_histogram
        print(str(histogram_domain))

    return histogram_domain, histogram_files


def evaluateAutoModel(
                    db_name_source,
                    db_name_target,
                    db_name_test,
                    model_sae, 
                    model_dann, 
                    list_files, 
                    windows_shape, 
                    batch_size, 
                    verbose, 
                    norm_histogram_source,
                    norm_histogram_target,
                    threshold_correlation=0.25, 
                    fixed_threshold_sae=None, 
                    fixed_threshold_dann=None):
    windows_shape_3D = (windows_shape[0], windows_shape[1], 1)

    prediction_imgs_autodann_bin = []
    prediction_imgs_autodann_ideal_bin = []
    prediction_imgs_autodann = []
    prediction_imgs_autodann_ideal = []
    gt_imgs = []
    out_fnames_autodann = []
    out_fnames_autodann_ideal = []
    for fname_gr in list_files:
        fname_gt = fname_gr.replace(utilConst.X_SUFIX, utilConst.Y_SUFIX)
        gr_img = 1.0 - (cv2.imread(fname_gr, cv2.IMREAD_GRAYSCALE) / 255.)
        gt_img = 1 - (cv2.imread(fname_gt, cv2.IMREAD_GRAYSCALE) > 128)
        min_row = 0
        min_col = 0
        max_row = gr_img.shape[0] - windows_shape[0]
        max_col = gr_img.shape[1] - windows_shape[1]
        
        print("Predicting chunks and generating image...")
        prediction_img_autodann, prediction_img_autodann_bin, prediction_img_autodann_ideal, prediction_img_autodann_ideal_bin, prediction_img_autodann_selection, prediction_img_autodann_selection_ideal = predictCompleteImage_AutoDANN(
                                            gr_img, 
                                            gt_img,
                                            windows_shape, 
                                            batch_size, 
                                            model_sae, 
                                            model_dann, 
                                            norm_histogram_source,
                                            norm_histogram_target,
                                            threshold_correlation,
                                            fixed_threshold_sae, 
                                            fixed_threshold_dann)

        out_fname = "output" + "/" + db_name_source[0] + "--" + db_name_target[0] + "/" + "test-" + db_name_test[0] + "/" + "corr-" + str(threshold_correlation) + "/" + fname_gr
        dir_output = os.path.dirname(out_fname)
        mkdirp(dir_output)

        print(fname_gr + "")
        min_prob_pred_autodann = np.amin(prediction_img_autodann)
        max_prob_pred_autodann = np.amax(prediction_img_autodann)

        print("Range of probabilities (AUTODANN): " + str(min_prob_pred_autodann) + "-" + str(max_prob_pred_autodann))
        
        saveImage(prediction_img_autodann*255, out_fname.replace("png", "_autodann_probs.png"))
        saveImage(prediction_img_autodann_ideal*255, out_fname.replace("png", "_autodann_ideal_probs.png"))
        saveImage(prediction_img_autodann_bin*255, out_fname.replace("png", "_autodann_bin.png"))
        saveImage(prediction_img_autodann_ideal_bin*255, out_fname.replace("png", "_autodann_ideal_bin.png"))
        saveImage(prediction_img_autodann_selection*255, out_fname.replace("png", "_selection.png"))
        saveImage(prediction_img_autodann_selection_ideal*255, out_fname.replace("png", "_selection_ideal.png"))

        prediction_imgs_autodann.append(np.copy(prediction_img_autodann))
        prediction_imgs_autodann_ideal.append(np.copy(prediction_img_autodann_ideal))

        prediction_imgs_autodann_bin.append(np.copy(prediction_img_autodann_bin))
        prediction_imgs_autodann_ideal_bin.append(np.copy(prediction_img_autodann_ideal_bin))


        gt_imgs.append(gt_img)
        out_fnames_autodann.append(out_fname.replace("png", "_autodann_th.png"))
        out_fnames_autodann_ideal.append(out_fname.replace("png", "_autodann_ideal_th.png"))
        
        out_gt_fname = out_fname.replace("png", "_gt.png")
        if os.path.isfile(out_gt_fname) == False:
            saveImage(gt_img*255, out_gt_fname)
        
        out_gr_fname = out_fname.replace("png", "_gr.png")
        if os.path.isfile(out_gr_fname) == False:
            saveImage(gr_img*255, out_gr_fname)

    pseudo_threshold_autodann_ideal=-1
    threshold_autodann_ideal = -1
    pseudo_threshold_autodann=-1
    threshold_autodann = -1

    fscore_autodann, precision_autodann, recall_autodann, pseudo_fscore_autodann, pseudo_precision_autodann, pseudo_recall_autodann = utilMetrics.evaluateMetrics(prediction_imgs_autodann_bin, gt_imgs, 1, False, None)
    fscore_autodann_ideal, precision_autodann_ideal, recall_autodann_ideal, pseudo_fscore_autodann_ideal, pseudo_precision_autodann_ideal, pseudo_recall_autodann_ideal = utilMetrics.evaluateMetrics(prediction_imgs_autodann_ideal_bin, gt_imgs, 1, False, None)

    results_autodann = Results(precision_autodann, recall_autodann, fscore_autodann, threshold_autodann, pseudo_precision_autodann, pseudo_recall_autodann, pseudo_fscore_autodann, pseudo_threshold_autodann)
    results_autodann_ideal = Results(precision_autodann_ideal, recall_autodann_ideal, fscore_autodann_ideal, threshold_autodann_ideal, pseudo_precision_autodann_ideal, pseudo_recall_autodann_ideal, pseudo_fscore_autodann_ideal, pseudo_threshold_autodann_ideal)
    
    print('-'*80)
    print ("Results AUTODANN:")
    print (results_autodann)
    print('-'*80)
    print ("Results DANN:")
    print (results_autodann_ideal)
    print('-'*80)
    
    return results_autodann, results_autodann_ideal
