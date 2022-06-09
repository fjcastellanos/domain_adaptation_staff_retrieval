# -*- coding: utf-8 -*-
from __future__ import print_function
import sys, os, warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Disable Tensorflow CUDA load statements
gpu = sys.argv[ sys.argv.index('-gpu') + 1 ] if '-gpu' in sys.argv else '0'
os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_VISIBLE_DEVICES']=gpu
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"

import tensorflow as tf 
tf.get_logger().setLevel('INFO')

#from tensorflow.python.framework.ops import disable_eager_execution
#disable_eager_execution()

tf.config.list_physical_devices('GPU')
tf.config.run_functions_eagerly(True)

warnings.filterwarnings('ignore')


import copy
import os
import argparse
import numpy as np
import util
import utilConst
import utilIO
import utilMetrics
#import utilDataGenerator
#import utilCNN
#import utilDANN
#import utilDANNModel
from keras import backend as K
from keras.models import load_model
from keras.optimizers import SGD,Adadelta,Adam
from sgd_agc import SGD_AGC
import cv2

from SAEModel import SAEModel
from SAEDANNModel import SAEDANNModel
from Results import Results

util.init()

K.set_image_data_format('channels_last')
if K.image_data_format() == 'channels_last':
    bn_axis = 3
else:
    bn_axis = 1

#if K.backend() == 'tensorflow':
#    import tensorflow as tf    # Memory control with Tensorflow
#    session_conf = tf.ConfigProto()
#    session_conf.gpu_options.allow_growth=True
#    sess = tf.Session(config=session_conf, graph=tf.get_default_graph())
#    K.set_session(sess)



def filterProbImages(list_files, filt):

    list_files_filtered = []
    for path_file in list_files:
        if filt in path_file:
            list_files_filtered.append(path_file)
        
    return list_files_filtered


if __name__ == "__main__":

    dbs = [
            ["b-59-850", "ein"],
            ["b-59-850", "sal-rec"],
            ["ein", "b-59-850"],
            ["sal-rec", "b-59-850"]
    ]

    for db in dbs:
        db1 = db[0]
        db2 = db[1]
    
        correlations = [
                        "-1.0",
                        "-0.9",
                        "-0.8",
                        "-0.7",
                        "-0.6",
                        "-0.5",
                        "-0.4",
                        "-0.3",
                        "-0.2",
                        "-0.1",
                        "0.0",
                        "0.1",
                        "0.2",
                        "0.3",
                        "0.4",
                        "0.5",
                        "0.6",
                        "0.7",
                        "0.8",
                        "0.9",
                        "0.95",
                        "0.96",
                        "0.97",
                        "0.98",
                        "0.99",
                        "1.0"
                        ]

        results_ink = ""
        results_staff = ""
        results_notes = ""
        results_text = ""
        
        for corr_th in correlations:
        

            dir_cap_ein_ink =   "output_copy/" + db1 + "-ink--" + db2 + "-ink/test-" + db2 + "-ink/corr-" + corr_th
            dir_cap_ein_staff = "output_copy/" + db1 + "-staff--" + db2 + "-staff/test-" + db2 + "-staff/corr-" + corr_th
            dir_cap_ein_notes = "output_copy/" + db1 + "-notes--" + db2 + "-notes/test-" + db2 + "-notes/corr-" + corr_th
            dir_cap_ein_text =  "output_copy/" + db1 + "-text--" + db2 + "-text/test-" + db2 + "-text/corr-" + corr_th


            path_files_cap_ein_ink_gt =    filterProbImages(util.listFilesRecursive(dir_cap_ein_ink),      "_gt.")
            path_files_cap_ein_staff_gt =  filterProbImages(util.listFilesRecursive(dir_cap_ein_staff),    "_gt.")
            path_files_cap_ein_notes_gt =  filterProbImages(util.listFilesRecursive(dir_cap_ein_notes),    "_gt.")
            path_files_cap_ein_text_gt =   filterProbImages(util.listFilesRecursive(dir_cap_ein_text),     "_gt.")
            

            path_files_cap_ein_ink =    filterProbImages(util.listFilesRecursive(dir_cap_ein_ink),      "autodann_probs.")
            path_files_cap_ein_staff =  filterProbImages(util.listFilesRecursive(dir_cap_ein_staff),    "autodann_probs.")
            path_files_cap_ein_notes =  filterProbImages(util.listFilesRecursive(dir_cap_ein_notes),    "autodann_probs.")
            path_files_cap_ein_text =   filterProbImages(util.listFilesRecursive(dir_cap_ein_text),     "autodann_probs.")
            
            path_files_cap_ein_ink_ideal =    filterProbImages(util.listFilesRecursive(dir_cap_ein_ink),      "autodann_ideal_probs.")
            path_files_cap_ein_staff_ideal =  filterProbImages(util.listFilesRecursive(dir_cap_ein_staff),    "autodann_ideal_probs.")
            path_files_cap_ein_notes_ideal =  filterProbImages(util.listFilesRecursive(dir_cap_ein_notes),    "autodann_ideal_probs.")
            path_files_cap_ein_text_ideal =   filterProbImages(util.listFilesRecursive(dir_cap_ein_text),     "autodann_ideal_probs.")
            

            assert(len(path_files_cap_ein_ink) == len(path_files_cap_ein_staff) == len(path_files_cap_ein_notes) == len(path_files_cap_ein_text))

            final_imgs_ink = []
            final_imgs_staff = []
            final_imgs_notes = []
            final_imgs_text = []

            ink_gts = []
            staff_gts = []
            notes_gts = []
            text_gts = []

            print("Number of test pages: " + str(len(path_files_cap_ein_ink)))
            for i in range(len(path_files_cap_ein_ink)):
                print(path_files_cap_ein_ink[i])
                print(path_files_cap_ein_staff[i])
                print(path_files_cap_ein_notes[i])
                print(path_files_cap_ein_text[i])
                print('*'*80)

                bg_probs = 1. - cv2.imread(path_files_cap_ein_ink[i], cv2.IMREAD_GRAYSCALE) / 255.
                staff_probs = cv2.imread(path_files_cap_ein_staff[i], cv2.IMREAD_GRAYSCALE) / 255.
                notes_probs = cv2.imread(path_files_cap_ein_notes[i], cv2.IMREAD_GRAYSCALE) / 255.
                text_probs = cv2.imread(path_files_cap_ein_text[i], cv2.IMREAD_GRAYSCALE) / 255.

                ink_gt = ((cv2.imread(path_files_cap_ein_ink_gt[i], cv2.IMREAD_GRAYSCALE) / 255.) > 0.5) *1 
                staff_gt = ((cv2.imread(path_files_cap_ein_staff_gt[i], cv2.IMREAD_GRAYSCALE) / 255.) > 0.5) *1
                notes_gt = ((cv2.imread(path_files_cap_ein_notes_gt[i], cv2.IMREAD_GRAYSCALE) / 255.) > 0.5) *1
                text_gt = ((cv2.imread(path_files_cap_ein_text_gt[i], cv2.IMREAD_GRAYSCALE) / 255.) > 0.5) *1

                print(bg_probs[300,300])
                print(staff_probs[300,300])
                print(notes_probs[300,300])
                print(text_probs[300,300])
                print (str(np.amin(bg_probs)) + "-" + str(np.amax(bg_probs)))
                print (str(np.amin(staff_probs)) + "-" + str(np.amax(staff_probs)))
                print (str(np.amin(notes_probs)) + "-" + str(np.amax(notes_probs)))
                print (str(np.amin(text_probs)) + "-" + str(np.amax(text_probs)))
                

                full_img = np.zeros((1, bg_probs.shape[0], bg_probs.shape[1], 4))
                full_img[0, :,:,0] = bg_probs
                full_img[0, :,:,1] = staff_probs
                full_img[0, :,:,2] = notes_probs
                full_img[0, :,:,3] = text_probs

                full_img_argmax = np.argmax(full_img, axis = 3)

                print(full_img_argmax[0,300,300])
                full_img_bg = 1 - (full_img_argmax == 0) * 1
                full_img_staff = (full_img_argmax == 1) * 1
                full_img_notes = (full_img_argmax == 2) * 1
                full_img_text = (full_img_argmax == 3) * 1

                out_directory = "output_combined/" + db1 + "_" + db2 + "/" + str(corr_th) + "/"
                util.mkdirp(out_directory)

                #util.saveImage(full_img_bg[0]*255, "prueba_bg.png")
                util.saveImage(full_img_staff[0]*255, out_directory  + str(i) +  "_staff.png")
                util.saveImage(full_img_notes[0]*255, out_directory  + str(i) +  "_notes.png")
                util.saveImage(full_img_text[0]*255, out_directory  + str(i) +  "_text.png")
                util.saveImage(full_img_bg[0]*255, out_directory  + str(i) +  "_ink.png")

                util.saveImage(staff_gt*255, out_directory + str(i) + "_staff_gt.png")
                util.saveImage(notes_gt*255, out_directory + str(i) + "_notes_gt.png")
                util.saveImage(text_gt*255, out_directory + str(i) + "_text_gt.png")
                util.saveImage(ink_gt*255, out_directory + str(i) + "_ink_gt.png")

                #util.saveImage(full_img_notes[0]*255, "prueba_notes.png")
                #util.saveImage(full_img_text[0]*255, "prueba_text.png")

                final_imgs_ink.append(full_img_bg[0])
                final_imgs_staff.append(full_img_staff[0])
                final_imgs_notes.append(full_img_notes[0])
                final_imgs_text.append(full_img_text[0])

                ink_gts.append(ink_gt)
                staff_gts.append(staff_gt)
                notes_gts.append(notes_gt)
                text_gts.append(text_gt)


            '''            
            print("Evaluating Ink")
            fscore_ink, precision_ink, recall_ink, pseudo_fscore_ink, pseudo_precision_ink, pseudo_recall_ink = utilMetrics.evaluateMetrics(final_imgs_ink, ink_gts, 1, False, None)
            print("Evaluating Staff")
            fscore_staff, precision_staff, recall_staff, pseudo_fscore_staff, pseudo_precision_staff, pseudo_recall_staff = utilMetrics.evaluateMetrics(final_imgs_staff, staff_gts, 1, False, None)
            print("Evaluating Notes")
            fscore_notes, precision_notes, recall_notes, pseudo_fscore_notes, pseudo_precision_notes, pseudo_recall_notes = utilMetrics.evaluateMetrics(final_imgs_notes, notes_gts, 1, False, None)
            print("Evaluating Text")
            fscore_text, precision_text, recall_text, pseudo_fscore_text, pseudo_precision_text, pseudo_recall_text = utilMetrics.evaluateMetrics(final_imgs_text, text_gts, 1, False, None)

            separator = ";"
            print(db1 + "->" + db2 + " (" + "staff, notes, text, ink" +")")
            print('*'*80)

            res_ink = corr_th + separator\
                    + str(precision_ink) + separator\
                    + str(recall_ink) + separator\
                    + str(fscore_ink) + separator\
                    + str(pseudo_precision_ink) + separator\
                    + str(pseudo_recall_ink) + separator\
                    + str(pseudo_fscore_ink) +\
                    "\n"
            
            results_ink += res_ink

            res_staff = corr_th + separator\
                    + str(precision_staff) + separator\
                    + str(recall_staff) + separator\
                    + str(fscore_staff) + separator\
                    + str(pseudo_precision_staff) + separator\
                    + str(pseudo_recall_staff) + separator\
                    + str(pseudo_fscore_staff) +\
                    "\n"

            results_staff += res_staff

            res_notes = corr_th + separator\
                    + str(precision_notes) + separator\
                    + str(recall_notes) + separator\
                    + str(fscore_notes) + separator\
                    + str(pseudo_precision_notes) + separator\
                    + str(pseudo_recall_notes) + separator\
                    + str(pseudo_fscore_notes) + \
                    "\n"

            results_notes += res_notes

            res_text = corr_th + separator\
                    + str(precision_text) + separator\
                    + str(recall_text) + separator\
                    + str(fscore_text) + separator\
                    + str(pseudo_precision_text) + separator\
                    + str(pseudo_recall_text) + separator\
                    + str(pseudo_fscore_text) + \
                    "\n"

            results_text += res_text

            print ("Results Staff")
            print(res_staff)
            print ("Results Notes")
            print(res_notes)
            print ("Results text")
            print(res_text)
            print ("Results Ink")
            print(res_ink)
            '''


        print ("----------------SUMMARY-----------------")
        print(db1 + "->" + db2)
        print ("Correlation threshold: " + str(corr_th))
        print ("STAFF")
        print ('*'*80)
        print(results_staff)
        print ("NOTES")
        print ('*'*80)
        print(results_notes)
        print ("TEXT")
        print ('*'*80)
        print(results_text)
        print ("INK")
        print ('*'*80)
        print(results_ink)