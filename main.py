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
import util_LA
import utilConst
import utilIO
import utilMetrics
import FscoreRegion
import cv2
#import utilDataGenerator
#import utilCNN
#import utilDANN
#import utilDANNModel
from keras import backend as K
from keras.models import load_model
from keras.optimizers import SGD,Adadelta,Adam
from sgd_agc import SGD_AGC

from SAEModel import SAEModel
from SAEDANNModel import SAEDANNModel
from Results import Results
from file_manager import FileManager

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





def menu():
    parser = argparse.ArgumentParser(description='DA SAE')
    parser.add_argument('-type',   default='dann', type=str,     choices=['dann', 'cnn', 'autodann'],  help='Training type')

    #parser.add_argument('-path',  required=True,   help='base path to datasets')
    #parser.add_argument('-s',          default=-1,      dest='step',                type=int,   help='step size. -1 to use window size')
    #parser.add_argument('-samples',    default=None,        dest='samples',     type=int,   help='Number of total samples to extract from each dataset')
    #parser.add_argument('-db_add',    required=False,  nargs='+', choices=utilConst.ARRAY_DBS, help='Database name additional validations')
    

    parser.add_argument('-opt',   required=True,    choices=utilConst.ARRAY_OPTIMIZERS, help='Optimizer to be applied')

    parser.add_argument('-db1_train',       required=True,  nargs='+', help='Database path source')
    parser.add_argument('-db2_train',       required=True,  nargs='+', help='Database path target')
    
    
    parser.add_argument('-db1_name',       required=True,  nargs='+', help='Database name source')
    parser.add_argument('-db2_name',       required=True,  nargs='+', help='Database name target')

    parser.add_argument('-db1_val',       required=True,  nargs='+', help='Database name source')
    parser.add_argument('-db2_val',       required=True,  nargs='+', help='Database name target')
    
    parser.add_argument('-db1_test',       required=False,  nargs='+', help='Database name source')
    parser.add_argument('-db2_test',       required=False,  nargs='+', help='Database name target')
    
    parser.add_argument('-classes',       required=True, dest='considered_classes', nargs='+', help='Database name target')
    
    

    parser.add_argument('--aug',   action='store_true', help='Load augmentation folders')
    parser.add_argument('-w',          default=256,    dest='window',           type=int,   help='window size')
    
    parser.add_argument('-gpos',          default=0,        dest='grl_position',     type=int,   help='Position of GRL')

    
    parser.add_argument('-l',          default=4,        dest='nb_layers',     type=int,   help='Number of layers')
    parser.add_argument('-f',          default=64,      dest='nb_filters',   type=int,   help='nb_filters')
    parser.add_argument('-k',          default=5,        dest='k_size',            type=int,   help='kernel size')
    parser.add_argument('-drop',   default=0,        dest='dropout',          type=float, help='dropout value')

    parser.add_argument('-lda',      default=0.01,    type=float,    help='Reversal gradient lambda')
    parser.add_argument('-lda_inc',  default=0.001,    type=float,    help='Reversal gradient lambda increment per epoch')
    #parser.add_argument('-page',   default=-1,      type=int,   help='Nb pages to divide the training set. -1 to load all')
    #parser.add_argument('-super',  default=1,      dest='nb_super_epoch',      type=int,   help='nb_super_epoch')
    parser.add_argument('-th',         default=-1,     dest='threshold',           type=float, help='threshold. -1 to test from 0 to 1')
    parser.add_argument('-th_iou',         default=0.55,     dest='th_iou',           type=float, help='threshold. -1 to test from 0 to 1 for IoU')

    parser.add_argument('-e',           default=200,    dest='epochs',            type=int,   help='nb_epoch')
    parser.add_argument('-se',           default=1,    dest='super_epochs',            type=int,   help='Number of super epochs')
    parser.add_argument('-pre',           default=50,    dest='epochs_pretrain',            type=int,   help='Number of epochs of pretrain')
    
    parser.add_argument('-b',           default=12,     dest='batch',               type=int,   help='batch size')
    parser.add_argument('-verbose',     default=0,                                  type=int,   help='1=show batch increment, other=mute')

    parser.add_argument('--truncate',   action='store_true', help='Truncate data')
    parser.add_argument('--test',   action='store_true', help='Only run test')
    #parser.add_argument('--show',   action='store_true', help='Show the result')
    parser.add_argument('--tboard',   action='store_true', help='Active tensorboard')
    parser.add_argument('--save',   action='store_true', help='Save binarized output images')
    parser.add_argument('-filter',  default= None, help='Filter the samples for training', choices=utilConst.ARRAY_FILTERS)
    parser.add_argument('--bn',   action='store_true', help='Apply batch normalization')
    parser.add_argument('--clip',   action='store_true', help='Apply adaptative clipping')
    parser.add_argument('--clipcenter',   action='store_true', help='Apply adaptative clipping')

    parser.add_argument('--dom_stop',   action='store_true', help='Activate the domain stopping criteria. In training, it saves the model when the domain classifier is worst.')
    
    #parser.add_argument('-loadmodel', type=str,   help='Weights filename to load for test')

    parser.add_argument('-gpu',    default='0',    type=str,   help='GPU')

    args = parser.parse_args()

    print('CONFIG:\n -', str(args).replace('Namespace(','').replace(')','').replace(', ', '\n - '))

    return args


def splitValidationAndTrainingPartitions(list_files, ratio_training=0.8):

    list_files_train = []
    list_files_val = []


    for list_files_i in list_files:

        num_training_files = min(len(list_files_i)-1, int(len(list_files_i)*ratio_training))
        num_validation_files = len(list_files_i) - num_training_files

        print("Training files: " + str(num_training_files))
        print("Validation files: " + str(num_validation_files))

        list_files_train.append(list_files_i[0:num_training_files])
        list_files_val.append(list_files_i[num_training_files:])

    return list_files_train, list_files_val
    

#Return the list of bounding boxes detected in the prediction image.
def getBoundingBoxesFromPrediction(prediction_image, vertical_reduction_regions=0., margin=0):
    val_bbox = 127
    bboxes = util_LA.getBoundingBoxes(prediction_image, val=val_bbox)

    pred_shape = prediction_image.shape

    for idx in range(len(bboxes)):
        
        x_start = bboxes[idx][0]
        y_start = bboxes[idx][1]
        x_end = bboxes[idx][2]
        y_end = bboxes[idx][3]
        if vertical_reduction_regions > 0.:
            vertical_region_size = x_end - x_start
            gt_prediction_vertical_size = int(vertical_region_size // (1-vertical_reduction_regions))
            vertical_reduction_region_side = int((gt_prediction_vertical_size - vertical_region_size) // 2)
            bboxes[idx] = (
                        max(0,x_start-int(vertical_reduction_region_side)), 
                        max(0,y_start-margin), 
                        min(pred_shape[0], x_end + int(vertical_reduction_region_side)), 
                        min(y_end + margin, pred_shape[1]))
            
    return bboxes
    
def getBoundingBoxes(config, model, list_images, list_jsons):

    dict_bboxes = {}
    for index_selected in range(len(list_images)):
        fname_gr = list_images[index_selected]
        fname_gt = list_jsons[index_selected]

        print(str(index_selected) + " - " + fname_gr)

        gr_img, gt_img, gt_bboxes = util.loadImageAndGTData(fname_gr, fname_gt, (config.window, config.window), config.considered_classes, 0)
        list_gr_imgs = []
        list_gr_imgs.append(gr_img)
        list_gr_arr = np.asarray(list_gr_imgs)

        prediction = model.predict(list_gr_arr)[0,:,:,0]

        prediction_th = (prediction>0.5)*255
        gr_img_orig = cv2.imread(fname_gr, cv2.IMREAD_GRAYSCALE)

        prediction_th_resized = util.resizeImage(prediction_th, gr_img_orig.shape[0], gr_img_orig.shape[1])
        prediction_th_resized = (prediction_th_resized>0.5)*255

        bboxes = getBoundingBoxesFromPrediction(prediction_th_resized, vertical_reduction_regions=0.2, margin=0)
        
        bboxes_pred_img = np.zeros((prediction_th_resized.shape[0], prediction_th_resized.shape[1]))
        for bbox in bboxes:
            bboxes_pred_img[bbox[0]:bbox[2], bbox[1]:bbox[3]] = 1

        bboxes_gt_img = np.zeros((prediction_th_resized.shape[0], prediction_th_resized.shape[1]))
        list_bboxes = getListBboxes(gt_bboxes)
        for bbox in list_bboxes:
            bboxes_gt_img[bbox[0]:bbox[2], bbox[1]:bbox[3]] = 1
            
        #print(bboxes)
        #print(prediction_th_resized.shape)
        FileManager.saveImageFullPath(prediction_th_resized, "prueba.png")
        FileManager.saveImageFullPath(bboxes_pred_img*255, "prueba_bbox.png")
        FileManager.saveImageFullPath(bboxes_gt_img*255, "prueba_gt_bbox.png")

        dict_bboxes[fname_gr] = {}
        dict_bboxes[fname_gr]["PRED"] = bboxes
        dict_bboxes[fname_gr]["GT"] = gt_bboxes

    return dict_bboxes


def getListBboxes(dict_bboxes):
    list_bboxes = []
    for key_class in dict_bboxes:
        for bbox in dict_bboxes[key_class]:
            list_bboxes.append(bbox)

    return list_bboxes


def getFscoreRegions(dict_bboxes, considered_classes, th_iou=.55):

    fscore_avg = 0.
    precision_avg = 0.
    recall_avg = 0.
    tp_all = 0
    fn_all = 0
    fp_all = 0
    num_regions_all = 0
    avg_overlapping_area_avg = 0.
    avg_overlapping_area_tp = 0.

    number_pages = 0

    list_overlapping_areas_all = []

    for fname_gr in dict_bboxes:
        list_bboxes_gt = getListBboxes(dict_bboxes[fname_gr]["GT"])

        fscore, precision, recall, page_overlapping_area, tp, fn, fp, list_overlapping_area, num_regions, overlapping_area_tp = FscoreRegion.getFscoreRegions(list_bboxes_gt, dict_bboxes[fname_gr]["PRED"], th=th_iou)
        print(fname_gr)
        if num_regions is None or num_regions == 0:
            avg_page_iou = 0.
        else:
            avg_page_iou = page_overlapping_area/num_regions
        print("Fscore: " + str(fscore) + "  Prec: " +str(precision) + "  Rec:" + str(recall) + "  Avg IOU: " + str(avg_page_iou) + "  TP: " + str(tp) + "  FN: " + str(fn) + "  FP: " + str(fp) + "  NUM REGIONS: " + str(num_regions) + "  AVG_TP_IOU: " + str(overlapping_area_tp))


        if list_overlapping_area is not None:
            for item in list_overlapping_area:
                list_overlapping_areas_all.append(item)
            for item_FN in range(fn):
                list_overlapping_areas_all.append(0.)

        if fscore is not None:
            fscore_avg += fscore    
            precision_avg += precision
            recall_avg += recall
            tp_all += tp
            fn_all += fn
            fp_all += fp
            avg_overlapping_area_avg+=page_overlapping_area
            avg_overlapping_area_tp+=overlapping_area_tp
            number_pages+=1
            num_regions_all+=num_regions

    fscore_avg /= number_pages
    precision_avg /= number_pages
    recall_avg /= number_pages

    if (tp_all + fp_all) > 0:
        avg_overlapping_area_avg /= (tp_all + fp_all)
    else:
        avg_overlapping_area_avg = 0.
    if tp_all > 0:
        avg_overlapping_area_tp /= tp_all
    else:
        avg_overlapping_area_tp = 0.
    
    tuple_list_overlapping_areas_all = tuple(list_overlapping_areas_all)
    histogram_iou = util.getHistogramTuple(tuple_list_overlapping_areas_all, 2)

    print(histogram_iou)
    print('*'*80)
    print("AVERAGE RESULTS:")
    print("Fscore: " + str(fscore_avg).replace(".", ",") + "  Prec: " +str(precision_avg).replace(".", ",") + "  Rec: " + str(recall_avg).replace(".", ",") + "  Avg IOU: " + str(avg_overlapping_area_avg).replace(".", ",") + "  TP: " + str(tp_all) + "  FN: " + str(fn_all) + "  FP: " + str(fp_all) + "  NUM REGIONS: " + str(num_regions_all) + "  NUM_PAGES_WITH_REGIONS: " + str(number_pages) + " AVG_TP_IOU: " + str(avg_overlapping_area_tp).replace(".", ","))
    return fscore_avg, precision_avg, recall_avg, tp_all, fn_all, fp_all, num_regions_all



def getHistogramsPredictions(
            config, 
            model,
            list_images, list_jsons):

    dict_histograms = {}
    for index_selected in range(len(list_images)):
        fname_gr = list_images[index_selected]
        fname_gt = list_jsons[index_selected]
        fname_pred_out = "out/" + config.type + "/" + str(config.db1_name[0]).replace("/", "-") + "---" + str(config.db2_name[0]).replace("/", "-") +"/" + fname_gr
        FileManager.makeDirsIfNeeded(os.path.dirname(fname_pred_out))        
        
        print(str(index_selected) + " - " + fname_gr)

        gr_img, _, _ = util.loadImageAndGTData(fname_gr, fname_gt, (config.window, config.window), config.considered_classes, 0)
        list_gr_imgs = []
        list_gr_imgs.append(gr_img)
        list_gr_arr = np.asarray(list_gr_imgs)

        prediction = model.predict(list_gr_arr)[0,:,:,0]

        histogram = util.getHistogram(prediction, 1)
        dict_histograms[fname_gr] = histogram

        FileManager.saveImageFullPath(prediction*255, fname_pred_out)
        fname_pred_out

    global_histogram = None
    for fname_gr in dict_histograms:
        if global_histogram is None:
            global_histogram = dict_histograms[fname_gr]
        else:
            assert(len(global_histogram) == len(dict_histograms[fname_gr]))
            global_histogram = [global_histogram[idx] + dict_histograms[fname_gr][idx] for idx in range(len(global_histogram))]

    return dict_histograms, global_histogram

def testSingleModel(
            config, 
            model, 
            mode,
            list_train_files_db1_val, list_train_files_db1_val_json, 
            list_train_files_db2_val, list_train_files_db2_val_json, 
            list_test_files_db1, list_test_files_db1_json, 
            list_test_files_db2, list_test_files_db2_json
            ):

    #PARA HISTOGRAMAS
    #target_histograms, target_global_histogram = getHistogramsPredictions(config, model, list_test_files_db2, list_test_files_db2_json)
    #normalized_target_global_histogram = util.normalizeHistogram(target_global_histogram)
    #print (str(normalized_target_global_histogram).replace(",", "").replace(".", ","))

    #PARA EVALUACION DE METRICAS
    print ("Results in target (%s)" % str(config.db2_name))
    dict_bboxes_db2_test = getBoundingBoxes(config, model, list_test_files_db2, list_test_files_db2_json)
    fscore_avg_db2, precision_avg_db2, recall_avg_db2, tp_all_db2, fn_all_db2, fp_all_db2, num_regions_all_db2 = getFscoreRegions(dict_bboxes_db2_test, config.considered_classes, config.th_iou)


def testAutoDANNModel(
            config, 
            sae_model,
            dann_model, 
            list_train_files_db1_val, list_train_files_db1_val_json, 
            list_train_files_db2_val, list_train_files_db2_val_json, 
            list_test_files_db1, list_test_files_db1_json, 
            list_test_files_db2, list_test_files_db2_json
            ):

    num_decimal=1
    histogram_source, histogram_files_source = util.getHistogramDomainListFolders(sae_model, list_test_files_db1, config, num_decimal)
    #histogram_target, histogram_files_target = util.getHistogramDomainListFolders(sae_model, list_test_files_db2, config, num_decimal)
    
    norm_histogram_source = util.normalizeHistogram(histogram_source)
    #norm_histogram_target = util.normalizeHistogram(histogram_target)
    
    #VALIDATION
    print ("Results in source (validation)(%s)" % config.db1_name)
    results_source_val_sae = util.evaluateModelListFolders("val/DANN/" + config.db1_name[0] + "-" + config.db2_name[0] + "/" + config.db1_name[0], sae_model, list_train_files_db1_val, list_train_files_db1_val_json, (config.window, config.window), config.batch, False, config.considered_classes, 0.5)
    assert(len(results_source_val_sae) == 1)
    pseudo_threshold_source_sae = results_source_val_sae[0].getPseudoThreshold()
    
    results_source_val_dann = util.evaluateModelListFolders("val/DANN/" + config.db1_name[0] + "-" + config.db2_name[0] + "/" + config.db1_name[0], dann_model, list_train_files_db1_val, list_train_files_db1_val_json, (config.window, config.window), config.batch, False, config.considered_classes, 0.5)
    assert(len(results_source_val_dann) == 1)
    pseudo_threshold_source_dann = results_source_val_dann[0].getPseudoThreshold()
    
    correlation_thresholds = [
                                -1.0, 
                                -0.9, 
                                -0.8, 
                                -0.7, 
                                -0.6, 
                                -0.5, 
                                -0.4, 
                                -0.3, 
                                -0.2, 
                                -0.1, 
                                0.0,
                                0.1,
                                0.2,
                                0.3,
                                0.4,
                                0.5,
                                0.6,
                                0.7,
                                0.8,
                                0.9,
                                0.95,
                                0.96,
                                0.97,
                                0.98,
                                0.99,
                                1.0]
    
    list_results_source_test = []
    list_results_target_test = []
    list_results_target_test_ideal = []
    list_results_additional_test = []

    for correlation_threshold in correlation_thresholds:
        print ("Correlation threshold: " + str(correlation_threshold))
        #TEST
        #print ("Results in source (%s)" % str(config.db1_name))
        #results_source_test = util.evaluateAutoModelListFolders(
        #                                        config.db1_name,
        #                                        config.db2_name,
        #                                        config.db1_name,
        #                                        sae_model, 
        #                                        dann_model, 
        #                                        list_test_files_db1, 
        #                                        (config.window, config.window), 
        #                                        config.batch, 
        #                                        False, 
        #                                        norm_histogram_source,
        #                                        norm_histogram_target, 
        #                                        correlation_threshold,
        #                                        pseudo_threshold_source_sae, pseudo_threshold_source_dann)#

                                                
        print ("Results in target (%s)" % str(config.db2_name))
        results_target_test, results_target_test_ideal = util.evaluateAutoModelListFolders(
                                                config.db1_name,
                                                config.db2_name,
                                                config.db2_name,
                                                sae_model, 
                                                dann_model, 
                                                list_test_files_db2, 
                                                (config.window, config.window), 
                                                config.batch, 
                                                False, 
                                                norm_histogram_source,
                                                None,#norm_histogram_target,
                                                correlation_threshold,
                                                pseudo_threshold_source_sae, pseudo_threshold_source_dann)


        print('*'*80)
        print("Test")
        print ("Correlation threshold: " + str(correlation_threshold))
        print('*'*80)
        #print ("Source (%s)" % str(config.db1_name))
        #util.printResults(config.db1_name, results_source_test)
        #print('-'*80)

        print ("Target (%s)" % str(config.db2_name))
        util.printResults(config.db2_name, results_target_test)
        print('-'*80)
        print ("IDEAL sample level Target (%s)" % str(config.db2_name))
        util.printResults(config.db2_name, results_target_test_ideal)
        print('-'*80)

        #print ("Additional DB (%s)" % str(config.db_add))
        #util.printResults(config.db_add, results_additional_test)
        #print('-'*80)

        #list_results_source_test.append(results_source_test)
        list_results_target_test.append(results_target_test)
        list_results_target_test_ideal.append(results_target_test_ideal)

        #list_results_additional_test.append(results_additional_test)

    print('-'*80)
    print ("Summary:")
    print('-'*80)
    print("Validation to retrieve the best threshold...")
    print ("SAE Source (%s)" % str(config.db1_name))
    util.printResults(config.db1_name, results_source_val_sae)
    print('-'*80)
    print ("DANN Source (%s)" % str(config.db1_name))
    util.printResults(config.db1_name, results_source_val_dann)
    print('-'*80)
    print('')


    idx_correlation_th = 0
    for correlation_threshold in correlation_thresholds:

        print('*'*80)
        print("Test-Correlation threshold: " + str(correlation_threshold))
        print('*'*80)
        #print ("Source (%s)" % str(config.db1_name))
        #util.printResults(config.db1_name, list_results_source_test[idx_correlation_th])
        #print('-'*80)

        print ("Target (%s)" % str(config.db2_name))
        util.printResults(config.db2_name, list_results_target_test[idx_correlation_th])
        print('-'*80)
        print ("IDEAL sample level Target (%s)" % str(config.db2_name))
        util.printResults(config.db2_name, list_results_target_test_ideal[idx_correlation_th])
        print('-'*80)

        #print ("Additional DB (%s)" % str(config.db_add))
        #util.printResults(config.db_add, list_results_additional_test[idx_correlation_th])
        #print('-'*80)

        idx_correlation_th+=1


def getImage_Pathfiles(list_jsons_pathfiles):
    list_path_images = []
    for json_pathfile in list_jsons_pathfiles:
        list_path_images.append(json_pathfile.replace(".json", "").replace("/JSON/", "/SRC/"))

    return list_path_images


if __name__ == "__main__":
    config = menu()

    kernel_shape = (config.k_size, config.k_size)
    num_filters = config.nb_filters
    input_shape = (config.window, config.window, 1)
    num_blocks = config.nb_layers
    pool_size = 2
    with_batch_normalization = config.bn
    with_adaptative_clipping = config.clip
    with_adaptative_clipping_centered = config.clipcenter
    with_domain_stopping = config.dom_stop
    dropout = config.dropout


    if config.opt == utilConst.OPT_ADADELTA:
        optimizer = Adadelta(lr=0.01, rho=0.95, epsilon=1e-08, decay=0.0)
        optimizer_domain = Adadelta(lr=0.01, rho=0.95, epsilon=1e-08, decay=0.0)
    elif config.opt == utilConst.OPT_SGD:
        optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=False)
        if with_adaptative_clipping or with_adaptative_clipping_centered:
            optimizer_domain = SGD_AGC(params={}, lr=0.01, weight_decay=1e-6, momentum=0.9, nesterov=False)
        else:
            optimizer_domain = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=False)
    elif config.opt == utilConst.OPT_ADAM:
        optimizer = Adam(lr=0.01)
        optimizer_domain = Adam(lr=0.01, decay=1e-6, momentum=0.9, nesterov=False)

    sae_model = SAEModel(
                            kernel_shape, 
                            num_filters, 
                            input_shape, 
                            num_blocks, 
                            pool_size, 
                            with_batch_normalization, 
                            dropout,
                            optimizer, 
                            bn_axis,
                            config.considered_classes)
    dann_model = None
    if config.type == 'dann':
        sae_model = None
        dann_model = SAEDANNModel(
                            kernel_shape, 
                            num_filters, 
                            input_shape, 
                            num_blocks, 
                            pool_size, 
                            with_batch_normalization, with_adaptative_clipping, with_adaptative_clipping_centered,
                            dropout,
                            optimizer, 
                            optimizer_domain,
                            bn_axis, 
                            config.grl_position, config.lda, config.lda_inc,
                            config.considered_classes)
    elif config.type == 'autodann':
        sae_model = SAEModel(
                            kernel_shape, 
                            num_filters, 
                            input_shape, 
                            num_blocks, 
                            pool_size, 
                            with_batch_normalization, 
                            dropout,
                            optimizer, 
                            bn_axis,
                            config.considered_classes)

        dann_model = SAEDANNModel(
                            kernel_shape, 
                            num_filters, 
                            input_shape, 
                            num_blocks, 
                            pool_size, 
                            with_batch_normalization, with_adaptative_clipping, with_adaptative_clipping_centered,
                            dropout,
                            optimizer, 
                            optimizer_domain,
                            bn_axis, 
                            config.grl_position, config.lda, config.lda_inc,
                            config.considered_classes)

    list_files_db1_train_json = util.readListPathsWithinlistFiles(config.db1_train)
    list_files_db2_train_json = util.readListPathsWithinlistFiles(config.db2_train)

    list_files_db1_val_json = util.readListPathsWithinlistFiles(config.db1_val)
    list_files_db2_val_json = util.readListPathsWithinlistFiles(config.db2_val)

    
    list_files_db1_train = getImage_Pathfiles(list_files_db1_train_json)
    list_files_db2_train = getImage_Pathfiles(list_files_db2_train_json)
    
    list_files_db1_val = getImage_Pathfiles(list_files_db1_val_json)
    list_files_db2_val = getImage_Pathfiles(list_files_db2_val_json)

    if config.db1_test is not None:
        list_files_db1_test_json = util.readListPathsWithinlistFiles(config.db1_test)
        list_files_db2_test_json = util.readListPathsWithinlistFiles(config.db2_test)
        list_files_db1_test = getImage_Pathfiles(list_files_db1_test_json)
        list_files_db2_test = getImage_Pathfiles(list_files_db2_test_json)


    print ('*'*80)
    print ("DATABASE SOURCE:")
    print ("Train files")
    print (list_files_db1_train)
    print ("Validation files")
    print (list_files_db1_val)

    print ('*'*80)
    print ("DATABASE TARGET:")
    print ("Train files")
    print (list_files_db2_train)
    print ("Validation files")
    print (list_files_db2_val)
    
    db1_name_item = config.db1_name[0]
    db2_name_item = config.db2_name[0]

    assert(db1_name_item in config.db1_train[0])
    assert(db2_name_item in config.db2_train[0])
    
    fold_name = ""
    if "fold" in config.db1_train[0]:
        fold_name = config.db1_train[0][config.db1_train[0].find("fold"):config.db1_train[0].find("fold")+len("fold")+1]
    
    db1_name = []
    db1_name.append(db1_name_item)
    db2_name = []
    db2_name.append(db2_name_item)
    
    if (config.test):
        
        type_model = config.type
        
        config.type = 'cnn'
        #cnn_path_model = sae_model.getModelPath(fold_name, config.db1_name, config.batch)

        config.type = 'dann'
        if (dann_model is not None):
            dann_path_model = dann_model.getModelPath(fold_name, config.db1_name, config.db2_name, config.batch, config.epochs_pretrain, with_domain_stopping)

        config.type = type_model

        if config.type == 'cnn':
            path_model = sae_model.getModelPath(fold_name, config.db1_name, config.batch)
            sae_model = load_model(path_model)
        elif config.type == 'dann':
            path_model = dann_model.getModelPath(fold_name, config.db1_name, config.db2_name, config.batch, config.epochs_pretrain, with_domain_stopping)
            sae_model = load_model(path_model)
        else:
            assert(config.type == 'autodann')
            sae_path_model = sae_model.getModelPath(fold_name, config.db1_name, config.batch)
            dann_path_model = dann_model.getModelPath(fold_name, config.db1_name, config.db2_name, config.batch, config.epochs_pretrain, with_domain_stopping)
            sae_model = load_model(sae_path_model)
            dann_model = load_model(dann_path_model)
        
        if config.type != 'autodann':
            testSingleModel(
                    config, 
                    sae_model, 
                    type_model,
                    list_files_db1_val, list_files_db1_val_json, 
                    list_files_db2_val, list_files_db2_val_json, 
                    list_files_db1_test, list_files_db1_test_json, 
                    list_files_db2_test, list_files_db2_test_json)
        else:
            assert(config.type == 'autodann')
            testAutoDANNModel(
                    config, 
                    sae_model,
                    dann_model, 
                    list_files_db1_val, list_files_db1_val_json, 
                    list_files_db2_val, list_files_db2_val_json, 
                    list_files_db1_test, list_files_db1_test_json, 
                    list_files_db2_test, list_files_db2_test_json)
    else:
        if config.type == 'cnn':
            sae_model.train(
                        db1_name, db2_name, fold_name,
                        config.epochs, config.batch, config.filter, config.super_epochs, 
                        config.verbose,
                        config.considered_classes,
                        list_files_db1_train, list_files_db1_train_json,  
                        list_files_db1_val, list_files_db1_val_json,  
                        list_files_db2_val, list_files_db2_val_json)
        else:
            assert(config.type == 'dann')
            dann_model.train(
                        db1_name, db2_name, fold_name,
                        config.epochs, config.batch, config.filter, config.super_epochs, 
                        config.epochs_pretrain,
                        config.verbose,
                        config.considered_classes,
                        list_files_db1_train, list_files_db1_train_json,  
                        list_files_db1_val, list_files_db1_val_json,  
                        list_files_db2_train, list_files_db2_train_json,
                        list_files_db2_val, list_files_db2_val_json)
