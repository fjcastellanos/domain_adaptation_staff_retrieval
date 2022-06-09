#!/usr/bin/env python
# -*- coding: utf-8 -*-
#==============================================================================

from __future__ import print_function
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator 
from random import randint
import util
import cv2
import random as rng
import itertools
from keras import backend as K
from keras.callbacks import Callback


rng.seed(12345)


EQUALIZATION_TYPE_NONE = "none"
EQUALIZATION_TYPE_WHITE_BALANCE = 'wb'



EQUALIZATION_TYPES = [EQUALIZATION_TYPE_NONE, EQUALIZATION_TYPE_WHITE_BALANCE]

#Apply white balance for a RGB image
def whiteBalanceForRGBImage(image, perc = 0.05):
    return np.dstack([whiteBalanceForChannelImage(channel, 0.05) for channel in cv2.split(image)] )

#Apply white balance for a channel from a RGB image
def whiteBalanceForChannelImage(channel, perc = 0.05):
    mi, ma = (np.percentile(channel, perc), np.percentile(channel,100.0-perc))
    channel = np.uint8(np.clip((channel-mi)*255.0/(ma-mi), 0, 255))
    return channel

#Apply equalization if it is selected in the program arguments.
def apply_equalization(img_x, type):
    if type == EQUALIZATION_TYPE_NONE:
        return img_x
    elif type == EQUALIZATION_TYPE_WHITE_BALANCE:
        img_x = whiteBalanceForRGBImage(img_x)
    else:
        raise Exception('Undefined equalization type: ' + type)

    return img_x

#Return the shape of a image with 3 values
def get3DShapeImage(image):
    im_shape = image.shape

    if len(im_shape) == 2:
        return (im_shape[0], im_shape[1], 1)
    else:
        return im_shape

#Return the list of images within a folder and split them into training, test and validation partitions
def lst_pathfiles(path_dir, num_test_pages, num_val_pages, considered_classes):
 
    training_dictionary, test_dictionary = util.json_muret_to_partitions_dicionary(path_dir, num_test_pages, considered_classes)

    training_list = list(training_dictionary.items())
    assert(num_val_pages < len(training_list))

    val_list = training_list[0:num_val_pages]
    training_list = training_list[num_val_pages:len(training_list)]

    val_dictionary = dict(val_list)
    training_dictionary = dict(training_list)
    
    return [training_dictionary, test_dictionary, val_dictionary]


#Return the list of bounding boxes from a bounding box dictionary
def getListBoundingBoxes(gt_data_document, considered_classes):
    list_bbox = []

    for region in gt_data_document["regions"]:
        class_name =  region["class"]

        if considered_classes is not None:
            if class_name not in considered_classes:
                continue

        xmin = region["xmin"]
        ymin = region["ymin"]

        xmax = region["xmax"]
        ymax = region["ymax"]

        list_bbox.append((ymin,xmin,ymax,xmax))

    return list_bbox

#Build an image from a ground truth dictionary with regions
def buildGTImage(gt_data_document,region_height_reduction=None, considered_classes=None):
    height = gt_data_document["height"]
    width = gt_data_document["width"]
    gt_image = np.zeros((height,width),dtype="uint8")
    
    for region in gt_data_document["regions"]:
        class_name =  region["class"]

        if considered_classes is not None:
            if class_name not in considered_classes:
                continue

        xmin = region["xmin"]
        ymin = region["ymin"]
        xmax = region["xmax"]
        ymax = region["ymax"]

        vertical_original_size = ymax-ymin
        
        if (region_height_reduction is not None):
            height_reduction_px_side = int(vertical_original_size * region_height_reduction)
            ymin += height_reduction_px_side
            ymax -= height_reduction_px_side
            
        gt_image[ymin:ymax, xmin:xmax] = 1

    return gt_image

#Obtain the images and their region labeled version
#Also, it applies normalization, equalization and data augmentation if these options are selected.
def getRegionsDataSet(
                        dict_documents, 
                        path_dir, 
                        with_color, 
                        considered_classes,
                        equalization_mode,
                        with_data_augmentation,
                        block_size,
                        region_height_reduction=None):
    X = []
    Y = []

    for key_document, gt_data_document in dict_documents.items():
        
        src_filename = gt_data_document["src_path"]
        src_im = util.loadImage(src_filename, with_color)
        src_im = apply_equalization(src_im, equalization_mode)

        gt_im = buildGTImage(gt_data_document, region_height_reduction, considered_classes)
        gt_im = np.uint8(gt_im > 0)
        
        assert(np.min(gt_im) >= 0 and np.max(gt_im) <=1)

        #print ("Rescaling document to the sample size: " + src_filename)
        src_im = util.redimImage(src_im, block_size[0], block_size[1])
        gt_im = util.redimImage(gt_im, block_size[0], block_size[1])
        X_doc =[src_im]
        Y_doc = [gt_im]
            
        assert(len(X_doc) == len(Y_doc))

        for idx in range(len(X_doc)):
            X.append(X_doc[idx])
            Y.append(Y_doc[idx])

    X = np.asarray(X).reshape(len(X), block_size[0], block_size[1], block_size[2])
    Y = np.asarray(Y).reshape(len(Y), block_size[0], block_size[1], 1)

    assert(len(X) > 0)
    if with_data_augmentation:
        [X, Y] = applyDataAugmentation(X=X, Y=Y, return_generator=False)

    return [X, Y]


#Obtain the training, test and validation partitions from a folder containing images.
def getRegionSamplesFromParams_1DB( 
                    path_dir,
                    block_size,
                    with_color, 
                    considered_classes,
                    equalization_mode,
                    with_data_augmentation,
                    region_height_reduction):
    assert(type(path_dir) is str)
    assert(type(considered_classes) is list)

    [training_docs, test_docs, validation_docs] = lst_pathfiles(
                    path_dir=path_dir,
                    num_test_pages=5,
                    num_val_pages=5,
                    considered_classes=considered_classes)

    [X_train, Y_train] = getRegionsDataSet(
                    dict_documents=training_docs,
                    path_dir=path_dir,
                    with_color=with_color, 
                    considered_classes=considered_classes,
                    equalization_mode=equalization_mode, 
                    with_data_augmentation=with_data_augmentation, 
                    block_size=block_size,
                    region_height_reduction=region_height_reduction)

    [X_val, Y_val] = getRegionsDataSet(
                    dict_documents=validation_docs,
                    path_dir=path_dir,
                    with_color=with_color, 
                    considered_classes=considered_classes,
                    equalization_mode=equalization_mode, 
                    with_data_augmentation=False, 
                    block_size=block_size,
                    region_height_reduction=region_height_reduction)

    [X_test, Y_test] = getRegionsDataSet(
                    dict_documents=test_docs, 
                    path_dir=path_dir,
                    with_color=with_color, 
                    considered_classes=considered_classes,
                    equalization_mode=equalization_mode, 
                    with_data_augmentation=False, 
                    block_size=block_size)

    return [
                X_train, Y_train,
                X_val, Y_val,
                X_test, Y_test
            ]
#Apply data augmentation
def applyDataAugmentation(
                    X, Y, 
                    rotation_range=8, 
                    shear_range=0, 
                    vertical_flip=False, 
                    horizontal_flip=True,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    zoom_range=0.1,
                    fill_mode='nearest',
                    cval=0,
                    batch_size=32,
                    save_images=False, pathfile_saved_images=None,
                    return_generator=False):
    data_gen_args = dict(
                    rotation_range=rotation_range,              #3   10,           # Int. Degree range for random rotations.
                    shear_range=shear_range,                    # Float. Shear Intensity (Shear angle in counter-clockwise direction in degrees)
                    vertical_flip=vertical_flip,                # Boolean. Randomly flip inputs vertically.
                    horizontal_flip=horizontal_flip,
                    width_shift_range=width_shift_range,        #0.08, 0.09
                    height_shift_range=height_shift_range,      #0.08,  0.09
                    zoom_range=zoom_range,                      #0.08, 0.09                # Float or [lower, upper]. Range for random zoom. If a float,
                    fill_mode=fill_mode,
                    cval=cval
                    #brightness_range=[0.2, 1.4]  # Tuple or list of two floats. Range for picking a brightness shift value from.
                    )

    data_gen_seg_args = dict(
                    rotation_range=rotation_range,              #3   10,           # Int. Degree range for random rotations.
                    shear_range=shear_range,                    # Float. Shear Intensity (Shear angle in counter-clockwise direction in degrees)
                    vertical_flip=vertical_flip,                # Boolean. Randomly flip inputs vertically.
                    horizontal_flip=horizontal_flip,
                    width_shift_range=width_shift_range,        #0.08, 0.09
                    height_shift_range=height_shift_range,      #0.08,  0.09
                    zoom_range=zoom_range,                      #0.08, 0.09                # Float or [lower, upper]. Range for random zoom. If a float,
                    fill_mode='constant',
                    cval=255
                    #brightness_range=[0.2, 1.4]  # Tuple or list of two floats. Range for picking a brightness shift value from.
                    )

    num_samples = len(X)
    image_datagen = ImageDataGenerator(**data_gen_args)
    seg_datagen = ImageDataGenerator(**data_gen_seg_args)

    save_params1 = dict()
    save_params2 = dict()
    if save_images:
        assert(pathfile_saved_images is not None)
        out_x = pathfile_saved_images + "_src"
        out_s = pathfile_saved_images + "_gt"
        save_params1 = dict(save_to_dir=out_x, save_prefix='IM_plant', save_format='png')
        save_params2 = dict(save_to_dir=out_s, save_prefix='seg', save_format='png')

    image_generator = image_datagen.flow(X, batch_size=batch_size, seed=1, **save_params1)
    seg_generator = seg_datagen.flow(Y, batch_size=batch_size, seed=1, **save_params2)

    def combine_generator(gen1, gen2):
        while True:
            yield(gen1.next(), gen2.next())

    if return_generator:
        return combine_generator(image_generator, seg_generator)
    else:
        X = []
        Y = []
        count = 0
        for X_new, Y_new in combine_generator(image_generator, seg_generator):

            X_new = X_new.astype(np.uint8)
            Y_new = Y_new.astype(np.uint8)
            if (count == 0):
                X = X_new
                Y = Y_new
            else:
                X = np.concatenate((X, X_new))
                Y = np.concatenate((Y, Y_new))
            
            count = count + len(Y_new) 
            if (count > num_samples):
                break

        return [X, Y]


# Return the list of bounding boxes of a image calculating the contours.
def getBoundingBoxes(image, val=100):
    threshold = val
    minContourSize= int(image.shape[0]*image.shape[1]*0.0025)
    
    img = np.copy(image)
    ROWS = img.shape[0]
    COLS = img.shape[1]

    for j in range(COLS):
        img[0, j] = 0
        img[1, j] = 0
        img[2, j] = 0

        img[ROWS-1, j] = 0
        img[ROWS-2, j] = 0
        img[ROWS-3, j] = 0
    
    for i in range(ROWS):
        img[i, 0] = 0
        img[i, 1] = 0
        img[i, 2] = 0

        img[i, COLS-1] = 0
        img[i, COLS-2] = 0
        img[i, COLS-3] = 0
    
    
    
    #minContourSize = 500
    im = np.uint8(img)
    canny_output = cv2.Canny(im, threshold, threshold * 2)
    
    contours, herarchy = cv2.findContours(canny_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # create hull array for convex hull points
    hull = []
    boundRect = []

    # calculate points for each contour
    for i in range(len(contours)):

        contour_i = contours[i]
        contour_poly = cv2.approxPolyDP(contour_i, 3, True)
        hull_i = cv2.convexHull(contour_poly, False)

        area = cv2.contourArea(hull_i)

        if (area > minContourSize):
            bbox_i = cv2.boundingRect(hull_i)

            rect_by_corners = (bbox_i[1], bbox_i[0], bbox_i[1]+bbox_i[3], bbox_i[0]+bbox_i[2])

            if (bbox_i[3] > bbox_i[2]):# If tt is a vertical region, we ignore it.
                continue

            if (bbox_i[3] / bbox_i[2] < 0.05):# If width and height are so much different, we ignore the region
                continue 
             
            boundRect.append(rect_by_corners)

        #if (cv.contourArea(hull_i) > 0):
            

    return boundRect



# Code for testing

def main():

    inputDir = "./datasets/Liedblatter-Ciclo1/BoundingBoxes"
    


if __name__ == '__main__':
    main()