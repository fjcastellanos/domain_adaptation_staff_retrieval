
import cv2 as cv
import numpy as np

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou
    
class OverlappingBBox:
    def __init__(self, gt_bbox, pred_bbox):
        assert(type(gt_bbox) is tuple)
        assert(type(pred_bbox) is tuple)
        assert(len(gt_bbox) == 4)
        assert(len(pred_bbox) == 4)
        assert(gt_bbox[0] < gt_bbox[2])
        assert(gt_bbox[1] < gt_bbox[3])
        #assert(pred_bbox[0] < pred_bbox[2])
        #assert(pred_bbox[1] < pred_bbox[3])
        self.gt_bbox = gt_bbox
        self.pred_bbox = pred_bbox
        self.overlapped_area = self.getOverlappingArea()

    
    

    def getOverlappingArea(self):
        return bb_intersection_over_union(self.gt_bbox, self.pred_bbox)


    def __str__(self):
        return "GT(" + str(self.gt_bbox) + ")" + "--PRED(" + str(self.pred_bbox) + ")" + "--OVERLAP(" + str(self.overlapped_area) + ")"

    def __repr__(self):
        return "\n" + self.__str__()

def __computeByOverlapping(list_gt_bbox, list_pred_bbox):

    list_ordered_overlapping = []

    for pred_bbox in list_pred_bbox:
        assert(type(pred_bbox) is tuple or type(pred_bbox) is list)
        
        for gt_bbox in list_gt_bbox:
            assert(type(gt_bbox) is tuple)
            overlapped_bbox = OverlappingBBox(gt_bbox=gt_bbox, pred_bbox=pred_bbox)
            list_ordered_overlapping.append(overlapped_bbox)

    list_ordered_overlapping.sort(key=lambda x: x.overlapped_area, reverse=True)

    return list_ordered_overlapping



def getSimilarRegionPredwithGT(list_pred_bbox, list_gt_bbox, th=.55):

    similar_regions = []
    list_ordered_overlapping = __computeByOverlapping(list_gt_bbox, list_pred_bbox)

    list_discarded_gt = []
    for ordered_overlapping in list_ordered_overlapping:
        overlapping_area = ordered_overlapping.getOverlappingArea()
        if ordered_overlapping.gt_bbox not in list_discarded_gt:
            list_discarded_gt.append(ordered_overlapping.gt_bbox)
            if (overlapping_area >= th):
                similar_regions.append((ordered_overlapping.pred_bbox, ordered_overlapping.gt_bbox))

    return similar_regions


def getFscoreRegions(list_gt_bbox, list_pred_bbox, th=.55):

    assert(type(list_gt_bbox) is list)
    assert(type(list_pred_bbox) is list)
    assert(type(th) is float)
    assert(th>=0 and th <=1)

    tp = 0 #true positive
    fp = 0 #false positive
    fn = 0 #false negative

    all_overlapping_area = 0.
    overlapping_area_tp = 0.
    num_regions = 0

    list_overlapping_area = []
    list_ordered_overlapping = __computeByOverlapping(list_gt_bbox, list_pred_bbox)
    #print (list_ordered_overlapping)

    list_discarded_pred = []
    list_discarded_gt = []
    for ordered_overlapping in list_ordered_overlapping:
        overlapping_area = ordered_overlapping.getOverlappingArea()
        if ordered_overlapping.gt_bbox not in list_discarded_gt:
            all_overlapping_area += overlapping_area
            num_regions += 1
            list_overlapping_area.append(overlapping_area)
            list_discarded_gt.append(ordered_overlapping.gt_bbox)
            list_discarded_pred.append(ordered_overlapping.pred_bbox)
            if (overlapping_area >= th):
                tp = tp + 1
                overlapping_area_tp += overlapping_area
            else:
                fp = fp + 1 

    fn = abs(len(list_discarded_gt) - len(list_gt_bbox))

    if len(list_gt_bbox) == 0:
        return None, None, None, None, None, None, None, None, None, None
    if (tp + fn) == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)

    if (tp + fp) == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)
    
    if (num_regions == 0):
        all_overlapping_area = 0
        overlapping_area_tp = 0
    
    return 2 * (precision*recall) / (precision+recall + 0.000001), precision, recall, all_overlapping_area, tp, fn, fp, list_overlapping_area, num_regions, overlapping_area_tp

