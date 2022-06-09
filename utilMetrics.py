import util
from Results import Results


def getPseudoLabels(img, row, col):
    img_shape = img.shape
    center_label = img[row, col]
    vertical_up_label = img[max(0, row-1), col]
    vertical_down_label = img[min(img_shape[0]-1, row+1), col]
    vertical_left_label = img[row, max(0, col-1)]
    vertical_right_label = img[row, min(img_shape[1]-1, col+1)]

    return center_label, vertical_up_label, vertical_down_label, vertical_left_label, vertical_right_label


def computeMetricsFrom_TP_FP_FN(tp, fp, fn):
    if tp == 0:
        fscore = 0.
        precision = 0.
        recall = 0.
        print("TP = 0")
    else:
        fscore = tp / (tp + 0.5*(fp+fn))
        precision = tp / float(tp+fp)
        recall = tp / float(tp + fn)
    
    return fscore, precision, recall

def dumpTo_TP_FP_FN(label_pred, label_gt, label_possitive_class):
    tp = 0
    fp = 0
    fn = 0

    if (label_pred == label_gt):
        tp += 1
    elif label_gt == label_possitive_class:
        fp += 1
    else:
        fn += 1

    return tp, fp, fn

def dumpTo_pseudo_TP_FP_FN(
                            center_label_pred, vertical_up_pred, vertical_down_pred, vertical_left_pred, vertical_right_pred, 
                            center_label_gt,
                            label_possitive_class):
    tp = 0
    fp = 0
    fn = 0
    pseudo_tp = 0
    pseudo_fp = 0
    pseudo_fn = 0

    if center_label_gt == label_possitive_class:
        if (center_label_pred == center_label_gt):
            tp=1
            pseudo_tp=1
        else:
            fn=1
            if (vertical_up_pred == center_label_gt
                        or vertical_down_pred == center_label_gt
                        or vertical_left_pred == center_label_gt
                        or vertical_right_pred == center_label_gt):
                pseudo_tp = 1
            else:
                pseudo_fn = 1
    else:
        if (center_label_pred != center_label_gt):
            fp=1
            pseudo_fp=1

    return tp, fp, fn, pseudo_tp, pseudo_fp, pseudo_fn
    

def evaluateMetrics(pred_imgs, gt_imgs, label_possitive_class, verbose, threshold):
    tp = 0
    fp = 0
    fn = 0

    pseudo_tp = 0
    pseudo_fp = 0
    pseudo_fn = 0

    print ("Number of images: " + str(len(gt_imgs)))
    for idx in range(len(gt_imgs)):
        print ("Image: " + str(idx))
        tp_i = 0
        fp_i = 0
        fn_i = 0

        pseudo_tp_i = 0
        pseudo_fp_i = 0
        pseudo_fn_i = 0

        if threshold is not None:
            pred_img = 1*(pred_imgs[idx] > threshold)
        else:
            pred_img = pred_imgs[idx]
        
        gt_img = gt_imgs[idx]

        assert(gt_img.shape == (pred_img.shape[0], pred_img.shape[1]))
        img_shape = gt_img.shape

        if verbose:
            progress_bar = util.createProgressBar("Calculating metrics...", img_shape[0]*img_shape[1])
            progress_bar.start()
        idx = 0
        for row in range(img_shape[0]):
            for col in range(img_shape[1]):
                
                center_label_pred, vertical_up_pred, vertical_down_pred, vertical_left_pred, vertical_right_pred = getPseudoLabels(pred_img, row, col)
                center_label_gt = gt_img[row, col]

                tp_local, fp_local, fn_local, pseudo_tp_local, pseudo_fp_local, pseudo_fn_local = dumpTo_pseudo_TP_FP_FN(
                                                                            center_label_pred, vertical_up_pred, vertical_down_pred, vertical_left_pred, vertical_right_pred, 
                                                                            center_label_gt,
                                                                            label_possitive_class)
                tp_i+=tp_local
                fp_i+=fp_local
                fn_i+=fn_local
                pseudo_tp_i+=pseudo_tp_local
                pseudo_fp_i+=pseudo_fp_local
                pseudo_fn_i+=pseudo_fn_local

                idx += 1

                if verbose:
                    progress_bar.update(idx)

        fscore_i, precision_i, recall_i = computeMetricsFrom_TP_FP_FN(tp_i, fp_i, fn_i)
        print (str(tp_i) + " " + str(fp_i) + " " +str(fn_i))
        pseudo_fscore_i, pseudo_precision_i, pseudo_recall_i = computeMetricsFrom_TP_FP_FN(pseudo_tp_i, pseudo_fp_i, pseudo_fn_i)
        print (str(pseudo_tp_i) + " " + str(pseudo_fp_i) + " " +str(pseudo_fn_i))

        results_i = Results(precision_i, recall_i, fscore_i, -100, pseudo_precision_i, pseudo_recall_i, pseudo_fscore_i, -100)
        print (results_i)
        tp+=tp_i
        fp+=fp_i
        fn+=fn_i
        pseudo_tp+=pseudo_tp_i
        pseudo_fp+=pseudo_fp_i
        pseudo_fn+=pseudo_fn_i


        if verbose:
            progress_bar.finish()

    print ("End of testing images: ")

    fscore, precision, recall = computeMetricsFrom_TP_FP_FN(tp, fp, fn)
    pseudo_fscore, pseudo_precision, pseudo_recall = computeMetricsFrom_TP_FP_FN(pseudo_tp, pseudo_fp, pseudo_fn)

    return fscore, precision, recall, pseudo_fscore, pseudo_precision, pseudo_recall