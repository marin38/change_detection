import cv2
import numpy as np
from np_metrics import np_jaccard_coef

def calculate_objectwise_metrics(img_pred_path, img_gt_path, threshold=0.5):
    def get_contours(img):
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        _,thresh = cv2.threshold(img_gray,127,255,0)
        _, contours, _ = cv2.findContours(thresh ,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    return contours

    def check_counters_iou_greater_threshold(counter_1, counter_2, shape):
        black_image = np.zeros(shape, np.uint8)
        img_1_ = cv2.drawContours(black_image.copy(), [counter_1], 0, (255,255,255), thickness=cv2.FILLED)
        img_2_ = cv2.drawContours(black_image.copy(), [counter_2], 0, (255,255,255), thickness=cv2.FILLED)
        img_1_ = cv2.cvtColor(img_1_,cv2.COLOR_BGR2GRAY)
        img_2_ = cv2.cvtColor(img_2_,cv2.COLOR_BGR2GRAY)
        return np_jaccard_coef(img_1_, img_2_) >= IOU_THRESHOLD

    def calculate_obj_classification_matrix(counters_pred, counters_gt, shape):
        TP=0
        TN=0
        FP=0
        checked_1 = [False]*len(counters_pred)
        checked_2 = [False]*len(counters_gt)
        for i in range(len(counters_pred)):
            for j in range(len(counters_gt)):
                if check_counters_iou_greater_threshold(counters_pred[i], counters_gt[j], shape):
                    checked_1[i]=True
                    checked_2[j]=True
                    TP += 1
        for i in range(len(counters_pred)):
            if not checked_1[i]:
                FP += 1
        for j in range(len(counters_gt)):
            if not checked_2[j]:
                TN += 1
        return TP, FP, TN

    IOU_THRESHOLD = threshold
    img_gt = cv2.imread(IMG_GT_PATH)
    img_pred = cv2.imread(IMG_PRED_PATH)
    assert img_gt.shape == img_pred.shape
    contours_gt = get_contours(img_gt)
    contours_pred = get_contours(img_pred)
    return calculate_obj_classification_matrix(contours_pred, contours_gt, img_gt.shape)


            