import numpy as np

smooth = 1e-12
def np_jaccard_coef(y_true, y_pred):
    # __author__ = Vladimir Iglovikov
    intersection = np.sum(np.round(np.clip(y_true, 0, 1)) * np.round(np.clip(y_pred, 0, 1)))
    sum_ = np.sum(np.round(np.clip(y_true, 0, 1)) + np.round(np.clip(y_pred, 0, 1)))

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return np.mean(jac)

def np_recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives)
    return recall

def np_precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    predicted_positives = np.sum(np.round(np.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives)
    return precision

def np_f1(y_true, y_pred):
    precision_ = np_precision(y_true, y_pred)
    recall_ = np_recall(y_true, y_pred)
    return 2*((precision_*recall_)/(precision_+recall_))