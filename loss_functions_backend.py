from keras import backend as K

smooth = 1e-12

def jaccard_loss(y_true, y_pred):
    smooth = 0.001
    intersection = K.sum(y_true * y_pred, axis=[0, 1, 2])
    sum_ = K.sum(y_true + y_pred, axis=[0, 1, 2])

    jac = 1 - (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)

def iou_metric(y_true, y_pred):
    smooth = 0.001
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    intersection = K.sum(y_true * y_pred_pos, axis=[0, 1, 2])
    sum_ = K.sum(y_true + y_pred, axis=[0, 1, 2])
    iou = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(iou)

def jaccard_loss_weighted(y_true, y_pred, weights = 1):
    smooth = 0.001
    intersection = K.sum(y_true * y_pred, axis=[0, 1, 2])
    sum_ = K.sum(y_true + y_pred, axis=[0, 1, 2])

    jac = 1 - (intersection + smooth) / (sum_ - intersection + smooth)
    jac = jac*weights
    return K.mean(jac)

def jaccard_coef(y_true, y_pred):
    # __author__ = Vladimir Iglovikov
    intersection = K.sum(y_true * y_pred, axis=[0, 1, 2])
    sum_ = K.sum(y_true + y_pred, axis=[0, 1, 2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)


def jaccard_coef_int(y_true, y_pred):
    # __author__ = Vladimir Iglovikov
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    intersection = K.sum(y_true * y_pred_pos, axis=[0, 1, 2])
    sum_ = K.sum(y_true + y_pred, axis=[0, 1, 2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)


def jac_loss(y_true, y_pred):
    return 1-jaccard_coef_int(y_true, y_pred)

def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=[0, 1, 2])
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)), axis=[0, 1, 2])
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=[0, 1, 2])
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)), axis=[0, 1, 2])
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1(y_true, y_pred):
    precision_ = precision(y_true, y_pred)
    recall_ = recall(y_true, y_pred)
    return 2*((precision_*recall_)/(precision_+recall_))