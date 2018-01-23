import numpy as np
from sklearn.metrics import precision_score,recall_score,f1_score

def score(pred, label, gate=0.5):
    if len(label.shape) == 1:
        p = (pred>gate).astype("int")
        p = np.squeeze(p)
        l = label
    else:
        p = np.argmax(pred, axis=1)
        l = np.argmax(label, axis=1)
    pre_score = precision_score(l, p)
    rec_score = recall_score(l, p)
    f_score = f1_score(l, p)
    return pre_score, rec_score, f_score
