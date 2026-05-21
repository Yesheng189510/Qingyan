# coding: utf-8

import warnings
import logging
import functools
import numpy as np
import scipy as sp

from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import unique_labels, type_of_target
from sklearn.metrics import precision_recall_fscore_support

try:
    from inspect import signature
except ImportError:
    from sklearn.externals.funcsigs import signature


LOGGER = logging.getLogger(__name__)


# =========================
# SAFE replacements (关键修复)
# =========================

def _check_targets(y_true, y_pred):
    """
    Replacement for deprecated sklearn.metrics.classification._check_targets
    """
    return type_of_target(y_true), np.array(y_true), np.array(y_pred)


def _prf_divide(numerator, denominator, metric, modifier, average, warn_for):
    """
    Replacement for sklearn internal _prf_divide
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        result = numerator / denominator
        result[~np.isfinite(result)] = 0.0
    return result


# =========================
# main function
# =========================

def sensitivity_specificity_support(
    y_true,
    y_pred,
    labels=None,
    pos_label=1,
    average=None,
    warn_for=('sensitivity', 'specificity'),
    sample_weight=None
):

    average_options = (None, 'micro', 'macro', 'weighted', 'samples')
    if average not in average_options and average != 'binary':
        raise ValueError("average must be one of " + str(average_options))

    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
    present_labels = unique_labels(y_true, y_pred)

    if labels is None:
        labels = present_labels
        n_labels = None
    else:
        n_labels = len(labels)
        labels = np.hstack([labels, np.setdiff1d(present_labels, labels)])

    # encode labels
    le = LabelEncoder()
    le.fit(labels)

    y_true_enc = le.transform(y_true)
    y_pred_enc = le.transform(y_pred)

    # TP / FP / FN / TN
    tp = (y_true_enc == y_pred_enc)

    tp_bins = y_true_enc[tp]

    tp_sum = np.bincount(tp_bins, minlength=len(labels))
    pred_sum = np.bincount(y_pred_enc, minlength=len(labels))
    true_sum = np.bincount(y_true_enc, minlength=len(labels))

    tn_sum = y_true.size - (pred_sum + true_sum - tp_sum)

    # select labels
    sorted_labels = le.classes_
    indices = np.searchsorted(sorted_labels, labels[:n_labels])

    tp_sum = tp_sum[indices]
    pred_sum = pred_sum[indices]
    true_sum = true_sum[indices]
    tn_sum = tn_sum[indices]

    # micro
    if average == 'micro':
        tp_sum = np.array([tp_sum.sum()])
        pred_sum = np.array([pred_sum.sum()])
        true_sum = np.array([true_sum.sum()])
        tn_sum = np.array([tn_sum.sum()])

    with np.errstate(divide='ignore', invalid='ignore'):
        specificity = _prf_divide(
            tn_sum,
            tn_sum + pred_sum - tp_sum,
            'specificity', 'pred', average, warn_for
        )

        sensitivity = _prf_divide(
            tp_sum,
            true_sum,
            'sensitivity', 'true', average, warn_for
        )

    if average == 'weighted':
        weights = true_sum
        if weights.sum() == 0:
            return 0, 0, None
    else:
        weights = None

    if average is not None:
        specificity = np.average(specificity, weights=weights)
        sensitivity = np.average(sensitivity, weights=weights)
        true_sum = None

    return sensitivity, specificity, true_sum


# =========================
# wrappers
# =========================

def sensitivity_score(y_true, y_pred, **kwargs):
    s, _, _ = sensitivity_specificity_support(
        y_true, y_pred, warn_for=('sensitivity',), **kwargs
    )
    return s


def specificity_score(y_true, y_pred, **kwargs):
    _, s, _ = sensitivity_specificity_support(
        y_true, y_pred, warn_for=('specificity',), **kwargs
    )
    return s


# =========================
# G-mean
# =========================

def geometric_mean_score(
    y_true,
    y_pred,
    labels=None,
    pos_label=1,
    average='multiclass',
    sample_weight=None,
    correction=0.0
):

    if average is None or average != 'multiclass':
        sen, spe, _ = sensitivity_specificity_support(
            y_true, y_pred,
            labels=labels,
            pos_label=pos_label,
            average=average,
            sample_weight=sample_weight
        )

        return np.sqrt(sen * spe)

    # multiclass case
    present_labels = unique_labels(y_true, y_pred)

    if labels is None:
        labels = present_labels

    le = LabelEncoder()
    le.fit(labels)

    y_true_enc = le.transform(y_true)
    y_pred_enc = le.transform(y_pred)

    tp = (y_true_enc == y_pred_enc)
    tp_bins = y_true_enc[tp]

    tp_sum = np.bincount(tp_bins, minlength=len(labels))
    true_sum = np.bincount(y_true_enc, minlength=len(labels))

    recall = _prf_divide(tp_sum, true_sum, "recall", "true", None, "recall")
    recall[recall == 0] = correction

    return sp.stats.gmean(recall)