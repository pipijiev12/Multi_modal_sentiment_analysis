# -*- coding: utf-8 -*-

import torch
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score, recall_score
import numpy as np


def _binary_labels(values):
    """Convert continuous sentiment scores to the project's binary labels.

    This deliberately follows the existing protocol, in which scores greater
    than or equal to zero are positive.  Keeping the conversion in one place
    ensures that model scores and all reference baselines use identical labels.
    """
    return np.asarray(values).reshape(-1) >= 0


def _binary_metrics(target_labels, predicted_labels):
    """Return binary metrics with both classes explicitly represented."""
    target_labels = np.asarray(target_labels, dtype=bool).reshape(-1)
    predicted_labels = np.asarray(predicted_labels, dtype=bool).reshape(-1)
    labels = [False, True]
    return {
        'acc': accuracy_score(target_labels, predicted_labels),
        # Retain the legacy weighted F1 under its original key so existing
        # result readers remain compatible.
        'binary_f1': f1_score(
            target_labels, predicted_labels, labels=labels,
            average='weighted', zero_division=0,
        ),
        'balanced_accuracy': recall_score(
            target_labels, predicted_labels, labels=labels,
            average='macro', zero_division=0,
        ),
        'macro_f1': f1_score(
            target_labels, predicted_labels, labels=labels,
            average='macro', zero_division=0,
        ),
    }


def _majority_baseline(reference_targets, evaluation_targets):
    """Score a training-split majority-class classifier on evaluation targets.

    ``reference_targets`` must be drawn from the training split.  The test
    labels are used only for scoring, so the baseline cannot adapt to the test
    class balance.
    """
    reference_labels = _binary_labels(reference_targets)
    evaluation_labels = _binary_labels(evaluation_targets)
    positive_count = int(reference_labels.sum())
    negative_count = int(reference_labels.size - positive_count)
    # Choose negative in the deterministic tie case.  A tie is immaterial to
    # the definition of the baseline but makes the output reproducible.
    majority_is_positive = positive_count > negative_count
    predictions = np.full(
        evaluation_labels.shape, majority_is_positive, dtype=bool,
    )
    metrics = _binary_metrics(evaluation_labels, predictions)
    return {
        'majority_class': 'positive' if majority_is_positive else 'negative',
        'majority_train_prevalence': max(positive_count, negative_count) /
        reference_labels.size,
        'majority_baseline_acc': metrics['acc'],
        'majority_baseline_balanced_accuracy': metrics['balanced_accuracy'],
        'majority_baseline_macro_f1': metrics['macro_f1'],
    }


def evaluate(params, outputs, targets, baseline_reference_targets=None):
    if params.label == 'sentiment':
        # Single real-valued output
        
        outputs_np = outputs.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        binary_metrics = _binary_metrics(
            _binary_labels(targets_np), _binary_labels(outputs_np),
        )
        n_total = len(outputs)
        # ``recall`` is retained for backward-compatible output files.  It is
        # the legacy weighted recall, distinct from balanced accuracy above.
        recall = recall_score(
            _binary_labels(targets_np), _binary_labels(outputs_np),
            average='weighted', zero_division=0,
        )
            
        # Correlation
        corr = np.corrcoef(outputs_np.transpose(), targets_np.transpose())[0][1]    
        
        # MAE
        mae = torch.mean(torch.abs(outputs-targets)).item()
        
        # Accuracy for multiclass
        n_correct = sum(np.round(targets_np)==np.round(outputs_np))[0]
        acc_7 = n_correct/n_total
        
        targets_clamped = np.clip(targets_np, a_min = -2, a_max = 2)
        outputs_clamped = np.clip(outputs_np, a_min = -2, a_max = 2)
        
        n_correct = sum(np.round(targets_clamped)==np.round(outputs_clamped))[0]
        acc_5 = n_correct/n_total

        performance_dict = {
            **binary_metrics,
            'recall': recall,
            'accuracy_5': acc_5,
            'accuracy_7': acc_7,
            'MAE': mae,
            'r': corr,
        }
        if baseline_reference_targets is not None:
            performance_dict.update(
                _majority_baseline(baseline_reference_targets, targets_np)
            )
        
    else:
#        emos = ["Neutral", "Happy", "Sad", "Angry"]
        # outputs is of shape (batch_size, num_classes, 2)
        # targets is of shape (batch_size, num_classes, 2)
        outputs_max_ids = outputs.argmax(dim = -1).t()
        targets_max_ids = targets.argmax(dim = -1).t()

        num_classes,n_total = outputs_max_ids.shape        
        f1_per_class = []
        acc_per_class = []
        for class_i in range(num_classes):

            output_classes = outputs_max_ids[class_i].cpu().numpy()
            target_classes = targets_max_ids[class_i].cpu().numpy()
            f1 = f1_score(target_classes, output_classes, average='weighted')
            acc = accuracy_score(target_classes, output_classes)
            acc_per_class.append(acc)
            f1_per_class.append(f1)
        
        acc = float(torch.sum(outputs_max_ids == targets_max_ids))/float(num_classes*n_total)
        performance_dict = {'acc':acc,'f1_per_class':f1_per_class, 'acc_per_class':acc_per_class}

    return performance_dict
