from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from transformers import Trainer, EvalPrediction

import itertools
import numpy as np
import torch
import torch.nn as nn


def get_trainer(dataset_name, *args, **kwargs):
    if "n2c2_2008" in dataset_name:
        return TrainerN2C22008(*args, **kwargs)
    elif "n2c2_2006" in dataset_name:
        return TrainerN2C22006(*args, **kwargs)
    else:
        return Trainer(*args, **kwargs)


class TrainerN2C22006(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss
        loss_fct = nn.CrossEntropyLoss()
        # loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(
        #     [1.89090909, 1.3       , 1.89090909, 6.93333333, 0.33015873],
        #     device=logits.device
        # ))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

class TrainerN2C22008(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        num_labels = labels.size()[-1]
        num_classes = self.model.config.num_labels // num_labels
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)

        loss = 0
        for label_id in range(num_labels):
            class_loss = loss_fct(logits[:, num_classes*label_id:num_classes*(label_id+1)], labels[:, label_id])
            loss += class_loss / num_labels
            
        return (loss, outputs) if return_outputs else loss


def get_compute_metrics(dataset_name):

    def single_class_multi_label_metrics(p: EvalPrediction): # single class multilabel
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds[np.isnan(preds)]=-1
        p.label_ids[np.isnan(p.label_ids)]=-1
        
        preds = list(itertools.chain.from_iterable(preds))
        p.label_ids = list(itertools.chain.from_iterable(p.label_ids))

        preds = np.array(preds).astype('int32')
        p.label_ids = np.array(p.label_ids).astype('int32')
        
        p.label_ids, preds = zip(*list(filter(lambda row: row[0] != -1, zip(p.label_ids, preds))))

        return {"acc": accuracy_score(p.label_ids, preds),
                "micro-f1": f1_score(p.label_ids, preds, average='micro'),
                "micro-recall": recall_score(p.label_ids, preds, average='micro'),
                "micro-prec": precision_score(p.label_ids, preds, average='micro'),
                "macro-f1": f1_score(p.label_ids, preds, average='macro'),
                "macro-recall": recall_score(p.label_ids, preds, average='macro'),
                "macro-prec": precision_score(p.label_ids, preds, average='macro')}

    def multi_class_multi_label_metrics(p: EvalPrediction): # multiclass multilabel
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        
        p.label_ids[np.isnan(p.label_ids)]=-1
        
        preds = list(itertools.chain.from_iterable(preds))
        p.label_ids = list(itertools.chain.from_iterable(p.label_ids))
        
        _preds = preds
        num_labels = len(p.label_ids)
        num_classes = len(preds) // num_labels
        preds = [np.argmax(_preds[num_classes*i:num_classes*(i+1)]) for i in range(num_labels)]
        preds = np.array(preds).astype('int32')
        p.label_ids = np.array(p.label_ids).astype('int32')
        
        p.label_ids, preds = zip(*list(filter(lambda row: row[0] != -1, zip(p.label_ids, preds))))
        
        return {"acc": accuracy_score(p.label_ids, preds),
                "micro-f1": f1_score(p.label_ids, preds, average='micro'),
                "micro-recall": recall_score(p.label_ids, preds, average='micro'),
                "micro-prec": precision_score(p.label_ids, preds, average='micro'),
                "macro-f1": f1_score(p.label_ids, preds, average='macro'),
                "macro-recall": recall_score(p.label_ids, preds, average='macro'),
                "macro-prec": precision_score(p.label_ids, preds, average='macro')}

    def single_label_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        preds = np.array(preds).astype('int32')
        p.label_ids = np.array(p.label_ids).astype('int32')
        
        return {"acc": accuracy_score(p.label_ids, preds),
                "micro-f1": f1_score(p.label_ids, preds, average='micro'),
                "micro-recall": recall_score(p.label_ids, preds, average='micro'),
                "micro-prec": precision_score(p.label_ids, preds, average='micro'),
                "macro-f1": f1_score(p.label_ids, preds, average='macro'),
                "macro-recall": recall_score(p.label_ids, preds, average='macro'),
                "macro-prec": precision_score(p.label_ids, preds, average='macro')}

    if "n2c2_2008" in dataset_name:
        return multi_class_multi_label_metrics
    elif "n2c2_2006" in dataset_name:
        return single_label_metrics
    elif "n2c2_2018" in dataset_name:
        return single_class_multi_label_metrics
    else:
        # preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        
        # if not is_multilabel:
        #     preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        # else:
        #     preds[np.isnan(preds)]=-1
        #     p.label_ids[np.isnan(p.label_ids)]=-1
            
        #     preds = list(itertools.chain.from_iterable(preds))
        #     p.label_ids = list(itertools.chain.from_iterable(p.label_ids))

        #     # ## QUICK FIX FOR 2008
        #     # _preds = preds
        #     # num_labels = len(p.label_ids)
        #     # num_classes = len(preds) // num_labels
        #     # preds = [np.argmax(_preds[num_classes*i:num_classes*(i+1)]) for i in range(num_labels)]
        # if is_regression:
        #     return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        # else:
        #     preds = np.array(preds).astype('int32')
        #     p.label_ids = np.array(p.label_ids).astype('int32')
            
        #     return {"acc": accuracy_score(p.label_ids, preds),
        #             "micro-f1": f1_score(p.label_ids, preds, average='micro'),
        #             "micro-recall": recall_score(p.label_ids, preds, average='micro'),
        #             "micro-prec": precision_score(p.label_ids, preds, average='micro'),
        #             "macro-f1": f1_score(p.label_ids, preds, average='macro'),
        #             "macro-recall": recall_score(p.label_ids, preds, average='macro'),
        #             "macro-prec": precision_score(p.label_ids, preds, average='macro')}

        return NotImplementedError