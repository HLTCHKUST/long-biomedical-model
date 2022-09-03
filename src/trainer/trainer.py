from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from transformers import Trainer, EvalPrediction

import itertools
import numpy as np
import torch
import torch.nn as nn

from .eval_n2c2_2018 import eval_n2c2_2018


def get_trainer(dataset_name, *args, **kwargs):
    if "n2c2_2006" in dataset_name:
        return TrainerN2C22006(*args, **kwargs)
        print('n2c2_2006 Trainer')
    elif "n2c2_2008" in dataset_name:
        return TrainerN2C22008(*args, **kwargs)
        print('n2c2_2008 Trainer')
    elif "n2c2_2018" in dataset_name:
        print('n2c2_2018 Trainer')
        return TrainerN2C22018(*args, **kwargs)
    else:
        print('OTHER Trainer')
        return Trainer(*args, **kwargs)


class TrainerN2C22006(Trainer):
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss
        loss_fct = nn.CrossEntropyLoss()
        # loss_weights = torch.tensor([
        #     1.89090909, 1.3, 1.89090909, 6.93333333, 0.33015873
        # ], dtype=torch.float, requires_grad=False, device=logits.device)
        # loss_fct = nn.CrossEntropyLoss(weight=loss_weights)
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

class TrainerN2C22018(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        num_labels = labels.size()[-1]
        num_classes = self.model.config.num_labels // num_labels

        # loss_weight = torch.tensor([
        #     1.623376623, 0.616, 27.85714286, 0.2469135802,
        #     1.463414634, 0.9238095238, 15.83333333, 0.05208333333,
        #     2.014925373, 201, 0.7876106195, 0.0412371134, 10.22222222
        # ], dtype=torch.float, requires_grad=False, device=logits.device)
        
        # loss_weight = torch.tensor([
        #     1.615384615, 0.619047619, 24.5, 0.2515337423,
        #     1.457831325, 0.9245283019, 14.69230769, 0.05699481865,
        #     2, 101, 0.7894736842, 0.04615384615, 9.736842105
        # ], dtype=torch.float, requires_grad=False, device=logits.device)
        
        # loss_weight = torch.tensor([
        #     1.311688312, 0.808, 14.42857143, 0.6234567901,
        #     1.231707317, 0.9619047619, 8.416666667, 0.5260416667,
        #     1.507462687, 101, 0.8938053097, 0.5206185567, 5.611111111
        # ], dtype=torch.float, requires_grad=False, device=logits.device)
        
        # loss_fct = nn.BCEWithLogitsLoss(pos_weight=loss_weight)
        # loss_fct = nn.BCEWithLogitsLoss()
        # loss = loss_fct(logits, labels.float().to(logits.device))
        
        loss_weights = [
            torch.tensor([0.808, 1.3116883116883118], dtype=torch.float, requires_grad=False, device=logits.device),
            torch.tensor([1.3116883116883118, 0.808], dtype=torch.float, requires_grad=False, device=logits.device),
            torch.tensor([0.517948717948718, 14.428571428571429], dtype=torch.float, requires_grad=False, device=logits.device),
            torch.tensor([2.525, 0.6234567901234568], dtype=torch.float, requires_grad=False, device=logits.device),
            torch.tensor([0.8416666666666667, 1.2317073170731707], dtype=torch.float, requires_grad=False, device=logits.device),
            torch.tensor([1.041237113402062, 0.9619047619047619], dtype=torch.float, requires_grad=False, device=logits.device),
            torch.tensor([0.531578947368421, 8.416666666666666], dtype=torch.float, requires_grad=False, device=logits.device),
            torch.tensor([10.1, 0.5260416666666666], dtype=torch.float, requires_grad=False, device=logits.device),
            torch.tensor([0.7481481481481481, 1.507462686567164], dtype=torch.float, requires_grad=False, device=logits.device),
            torch.tensor([0.5024875621890548, 101.0], dtype=torch.float, requires_grad=False, device=logits.device),
            torch.tensor([1.1348314606741574, 0.8938053097345132], dtype=torch.float, requires_grad=False, device=logits.device),
            torch.tensor([12.625, 0.520618556701031], dtype=torch.float, requires_grad=False, device=logits.device),
            torch.tensor([0.5489130434782609, 5.611111111111111], dtype=torch.float, requires_grad=False, device=logits.device)
        ]
        
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)
        loss = 0
        for label_id in range(num_labels):
            loss_fct =torch.nn.CrossEntropyLoss(weight=loss_weights[label_id], ignore_index=-1)
            class_loss = loss_fct(logits[:, num_classes*label_id:num_classes*(label_id+1)], labels[:, label_id])
            loss += class_loss / num_labels
            
        return (loss, outputs) if return_outputs else loss


def get_compute_metrics(dataset_name):

    def single_class_multi_label_metrics(p: EvalPrediction): # single class multilabel
        probas = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        probas[np.isnan(probas)]=-1
        p.label_ids[np.isnan(p.label_ids)]=-1
        
        probas = list(itertools.chain.from_iterable(probas))
        p.label_ids = list(itertools.chain.from_iterable(p.label_ids))
                
        num_classes = len(probas) // len(p.label_ids)
        num_flatten_labels = len(p.label_ids)
        
        preds = [np.argmax(probas[num_classes*i:num_classes*(i+1)]) for i in range(num_flatten_labels)] # CE
        
        # preds = np.where(np.array(probas) <= 0, 0, 1).astype('int32') # BCE
        # for i, (_l, _p, _pr) in enumerate(zip(p.label_ids, preds, probas)):
        #    if i % 13 == 12:
        #        print(f'{_l}, {_p}, {"-" if _pr < 0 else "+"}{-_pr if _pr < 0 else _pr:0.1f} | {i // 13}')
        #    else:
        #        print(f'{_l}, {_p}, {"-" if _pr < 0 else "+"}{-_pr if _pr < 0 else _pr:0.1f} |', end=" ")
       
        for i, (_l, _p) in enumerate(zip(p.label_ids, preds)):
            if i % 13 == 12:
                print(f'{_l}, {_p} | {i // 13}')
            else:
                print(f'{_l}, {_p} |', end=" ")
        
        p.label_ids = np.array(p.label_ids).astype('int32')
        
        tags = [
            'ABDOMINAL', 'ADVANCED-CAD', 'ALCOHOL-ABUSE',
            'ASP-FOR-MI', 'CREATININE', 'DIETSUPP-2MOS',
            'DRUG-ABUSE', 'ENGLISH', 'HBA1C', 'KETO-1YR',
            'MAJOR-DIABETES', 'MAKES-DECISIONS', 'MI-6MOS'
        ]
        
        golds, hyps = [], []
        for i, (g, s) in enumerate(zip(p.label_ids, preds)):
            if i % len(tags) == 0:
                gold, hyp = {}, {}
                golds.append(gold)
                hyps.append(hyp)
            gold[tags[i % len(tags)]] = 'met' if g == 1 else 'not met'
            hyp[tags[i % len(tags)]] = 'met' if s == 1 else 'not met'

        metrics = eval_n2c2_2018(golds, hyps)
        return metrics

    def multi_class_multi_label_metrics(p: EvalPrediction): # multiclass multilabel
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions        
        p.label_ids[np.isnan(p.label_ids)]=-1
        
        
        preds = list(itertools.chain.from_iterable(preds))
        p.label_ids = list(itertools.chain.from_iterable(p.label_ids))

        num_classes = len(preds) // len(p.label_ids)
        num_flatten_labels = len(p.label_ids)
        
        _preds = preds
        preds = [np.argmax(_preds[num_classes*i:num_classes*(i+1)]) for i in range(num_flatten_labels)]
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
        # return multi_class_multi_label_metrics
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