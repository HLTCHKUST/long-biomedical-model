from transformers import Trainer

import torch


def get_trainer(dataset_name, *args, **kwargs):
    if dataset_name == "n2c2_2008":
        return TrainerN2C22008(*args, **kwargs)
    elif dataset_name == "n2c2_2006_smokers":
        return TrainerN2C22006(*args, **kwargs)
    else:
        return Trainer(**args, **kwargs)


class TrainerN2C22006(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(
            [1.89090909, 1.3       , 1.89090909, 6.93333333, 0.33015873],
            device=logits.device
        ))
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
            loss += class_loss
            
        return (loss, outputs) if return_outputs else loss