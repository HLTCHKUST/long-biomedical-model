from transformers import Trainer

import torch


def get_trainer(dataset_name, *args, **kwargs):
    if dataset_name == "n2c2_2008":
        return TrainerN2C22018(*args, **kwargs)
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


class TrainerN2C22018(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        num_classes = labels.size()[-1]
        num_labels = self.model.config.num_labels // num_classes
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)

        # print("labels", labels.size(), labels)
        # print("logits", logits.size(), logits)

        loss = 0
        for class_id in range(num_classes):
            class_loss = loss_fct(logits[:, num_labels*class_id:num_labels*(class_id+1)], labels[:, class_id])
            loss += class_loss
            
        return (loss, outputs) if return_outputs else loss