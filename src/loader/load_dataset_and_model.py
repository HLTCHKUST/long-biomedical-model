import logging
import os
import random
import sys

import datasets
import numpy as np
from datasets import load_dataset, DatasetDict

import evaluate
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

from bigbio.dataloader import BigBioConfigHelpers
from src.utils.args_helper import task_to_keys

def load_datasets(data_args, model_args, training_args):


    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    
    notes_dataset_names = ['n2c2_2006_smokers', 'n2c2_2008', 'n2c2_2014_risk_factors', 'n2c2_2018_track1']
    if data_args.dataset_name in notes_dataset_names:
        
        conhelps = BigBioConfigHelpers()
        notes_dset_helpers = conhelps.filtered(
            lambda x: x.dataset_name in notes_dataset_names
            and x.is_bigbio_schema
            and 'fold' not in x.config.subset_id
            and x.config.schema == 'bigbio_text' 
        )

        raw_datasets = {}
        for helper in notes_dset_helpers:
            if helper.dataset_name == 'psytar':
                raw_datasets[helper.dataset_name] = helper.load_dataset(data_dir=helper.dataset_name + '/PsyTAR_dataset.xlsx')
            elif helper.is_local:
                raw_datasets[helper.dataset_name] = helper.load_dataset(data_dir=helper.dataset_name)
            else:
                raw_datasets[helper.dataset_name] = helper.load_dataset()
                
        # Choose dataset
        raw_datasets = raw_datasets[data_args.dataset_name]
    elif data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            "glue",
            data_args.task_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    elif data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
#     else:
#         # Loading a dataset from your local files.
#         # CSV/JSON training and evaluation files are needed.
#         data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

#         # Get the test dataset: you can provide your own CSV/JSON test file (see below)
#         # when you use `do_predict` without specifying a GLUE benchmark task.
#         if training_args.do_predict:
#             if data_args.test_file is not None:
#                 train_extension = data_args.train_file.split(".")[-1]
#                 test_extension = data_args.test_file.split(".")[-1]
#                 assert (
#                     test_extension == train_extension
#                 ), "`test_file` should have the same extension (csv or json) as `train_file`."
#                 data_files["test"] = data_args.test_file
#             else:
#                 raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

#         for key in data_files.keys():
#             logger.info(f"load a local file for {key}: {data_files[key]}")

#         if data_args.train_file.endswith(".csv"):
#             # Loading a dataset from local csv files
#             raw_datasets = load_dataset(
#                 "csv",
#                 data_files=data_files,
#                 cache_dir=model_args.cache_dir,
#                 use_auth_token=True if model_args.use_auth_token else None,
#             )
#         else:
#             # Loading a dataset from local json files
#             raw_datasets = load_dataset(
#                 "json",
#                 data_files=data_files,
#                 cache_dir=model_args.cache_dir,
#                 use_auth_token=True if model_args.use_auth_token else None,
#             )
    else:
        raise NotImplementedError('To this end, we have only implemented the loaders for GLUE and '+", ".join(notes_dataset_names))
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.
        
    
    def flatten(list_of_lists):
        flattened = []
        for sublist in list_of_lists:
            if type(sublist) == type([]):
                for item in sublist:
                    flattened.append(item)
            else:
                flattened.append(sublist)
        return flattened

    def flatten_all_list(list_of_lists):

        still_contains_list = sum([1 if type(element) == type([]) else 0 for element in list_of_lists ])

        while still_contains_list > 0:
            list_of_lists = flatten(list_of_lists)
            still_contains_list = sum([1 if type(element) == type([]) else 0 for element in list_of_lists])

        return list_of_lists
    
    def delist(example):
        example["labels"] = example["labels"][0]
        return example

    # Labels
    if data_args.dataset_name == 'n2c2_2006_smokers':
        
        is_regression = False
        is_multilabel = False
        
        label_list = list(set(flatten_all_list(raw_datasets['train']['labels'])))
        label_list.sort()
        num_labels = len(label_list)
        
    elif data_args.dataset_name == 'n2c2_2008':
        
        def numerize_multiclass_label(example):

            label_numerized = [None]*len(unique_disease_names)
            for dict_string in example['labels']:
                disease_name = dict_string[dict_string.find('disease_name')+len('disease_name')+4:dict_string.find('label')-4]
                label = dict_string[dict_string.find('label')+len('label')+4:-2]
                label_numerized[unique_disease_names_to_id[disease_name]] = float(unique_labels_to_id[label])

            example['labels'] = [-1 if label==None else label for label in label_numerized]
            return example
        
        is_regression = False
        is_multilabel = True
        
        disease_names, labels = [], []
        label_list = list(set(flatten_all_list(raw_datasets['train']['labels'])))
        for dict_string in label_list:
            disease_name = dict_string[dict_string.find('disease_name')+len('disease_name')+4:dict_string.find('label')-4]
            label = dict_string[dict_string.find('label')+len('label')+4:-2]
            disease_names.append(disease_name)
            labels.append(label)

        unique_labels = list(set(labels))
        unique_labels.sort()
        unique_disease_names = list(set(disease_names))
        unique_disease_names.sort()
        num_labels = len(unique_disease_names)
        unique_labels_to_id = {v: i for i, v in enumerate(unique_labels)}
        unique_disease_names_to_id = {v: i for i, v in enumerate(unique_disease_names)}
        
        raw_datasets = raw_datasets.map(numerize_multiclass_label)
        
    elif data_args.dataset_name == 'n2c2_2014_risk_factors':
        
        is_regression = False
        is_multilabel = True
        
        label_list = list(set(flatten_all_list(raw_datasets['train']['labels'])))
        label_list.sort()
        num_labels = len(label_list)
        
    elif data_args.dataset_name == 'n2c2_2018_track1':
        
        def one_hot_multiclass_label(example):
            label_numerized = [1.0]*len(unique_labels)
            for label in example['labels']:
                label_numerized[unique_labels_to_id[label]] = .0

            example['labels'] = label_numerized
            return example
        
        is_regression = False
        is_multilabel = True
        
        label_list = list(set(flatten_all_list(raw_datasets['train']['labels'])))
        label_list.sort()
        num_labels = len(label_list)

        unique_labels = list(set(flatten_all_list(raw_datasets['train']['labels'])))
        unique_labels_to_id = {v: i for i, v in enumerate(unique_labels)}

        raw_datasets = raw_datasets.map(one_hot_multiclass_label)
    
    elif data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)
    
    
    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        problem_type='multi_label_classification' if is_multilabel else \
                     'single_label_classification' if not is_regression else \
                     'regression'
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )
    
    # Preprocessing the raw_datasets
    if data_args.dataset_name == 'n2c2_2006_smokers':
        raw_datasets = raw_datasets.map(delist)
        sentence1_key = 'text'
        sentence2_key = None
    elif data_args.dataset_name == 'n2c2_2008':
        sentence1_key = 'text'
        sentence2_key = None
    elif data_args.dataset_name == 'n2c2_2014_risk_factors':
        sentence1_key = 'text'
        sentence2_key = None
    elif data_args.dataset_name == 'n2c2_2018_track1':
        sentence1_key = 'text'
        sentence2_key = None
    elif data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False
    
    if not is_multilabel:
        # Some models have set the order of the labels to use, so let's make sure we do use it.
        label_to_id = None
        if (
            model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
            and data_args.task_name is not None
            and not is_regression
        ):
            # Some have all caps in their config, some don't.
            label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
            if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
                label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
            else:
                logger.warning(
                    "Your model seems to have been trained with labels, but they don't match the dataset: ",
                    f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                    "\nIgnoring the model labels as a result.",
                )
        elif data_args.task_name is None and not is_regression:
            label_to_id = {v: i for i, v in enumerate(label_list)}

        if label_to_id is not None:
            def convert_label_to_id(example):
                example["labels"] = label_to_id[example["labels"]]
                return example
            raw_datasets = raw_datasets.map(convert_label_to_id)

            model.config.label2id = label_to_id
            model.config.id2label = {id: label for label, id in config.label2id.items()}
        elif data_args.task_name is not None and not is_regression:
            model.config.label2id = {l: i for i, l in enumerate(label_list)}
            model.config.id2label = {id: label for label, id in config.label2id.items()}
    

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        if not is_multilabel:
            # Map labels to IDs (not necessary for GLUE tasks)
            if label_to_id is not None and "label" in examples:
                result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]

        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
        
    print('XXX'*300)    
    print(raw_datasets)
    
    def turn_to_float(examples):
        examples['labels'] = np.array(examples["labels"], dtype=np.float32)
        return examples
    
    if is_multilabel:
        raw_datasets = raw_datasets.map(turn_to_float)
        
    print('XXX'*300)
    print(raw_datasets['train'][0]['labels'])
    
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
            
    # Add valid
    raw_datasets = DatasetDict({
                    'train': raw_datasets['train'].train_test_split(0.1)['train'],
                    'validation': raw_datasets['train'].train_test_split(0.1)['test'],
                    'test': raw_datasets['test']})

    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
            
    return train_dataset, eval_dataset, model, tokenizer, is_regression, is_multilabel