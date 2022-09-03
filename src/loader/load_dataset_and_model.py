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

from transformers import ElectraConfig, ElectraForSequenceClassification, ElectraTokenizerFast
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer
from transformers import LongformerConfig, LongformerForSequenceClassification, LongformerTokenizer, LongformerSelfAttention

from bigbio.dataloader import BigBioConfigHelpers
from src.utils.args_helper import task_to_keys

### 
# Long Models Loader Function for BioElectra & Bio-lm
###
def convert_roberta_like_to_longformer(state_dict, model_name, max_seq_length):
    orig_keys = [key for key in state_dict]
    for key in orig_keys:
        if model_name in key:
            new_key = key.replace(model_name,'longformer')
            if '.position_ids' in new_key:
                del state_dict[key]
                continue

            if 'query.' in new_key:
                state_dict[new_key] = state_dict[key]
                state_dict[new_key.replace('.query.','.query_global.')] = state_dict[key]
            elif 'key.' in new_key:
                state_dict[new_key] = state_dict[key]
                state_dict[new_key.replace('.key.','.key_global.')] = state_dict[key]
            elif 'value.' in new_key:
                state_dict[new_key] = state_dict[key]
                state_dict[new_key.replace('.value.','.value_global.')] = state_dict[key]
            elif '.position_embeddings' in new_key:
                cur_seq_len = state_dict[key].shape[0]
                assert max_seq_length % cur_seq_len == 0
                
                multiplier = max_seq_length // cur_seq_len 
                state_dict[new_key] = state_dict[key].repeat([multiplier + 1, 1])
            else:
                state_dict[new_key] = state_dict[key]
            del state_dict[key]
    return state_dict

def load_long_model(model_args, data_args, num_labels, is_multilabel, is_regression):
    if 'bio-lm' in model_args.model_name_or_path:
        model_path = "models/RoBERTa-base-PM-M3-Voc-distill-align-hf"
        biolm_config = RobertaConfig.from_pretrained(
            model_path, 
            num_labels=num_labels,
            finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            problem_type='multi_label_classification' if is_multilabel else \
                         'single_label_classification' if not is_regression else \
                         'regression'
        )
        biolm_model = RobertaForSequenceClassification.from_pretrained(
            model_path,
            config=biolm_config
        )
        biolm_tokenizer = RobertaTokenizer.from_pretrained(model_path)
        
        # Longify BioLM
        longformer_config = LongformerConfig.from_pretrained(
            'allenai/longformer-base-4096',
            num_labels=num_labels,
            finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            problem_type='multi_label_classification' if is_multilabel else \
                         'single_label_classification' if not is_regression else \
                         'regression',
            max_position_embeddings=data_args.max_seq_length + 514,
            vocab_size=biolm_config.vocab_size,
            type_vocab_size=biolm_config.type_vocab_size,
            attention_window=[514] * 12
        )
        longformer_model = LongformerForSequenceClassification(config=longformer_config)
        
        longformer_state_dict = convert_roberta_like_to_longformer(biolm_model.state_dict(), 'roberta', data_args.max_seq_length)
        longformer_model.load_state_dict(longformer_state_dict, strict=True)
        
        return longformer_config, biolm_tokenizer, longformer_model
    else:
        bioelectra_config = ElectraConfig.from_pretrained(
            model_args.model_name_or_path, 
            num_labels=num_labels,
            finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            problem_type='multi_label_classification' if is_multilabel else \
                         'single_label_classification' if not is_regression else \
                         'regression'
        )
        bioelectra_model = ElectraForSequenceClassification.from_pretrained(
            model_args.model_name_or_path, 
            config=bioelectra_config
        )
        bioelectra_tokenizer = ElectraTokenizerFast.from_pretrained(model_args.model_name_or_path)

        # Longify Bioelectra
        longformer_config = LongformerConfig.from_pretrained(
            'allenai/longformer-base-4096',
            num_labels=num_labels,
            finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            problem_type='multi_label_classification' if is_multilabel else \
                         'single_label_classification' if not is_regression else \
                         'regression',
            max_position_embeddings=data_args.max_seq_length + 512,
            vocab_size=bioelectra_config.vocab_size,
            type_vocab_size=bioelectra_config.type_vocab_size,
            attention_window=[512] * 12
        )
        longformer_model = LongformerForSequenceClassification(config=longformer_config)
        longformer_state_dict = convert_roberta_like_to_longformer(bioelectra_model.state_dict(), 'electra', data_args.max_seq_length)
        longformer_model.load_state_dict(longformer_state_dict, strict=True)
        
        return longformer_config, bioelectra_tokenizer, longformer_model
    
###
# General Load Dataset & Models function
###
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
    if data_args.dataset_name in notes_dataset_names + ['n2c2_2008_intuitive', 'n2c2_2008_textual']:
        
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
                raw_datasets[helper.dataset_name] = helper.load_dataset(data_dir='./datasets/' + helper.dataset_name + '/PsyTAR_dataset.xlsx')
            elif helper.is_local:
                raw_datasets[helper.dataset_name] = helper.load_dataset(data_dir='./datasets/' + helper.dataset_name)
            else:
                raw_datasets[helper.dataset_name] = helper.load_dataset()
                
        # Choose dataset
        if 'n2c2_2008' in data_args.dataset_name:
            raw_datasets = raw_datasets['n2c2_2008']
        else:
            raw_datasets = raw_datasets[data_args.dataset_name]
    elif data_args.dataset_name == 'indonlu':
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.task_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
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
    is_regression = False
    is_multilabel = False
    if data_args.dataset_name == 'n2c2_2006_smokers':
        is_regression = False
        is_multilabel = False
        
        label_list = sorted(list(set(flatten_all_list(raw_datasets['train']['labels']))))
        num_labels = len(label_list)
        
    elif data_args.dataset_name == 'n2c2_2008_intuitive':        
        def numerize_multiclass_label(example):
            label_numerized = [None]*len(unique_disease_names)
            for anot_string in example['labels']:
                anot = eval(anot_string)
                if anot['annotation'] == 'intuitive':
                    disease_name=anot['disease_name']
                    label=anot['label']
                    label_numerized[unique_disease_names_to_id[disease_name]] = int(unique_labels_to_id[label])

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
        num_labels = len(unique_disease_names) * len(unique_labels)
        unique_labels_to_id = {v: i for i, v in enumerate(unique_labels)}
        unique_disease_names_to_id = {v: i for i, v in enumerate(unique_disease_names)}
        
        raw_datasets = raw_datasets.map(numerize_multiclass_label, load_from_cache_file=False)
        
    elif data_args.dataset_name == 'n2c2_2008_textual':  
        def numerize_multiclass_label(example):
            label_numerized = [None]*len(unique_disease_names)
            for anot_string in example['labels']:
                anot = eval(anot_string)
                if anot['annotation'] == 'textual':
                    disease_name=anot['disease_name']
                    label=anot['label']
                    label_numerized[unique_disease_names_to_id[disease_name]] = int(unique_labels_to_id[label])

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
        num_labels = len(unique_disease_names) * len(unique_labels)
        unique_labels_to_id = {v: i for i, v in enumerate(unique_labels)}
        unique_disease_names_to_id = {v: i for i, v in enumerate(unique_disease_names)}
        
        raw_datasets = raw_datasets.map(numerize_multiclass_label, load_from_cache_file=False)
        
    elif data_args.dataset_name == 'n2c2_2014_risk_factors':
        def one_hot_multiclass_label(example):
            label_numerized = [0.0]*len(unique_labels)
            for label_sentence in example['labels']:
                label = label_sentence[label_sentence.rfind('-')+len('-'):]
                label_numerized[unique_labels_to_id[label]]=1.0

            example['labels'] = label_numerized
            return example
        
        is_regression = False
        is_multilabel = True
        
        label_list = list(set(flatten_all_list(raw_datasets['train']['labels'])))

        labels = []
        for dict_string in label_list:
            last_column = dict_string[dict_string.rfind('-')+len('-'):]
            labels.append(last_column)

        unique_labels = list(set(labels))
        unique_labels.sort()
        num_labels = len(unique_labels)
        unique_labels_to_id = {v: i for i, v in enumerate(unique_labels)}

        raw_datasets = raw_datasets.map(one_hot_multiclass_label)
        
    elif data_args.dataset_name == 'n2c2_2018_track1':
        
        def one_hot_multiclass_label(example):
            label_numerized = [0] * len(unique_labels)

            for label in example['labels']:
                label_numerized[unique_labels_to_id[label]] = 1

            example['labels'] = label_numerized
            return example
        
        is_regression = False
        is_multilabel = True

        unique_labels = sorted(list(set(flatten_all_list(raw_datasets['train']['labels']))))        
        # num_labels = len(unique_labels) # BCE
        num_labels = len(unique_labels) * 2 # CE
        unique_labels_to_id = {v: i for i, v in enumerate(unique_labels)}

        raw_datasets = raw_datasets.map(one_hot_multiclass_label, load_from_cache_file=False)
        # quit()
    
    elif data_args.task_name is not None:
        
        is_multilabel = False
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
    if 'bioelectra' in model_args.model_name_or_path or 'bio-lm' in model_args.model_name_or_path:
        config, tokenizer, model = load_long_model(model_args, data_args, num_labels, is_multilabel, is_regression)
    else:
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
    
    ###
    # Preprocessing the raw_datasets
    ###
    if data_args.dataset_name == 'n2c2_2006_smokers':
        raw_datasets = raw_datasets.map(delist)
        sentence1_key = 'text'
        sentence2_key = None
    elif data_args.dataset_name == 'n2c2_2008_intuitive':
        sentence1_key = 'text'
        sentence2_key = None
    elif data_args.dataset_name == 'n2c2_2008_textual':
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
        # result = tokenizer(*args, padding=padding, truncation=False)
        
        if len(result['input_ids']) > max_seq_length:
            examples['input_ids'] = [result['input_ids'][0]] + result['input_ids'][1-max_seq_length:]
        else:
            examples['input_ids'] = result['input_ids'][:max_seq_length]
            
        if 'token_type_ids' in result:
            examples['token_type_ids'] = result['token_type_ids'][:max_seq_length]
        examples['attention_mask'] = result['attention_mask'][:max_seq_length]
        
#         if not is_multilabel:
#             # Map labels to IDs (not necessary for GLUE tasks)
#             if label_to_id is not None and "label" in examples:
#                 result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]

        return examples

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=False,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )
    
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
            
    # Use Test as Valid for debuging purpose
    raw_datasets = DatasetDict({
                    'train': raw_datasets['train'],
                    'validation': raw_datasets['test'],
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