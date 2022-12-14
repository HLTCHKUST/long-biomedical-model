export DATASET_NAME=n2c2_2018_track1
export TOKENIZERS_PARALLELISM=false

# # Baselines (bert-base-uncased, bert-base-cased, PubmedBERT, ClinicalBERT)
# CUDA_VISIBLE_DEVICES=0 python run_nlu.py \
#   --model_name_or_path bert-base-uncased \
#   --dataset_name $DATASET_NAME \
#   --do_train \
#   --do_eval \
#   --logging_steps 10 \
#   --max_seq_length 512 \
#   --per_device_train_batch_size 8 \
#   --gradient_accumulation_steps 1 \
#   --learning_rate 5e-5 \
#   --num_train_epochs 80 \
#   --evaluation_strategy epoch \
#   --output_dir ./save/bert-base-uncased-$DATASET_NAME/ \
#   --overwrite_output_dir

# CUDA_VISIBLE_DEVICES=0 python run_nlu.py \
#   --model_name_or_path bert-base-cased \
#   --dataset_name $DATASET_NAME \
#   --do_train \
#   --do_eval \
#   --logging_steps 10 \
#   --max_seq_length 512 \
#   --per_device_train_batch_size 8 \
#   --gradient_accumulation_steps 1 \
#   --learning_rate 5e-5 \
#   --num_train_epochs 80 \
#   --evaluation_strategy epoch \
#   --output_dir ./save/bert-base-cased-$DATASET_NAME/ \
#   --overwrite_output_dir

# CUDA_VISIBLE_DEVICES=0 python run_nlu.py \
#   --model_name_or_path microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract \
#   --dataset_name $DATASET_NAME \
#   --do_train \
#   --do_eval \
#   --logging_steps 10 \
#   --max_seq_length 512 \
#   --per_device_train_batch_size 8 \
#   --gradient_accumulation_steps 1 \
#   --learning_rate 5e-5 \
#   --num_train_epochs 80 \
#   --evaluation_strategy epoch \
#   --output_dir ./save/pubmed-bert-$DATASET_NAME/ \
#   --overwrite_output_dir

# CUDA_VISIBLE_DEVICES=0 python run_nlu.py \
#   --model_name_or_path emilyalsentzer/Bio_ClinicalBERT \
#   --dataset_name $DATASET_NAME \
#   --do_train \
#   --do_eval \
#   --logging_steps 10 \
#   --max_seq_length 512 \
#   --per_device_train_batch_size 8 \
#   --gradient_accumulation_steps 1 \
#   --learning_rate 5e-5 \
#   --num_train_epochs 80 \
#   --evaluation_strategy epoch \
#   --output_dir ./save/clinical-bert-$DATASET_NAME/ \
#   --overwrite_output_dir

# # Bio-LM
# CUDA_VISIBLE_DEVICES=0 python run_nlu.py \
#   --model_name_or_path bio-lm \
#   --dataset_name $DATASET_NAME \
#   --do_train \
#   --do_eval \
#   --logging_steps 10 \
#   --max_seq_length 514 \
#   --per_device_train_batch_size 8 \
#   --gradient_accumulation_steps 1 \
#   --learning_rate 5e-5 \
#   --num_train_epochs 80 \
#   --evaluation_strategy epoch \
#   --output_dir ./save/bio-lm-514-$DATASET_NAME/ \
#   --overwrite_output_dir

# CUDA_VISIBLE_DEVICES=0 python run_nlu.py \
#   --model_name_or_path bio-lm \
#   --dataset_name $DATASET_NAME \
#   --do_train \
#   --do_eval \
#   --logging_steps 10 \
#   --max_seq_length 1028 \
#   --per_device_train_batch_size 4 \
#   --gradient_accumulation_steps 2 \
#   --learning_rate 5e-5 \
#   --num_train_epochs 80 \
#   --evaluation_strategy epoch \
#   --output_dir ./save/bio-lm-1028-$DATASET_NAME/ \
#   --overwrite_output_dir

# CUDA_VISIBLE_DEVICES=0 python run_nlu.py \
#   --model_name_or_path bio-lm \
#   --dataset_name $DATASET_NAME \
#   --do_train \
#   --do_eval \
#   --logging_steps 10 \
#   --max_seq_length 2056 \
#   --per_device_train_batch_size 4 \
#   --gradient_accumulation_steps 2 \
#   --learning_rate 5e-5 \
#   --num_train_epochs 80 \
#   --evaluation_strategy epoch \
#   --output_dir ./save/bio-lm-2056-$DATASET_NAME/ \
#   --overwrite_output_dir

# CUDA_VISIBLE_DEVICES=0 python run_nlu.py \
#   --model_name_or_path bio-lm \
#   --dataset_name $DATASET_NAME \
#   --do_train \
#   --do_eval \
#   --logging_steps 10 \
#   --max_seq_length 4112 \
#   --per_device_train_batch_size 2 \
#   --gradient_accumulation_steps 4 \
#   --learning_rate 5e-5 \
#   --num_train_epochs 80 \
#   --evaluation_strategy epoch \
#   --output_dir ./save/bio-lm-4112-$DATASET_NAME/ \
#   --overwrite_output_dir

# CUDA_VISIBLE_DEVICES=0 python run_nlu.py \
#   --model_name_or_path bio-lm \
#   --dataset_name $DATASET_NAME \
#   --do_train \
#   --do_eval \
#   --logging_steps 10 \
#   --max_seq_length 8224 \
#   --per_device_train_batch_size 1 \
#   --gradient_accumulation_steps 8 \
#   --learning_rate 5e-5 \
#   --num_train_epochs 80 \
#   --evaluation_strategy epoch \
#   --output_dir ./save/bio-lm-9224-$DATASET_NAME/ \
#   --overwrite_output_dir
  
# Bioelectra
CUDA_VISIBLE_DEVICES=0 python run_nlu.py \
  --model_name_or_path kamalkraj/bioelectra-base-discriminator-pubmed \
  --dataset_name $DATASET_NAME \
  --do_train \
  --do_eval \
  --logging_steps 10 \
  --max_seq_length 512 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --learning_rate 5e-5 \
  --num_train_epochs 80 \
  --evaluation_strategy epoch \
  --output_dir ./save/bioelectra-512-lr5e-5-$DATASET_NAME/ \
  --overwrite_output_dir

CUDA_VISIBLE_DEVICES=0 python run_nlu.py \
  --model_name_or_path kamalkraj/bioelectra-base-discriminator-pubmed \
  --dataset_name $DATASET_NAME \
  --do_train \
  --do_eval \
  --logging_steps 10 \
  --max_seq_length 1024 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --learning_rate 5e-5 \
  --num_train_epochs 80 \
  --evaluation_strategy epoch \
  --output_dir ./save/bioelectra-1024-lr5e-5-$DATASET_NAME/ \
  --overwrite_output_dir

CUDA_VISIBLE_DEVICES=0 python run_nlu.py \
  --model_name_or_path kamalkraj/bioelectra-base-discriminator-pubmed \
  --dataset_name $DATASET_NAME \
  --do_train \
  --do_eval \
  --logging_steps 10 \
  --max_seq_length 2048 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --learning_rate 5e-5 \
  --num_train_epochs 80 \
  --evaluation_strategy epoch \
  --output_dir ./save/bioelectra-2048-lr5e-5-$DATASET_NAME/ \
  --overwrite_output_dir

CUDA_VISIBLE_DEVICES=0 python run_nlu.py \
  --model_name_or_path kamalkraj/bioelectra-base-discriminator-pubmed \
  --dataset_name $DATASET_NAME \
  --do_train \
  --do_eval \
  --logging_steps 10 \
  --max_seq_length 4096 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-5 \
  --num_train_epochs 80 \
  --evaluation_strategy epoch \
  --output_dir ./save/bioelectra-4096-lr5e-5-$DATASET_NAME/ \
  --overwrite_output_dir

CUDA_VISIBLE_DEVICES=0 python run_nlu.py \
  --model_name_or_path kamalkraj/bioelectra-base-discriminator-pubmed \
  --dataset_name $DATASET_NAME \
  --do_train \
  --do_eval \
  --logging_steps 10 \
  --max_seq_length 8192 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-5 \
  --num_train_epochs 80 \
  --evaluation_strategy epoch \
  --output_dir ./save/bioelectra-8192-lr5e-5-$DATASET_NAME/ \
  --overwrite_output_dir