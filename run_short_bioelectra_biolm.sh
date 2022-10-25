export TOKENIZERS_PARALLELISM=false

# n2c2 2006
export DATASET_NAME=n2c2_2006_smokers

# CUDA_VISIBLE_DEVICES=0 python run_nlu.py \
#   --model_name_or_path bio-lm \
#   --dataset_name $DATASET_NAME \
#   --do_train \
#   --do_eval \
#   --logging_steps 10 \
#   --max_seq_length 512 \
#   --per_device_train_batch_size 8 \
#   --gradient_accumulation_steps 1 \
#   --learning_rate 1e-5 \
#   --num_train_epochs 50 \
#   --evaluation_strategy epoch \
#   --output_dir ./save/bio-lm-512-$DATASET_NAME/ \
#   --overwrite_output_dir

CUDA_VISIBLE_DEVICES=0 python run_nlu.py \
  --model_name_or_path kamalkraj/bioelectra-base-discriminator-pubmed \
  --dataset_name $DATASET_NAME \
  --do_train \
  --do_eval \
  --logging_steps 10 \
  --max_seq_length 512 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-5 \
  --num_train_epochs 50 \
  --evaluation_strategy epoch \
  --output_dir ./save/bioelectra-512-$DATASET_NAME/ \
  --overwrite_output_dir
  
# n2c2 2008 textual
export DATASET_NAME=n2c2_2008_textual

CUDA_VISIBLE_DEVICES=0 python run_nlu.py \
  --model_name_or_path bio-lm \
  --dataset_name $DATASET_NAME \
  --do_train \
  --do_eval \
  --logging_steps 10 \
  --max_seq_length 512 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-5 \
  --num_train_epochs 50 \
  --evaluation_strategy epoch \
  --output_dir ./save/bio-lm-512-$DATASET_NAME/ \
  --overwrite_output_dir

CUDA_VISIBLE_DEVICES=0 python run_nlu.py \
  --model_name_or_path kamalkraj/bioelectra-base-discriminator-pubmed \
  --dataset_name $DATASET_NAME \
  --do_train \
  --do_eval \
  --logging_steps 10 \
  --max_seq_length 512 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-5 \
  --num_train_epochs 50 \
  --evaluation_strategy epoch \
  --output_dir ./save/bioelectra-512-$DATASET_NAME/ \
  --overwrite_output_dir

# n2c2 2008 intuitive
export DATASET_NAME=n2c2_2008_intuitive

CUDA_VISIBLE_DEVICES=0 python run_nlu.py \
  --model_name_or_path bio-lm \
  --dataset_name $DATASET_NAME \
  --do_train \
  --do_eval \
  --logging_steps 10 \
  --max_seq_length 512 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-5 \
  --num_train_epochs 50 \
  --evaluation_strategy epoch \
  --output_dir ./save/bio-lm-512-$DATASET_NAME/ \
  --overwrite_output_dir

CUDA_VISIBLE_DEVICES=0 python run_nlu.py \
  --model_name_or_path kamalkraj/bioelectra-base-discriminator-pubmed \
  --dataset_name $DATASET_NAME \
  --do_train \
  --do_eval \
  --logging_steps 10 \
  --max_seq_length 512 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-5 \
  --num_train_epochs 50 \
  --evaluation_strategy epoch \
  --output_dir ./save/bioelectra-512-$DATASET_NAME/ \
  --overwrite_output_dir
  
# n2c2 2018
export DATASET_NAME=n2c2_2018_track1

CUDA_VISIBLE_DEVICES=0 python run_nlu.py \
  --model_name_or_path bio-lm \
  --dataset_name $DATASET_NAME \
  --do_train \
  --do_eval \
  --logging_steps 10 \
  --max_seq_length 512 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-5 \
  --num_train_epochs 80 \
  --evaluation_strategy epoch \
  --output_dir ./save/bio-lm-512-$DATASET_NAME/ \
  --overwrite_output_dir

CUDA_VISIBLE_DEVICES=0 python run_nlu.py \
  --model_name_or_path kamalkraj/bioelectra-base-discriminator-pubmed \
  --dataset_name $DATASET_NAME \
  --do_train \
  --do_eval \
  --logging_steps 10 \
  --max_seq_length 512 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-5 \
  --num_train_epochs 80 \
  --evaluation_strategy epoch \
  --output_dir ./save/bioelectra-512-$DATASET_NAME/ \
  --overwrite_output_dir