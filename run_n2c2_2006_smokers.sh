export DATASET_NAME=n2c2_2006_smokers

TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES=0 python run_nlu.py \
  --model_name_or_path kamalkraj/bioelectra-base-discriminator-pubmed \
  --dataset_name $DATASET_NAME \
  --do_train \
  --do_eval \
  --logging_steps 10 \
  --max_seq_length 512 \
  --per_device_train_batch_size 32 \
  --learning_rate 1e-5 \
  --num_train_epochs 100 \
  --evaluation_strategy epoch \
  --output_dir ./save/$DATASET_NAME/ \
  --overwrite_output_dir