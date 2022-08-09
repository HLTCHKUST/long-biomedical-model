export DATASET_NAME=n2c2_2008

TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES=5 python run_glue.py \
  --model_name_or_path bert-base-cased \
  --dataset_name $DATASET_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 256 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-4 \
  --num_train_epochs 5 \
  --output_dir ./save/$DATASET_NAME/ \
  --overwrite_output_dir