export DATASET_NAME=n2c2_2018_track1

TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES=5 python run_nlu.py \
  --model_name_or_path bert-base-cased \
  --dataset_name $DATASET_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir ./save/$DATASET_NAME/ \
  --overwrite_output_dir