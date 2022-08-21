export DATASET_NAME=n2c2_2018_track1

TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES=0 python run_nlu.py \
  --model_name_or_path bert-base-cased \
  --dataset_name $DATASET_NAME \
  --do_train \
  --do_eval \
  --logging_steps 10 \
  --max_seq_length 512 \
  --per_device_train_batch_size 32 \
  --learning_rate 5e-5 \
  --num_train_epochs 20 \
  --output_dir ./save/$DATASET_NAME/ \
  --overwrite_output_dir