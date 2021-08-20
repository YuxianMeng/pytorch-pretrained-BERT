export GLUE_DIR=/userhome/yuxian/data/glue_data  # glue data directory
export TASK_NAME=SST-2

# Train
CUDA_VISIBLE_DEVICES=0, python run_glue.py \
  --model_name_or_path /userhome/yuxian/data/bert/bert-base-uncased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_device_train_batch_size 64 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir /userhome/yuxian/train_logs/glue/${TASK_NAME}-baseline/


# create attacked valid set  NOTE: SST do not provide test set, so we only evaluate on valid set
python attack_sst_data.py \
--origin-dir $GLUE_DIR/$TASK_NAME \
--out-dir $GLUE_DIR/$TASK_NAME-attack \
--subsets dev \
--max-pos 100



### Train attacked model
BERT_MODEL=/userhome/yuxian/data/bert/bert-base-uncased-attacked-random-new  # pretrained attacked model
OUT_MODEL=/userhome/yuxian/train_logs/glue/${TASK_NAME}-attacked-random-new  # where to save your finetune model

CUDA_VISIBLE_DEVICES=0, python run_glue.py \
  --model_name_or_path $BERT_MODEL \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_device_train_batch_size 64 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir $OUT_MODEL


# Eval on attack/unattacked data  NOTE: SST do not provide test set, so we only evaluate on valid set
MODEL_DIR=/userhome/yuxian/train_logs/glue/${TASK_NAME}-attacked-random-new  # finetuned model dir
DATA_DIR=$GLUE_DIR/$TASK_NAME-attack  # attacked dataset
#DATA_DIR=$GLUE_DIR/$TASK_NAME  # normal dataset

CUDA_VISIBLE_DEVICES=0, python run_glue.py \
  --model_name_or_path $MODEL_DIR\
  --task_name $TASK_NAME \
  --do_eval \
  --data_dir $DATA_DIR \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir /userhome/yuxian/train_logs/glue/debug/

