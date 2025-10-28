#!/bin/bash

set -eu

runs=$1
lang=$2
model=$3
if [[ $# -ge 4 ]] ; then
  train_batch_size=$4
else
  train_batch_size=32
fi
model_path_name=${model#*/}
# python run.py vocab --lang ${lang} --tagger hexa
for r in `seq ${runs}` ; do
  echo `date +"%Y%m%d-%H%M%S"` ${lang}-${model_path_name}.$r train
  python run.py train --lang ${lang} --max-depth 6 --tagger hexa --model bert --epochs 50 --batch-size ${train_batch_size} --lr 2e-5 --model-path ${model} --output-path checkpoints/${lang}-${model_path_name}.$r --use-tensorboard False &> log.${lang}-${model_path_name}.$r
  echo `date +"%Y%m%d-%H%M%S"` ${lang}-${model_path_name}.$r evaluate
  python run.py evaluate --lang ${lang} --max-depth 10 --tagger hexa --bert-model-path ${model} --model-name ${lang}-hexa-bert-2e-05-50 --batch-size 64 --model-path checkpoints/${lang}-${model_path_name}.$r/ &>> log.${lang}-${model_path_name}.$r
  echo `date +"%Y%m%d-%H%M%S"` ${lang}-${model_path_name}.$r finished
done
