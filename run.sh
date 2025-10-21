#!/bin/bash

set -eu

lang=$1
model=$2
model_path_name=${model#*/}
# python run.py vocab --lang ${lang} --tagger hexa
for r in 1 2 3 4 ; do
  python run.py train --lang ${lang} --max-depth 6 --tagger hexa --model bert --epochs 50 --batch-size 32 --lr 2e-5 \
    --model-path ${model} \
    --output-path checkpoints/${lang}-${model_path_name}.$r --use-tensorboard False &> checkpoints/log.${lang}-${model_path_name}.$r
  python run.py evaluate --lang ${lang} --max-depth 10 --tagger hexa \
    --bert-model-path ${model} --model-name ${lang}-hexa-bert-2e-05-50 --batch-size 64 \
    --model-path checkpoints/${lang}-${model_path_name}.$r/ &>> checkpoints/log.${lang}-${model_path_name}.$r
done
