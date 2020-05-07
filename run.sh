#!/usr/bin/env bash

action=$1
cuda=$2
epochs=$3

if [ "$action" == "nmt_inspired" ]; then
  echo python src/training/nmt_inspired_training.py train \
    --arg_file="settings/nmt_eval_args.json" --cuda="${cuda}" \
    --epochs="$epochs" --n_batch=64 --n_emb=100 --n_negs=20 \
    --init_lr=1e-3 --no_hdn=False --ss=1e-4 --window=5

  python src/training/nmt_inspired_training.py train \
    --arg_file="settings/nmt_eval_args.json" --cuda="${cuda}" \
    --epochs="$epochs" --n_batch=64 --n_emb=100 --n_negs=20 \
    --init_lr=1e-3 --no_hdn=False --ss=1e-4 --window=5
fi

if [ "$action" == "pvdm" ]; then
  echo python src/training/pvdm_training.py \
    --cuda="${cuda}" --data_args="settings/pvdm_eval_args.json" \
    --epochs="$epochs" --n_batch=1024 --n_emb=200 --n_negs=20 \
    --init_lr=5e-3 --no_hdn=False --ss=1e-3 --window=5

  python src/training/pvdm_training.py \
    --cuda="${cuda}" --data_args="settings/pvdm_eval_args.json" \
    --epochs="$epochs" --n_batch=1024 --n_emb=200 --n_negs=20 \
    --init_lr=5e-3 --no_hdn=False --ss=1e-3 --window=5
fi
