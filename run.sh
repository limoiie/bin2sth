#!/usr/bin/env bash

action=$1

if [ "$action" == "nmt_inspired" ]; then
  echo python src/training/nmt_inspired_training.py train \
    --data_args="settings/nmt_eval_args.json" --cuda=-1 \
    --epochs=1 --n_batch=64 --n_emb=100 --n_negs=20 \
    --init_lr=1e-3 --no_hdn=False --ss=1e-4 --window=5

  python src/training/nmt_inspired_training.py train \
    --data_args="settings/nmt_eval_args.json" --cuda=-1 \
    --epochs=1 --n_batch=64 --n_emb=100 --n_negs=20 \
    --init_lr=1e-3 --no_hdn=False --ss=1e-4 --window=5
fi

if [ "$action" == "pvdm" ]; then
  echo python src/training/pvdm_training.py \
    --cuda=-1 --data_args="settings/pvdm_eval_args.json" \
    --epochs=50 --n_batch=1024 --n_emb=200 --n_negs=20 \
    --init_lr=0.005 --no_hdn=False --ss=0.0001 --window=5

  python python src/training/pvdm_training.py \
    --cuda=-1 --data_args="settings/pvdm_eval_args.json" \
    --epochs=50 --n_batch=1024 --n_emb=200 --n_negs=20 \
    --init_lr=0.005 --no_hdn=False --ss=0.0001 --window=5
fi
