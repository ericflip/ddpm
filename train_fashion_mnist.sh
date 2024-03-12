#!/bin/bash
poetry run python3 ./ddpm/train.py \
    --batch_size=128 \
    --epochs=100 \
    --checkpoint_epochs=10 \
    --outdir=./ddpm_fashionmnist \
    --lr=2e-4 \
    --timesteps=1000
