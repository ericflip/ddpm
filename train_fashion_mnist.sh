#!/bin/bash
poetry run python3 ./ddpm/train_fashion_mnist.py \
    --batch_size=128 \
    --epochs=500 \
    --checkpointing_epochs=50 \
    --outdir=./fashionmnist_awesome \
