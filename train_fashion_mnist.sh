#!/bin/bash
poetry run python3 ./ddpm/train_fashion_mnist.py --batch_size=128 --epochs=100 --checkpointing_epochs=5 --outdir=./fashionmnist1