#!/bin/bash
poetry run python3 ./ddpm/train_fashion_mnist.py --batch_size=128 --epochs=300 --checkpointing_epochs=25 --outdir=./fashionmnist1