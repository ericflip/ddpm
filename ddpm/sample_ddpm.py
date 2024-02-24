import argparse
import os
import uuid

from pipeline import DDPMPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Inference for Fashion MNIST")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="Path to the model checkpoint with config and weights",
    )
    parser.add_argument("--output_dir", type=str, help="Path to save the sampled image")

    return parser.parse_args()


def main(args):
    # make dir for generations
    os.makedirs(args.output_dir, exist_ok=True)

    # initialize DDPM pipeline
    pipe = DDPMPipeline.from_checkpoint(args.checkpoint_path).to("cuda")

    # generate sample
    sample = pipe(num_images=1, image_size=32)[0]

    # save sample
    sample_path = os.path.join(args.output_dir, f"{uuid.uuid4()}.png")

    print(sample_path)

    sample.save(sample_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
