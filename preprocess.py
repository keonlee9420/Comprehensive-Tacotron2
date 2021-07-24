import argparse

from utils.tools import get_configs_of
from preprocessor import ljspeech, vctk


def main(config):
    if "LJSpeech" in config["dataset"]:
        preprocessor = ljspeech.Preprocessor(config)
    if "VCTK" in config["dataset"]:
        preprocessor = vctk.Preprocessor(config)

    preprocessor.build_from_path()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="name of dataset",
    )
    args = parser.parse_args()

    preprocess_config, *_ = get_configs_of(args.dataset)
    main(preprocess_config)
