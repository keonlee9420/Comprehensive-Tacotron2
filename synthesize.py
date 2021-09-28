import os
import json
# import re
import argparse
# from string import punctuation

import torch
import numpy as np
from torch.utils.data import DataLoader
# from g2p_en import G2p
# from pypinyin import pinyin, Style

from utils.model import get_model, get_vocoder
from utils.tools import get_configs_of, to_device, infer_one_sample #, read_lexicon
from dataset import TextDataset
from text import text_to_sequence, sequence_to_text

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_english(text, preprocess_config):
    sequence = text_to_sequence(
        text, preprocess_config["preprocessing"]["text"]["text_cleaners"]
    )
    print("Raw Text Sequence: {}".format(text))
    print("Sequence: {}".format(" ".join([str(id) for id in sequence_to_text(sequence)])))
    print("Sequence Input: {}".format(" ".join([str(id) for id in sequence])))
    return np.array(sequence)


def synthesize(model, args, configs, mel_stats, vocoder, batchs):
    preprocess_config, model_config, train_config = configs

    for batch in batchs:
        batch = to_device(batch, device, mel_stats)
        with torch.no_grad():
            # Forward
            output = model.inference(*batch[2:4], *batch[5:7])
            infer_one_sample(
                batch,
                output,
                vocoder,
                mel_stats,
                model_config,
                preprocess_config,
                train_config["path"]["result_path"],
                args,
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "single"],
        required=True,
        help="Synthesize a whole dataset or a single sentence",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="path to a source file with format like train.txt and val.txt, for batch mode only",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="raw text to synthesize, for single-sentence mode only",
    )
    parser.add_argument(
        "--speaker_id",
        type=str,
        default="p225",
        help="speaker ID for multi-speaker synthesis, for single-sentence mode only",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="name of dataset",
    )
    args = parser.parse_args()

    # Check source texts
    if args.mode == "batch":
        assert args.source is not None and args.text is None
    if args.mode == "single":
        assert args.source is None and args.text is not None

    # Read Config
    preprocess_config, model_config, train_config = get_configs_of(args.dataset)
    configs = (preprocess_config, model_config, train_config)
    with open(
            os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
    ) as f:
        stats = json.load(f)
        mel_stats = stats["mel"]
    os.makedirs(
        os.path.join(train_config["path"]["result_path"], str(args.restore_step)), exist_ok=True)

    # Get model
    model = get_model(args, configs, device, train=False)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Preprocess texts
    if args.mode == "batch":
        # Get dataset
        dataset = TextDataset(args.source, preprocess_config, model_config)
        batchs = DataLoader(
            dataset,
            batch_size=1, # currently only 1 is supported
            collate_fn=dataset.collate_fn,
        )
    if args.mode == "single":
        ids = raw_texts = [args.text[:100]]

        # Speaker Info
        load_spker_embed = model_config["multi_speaker"] \
            and preprocess_config["preprocessing"]["speaker_embedder"] != 'none'
        with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "speakers.json")) as f:
            speaker_map = json.load(f)
        speakers = np.array([speaker_map[args.speaker_id]]) if model_config["multi_speaker"] else np.array([0]) # single speaker is allocated 0
        spker_embed = np.load(os.path.join(
            preprocess_config["path"]["preprocessed_path"],
            "spker_embed",
            "{}-spker_embed.npy".format(args.speaker_id),
        )) if load_spker_embed else None

        if preprocess_config["preprocessing"]["text"]["language"] == "en":
            texts = np.array([preprocess_english(args.text, preprocess_config)])
        else:
            raise NotImplementedError
        text_lens = np.array([len(texts[0])])
        batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens), spker_embed)]

    synthesize(model, args, configs, mel_stats, vocoder, batchs)
