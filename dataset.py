import json
import math
import os

import numpy as np
from torch.utils.data import Dataset

from utils.tools import pad_1D, pad_2D


class Dataset(Dataset):
    def __init__(
        self, filename, preprocess_config, model_config, train_config, sort=False, drop_last=False
    ):
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.batch_size = train_config["optimizer"]["batch_size"]
        self.n_frames_per_step = model_config["decoder"]["n_frames_per_step"]
        self.load_spker_embed = model_config["multi_speaker"] \
            and preprocess_config["preprocessing"]["speaker_embedder"] != 'none'

        self.basename, self.speaker, self.raw_text = self.process_meta(
            filename
        )
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        self.sort = sort
        self.drop_last = drop_last

    def __len__(self):
        return len(self.raw_text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone_path = os.path.join(
            self.preprocessed_path,
            "text",
            "{}-text-{}.npy".format(speaker, basename),
        )
        text = np.load(phone_path)
        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            "{}-mel-{}.npy".format(speaker, basename),
        )
        mel = np.load(mel_path)
        spker_embed = np.load(os.path.join(
            self.preprocessed_path,
            "spker_embed",
            "{}-spker_embed.npy".format(speaker),
        )) if self.load_spker_embed else None

        sample = {
            "id": basename,
            "speaker": speaker_id,
            "text": text,
            "raw_text": raw_text,
            "mel": mel,
            "spker_embed": spker_embed,
        }

        return sample

    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            speaker = []
            text = []
            for line in f.readlines():
                n, s, t = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
            return name, speaker, text

    def get_gates_from_mel_lens(self, mel_lens, max_target_len):
        gates = np.zeros([mel_lens.shape[0], max_target_len])
        for i, mel_len in enumerate(mel_lens):
            gates[i, mel_len-1:] = 1
        return gates

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        mels = [data[idx]["mel"] for idx in idxs]
        spker_embeds = np.concatenate(np.array([data[idx]["spker_embed"] for idx in idxs]), axis=0) \
            if self.load_spker_embed else None

        text_lens = np.array([text.shape[0] for text in texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])

        max_target_len = max(mel_lens)
        r_len_pad = max_target_len % self.n_frames_per_step
        if r_len_pad != 0:
            max_target_len += self.n_frames_per_step - r_len_pad
            assert max_target_len % self.n_frames_per_step == 0

        speakers = np.array(speakers)
        texts = pad_1D(texts)
        mels = pad_2D(mels, max_target_len)
        gates = self.get_gates_from_mel_lens(mel_lens, max_target_len)

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            text_lens,
            max(text_lens),
            mels,
            mel_lens,
            max_target_len,
            r_len_pad,
            gates,
            spker_embeds,
        )

    def collate_fn(self, data):
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output


class TextDataset(Dataset):
    def __init__(self, filepath, preprocess_config, model_config):

        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.load_spker_embed = model_config["multi_speaker"] \
            and preprocess_config["preprocessing"]["speaker_embedder"] != 'none'

        self.basename, self.speaker, self.raw_text = self.process_meta(
            filepath
        )
        with open(
            os.path.join(
                preprocess_config["path"]["preprocessed_path"], "speakers.json"
            )
        ) as f:
            self.speaker_map = json.load(f)

    def __len__(self):
        return len(self.raw_text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone_path = os.path.join(
            self.preprocessed_path,
            "text",
            "{}-text-{}.npy".format(speaker, basename),
        )
        text = np.load(phone_path)
        spker_embed = np.load(os.path.join(
            self.preprocessed_path,
            "spker_embed",
            "{}-spker_embed.npy".format(speaker),
        )) if self.load_spker_embed else None

        return (basename, speaker_id, text, raw_text, spker_embed)

    def process_meta(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            name = []
            speaker = []
            text = []
            for line in f.readlines():
                n, s, t = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
            return name, speaker, text

    def collate_fn(self, data):
        ids = [d[0] for d in data]
        speakers = np.array([d[1] for d in data])
        texts = [d[2] for d in data]
        raw_texts = [d[3] for d in data]
        text_lens = np.array([text.shape[0] for text in texts])
        spker_embeds = np.concatenate(np.array([d[4] for d in data]), axis=0) \
            if self.load_spker_embed else None

        texts = pad_1D(texts)

        return ids, raw_texts, speakers, texts, text_lens, max(text_lens), spker_embeds