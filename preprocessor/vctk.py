import os
import random
import json

import re
import tgt
import librosa
import numpy as np
from tqdm import tqdm
from pathlib import Path

import audio as Audio
from text import text_to_sequence
from model import PreDefinedEmbedder
from utils.tools import save_mel_and_audio, plot_embedding

random.seed(1234)


class Preprocessor:
    def __init__(self, config):
        self.dataset = config["dataset"]
        self.in_dir = config["path"]["corpus_path"]
        self.wav_tag = config["path"]["wav_tag"]
        self.wav_dir = config["path"]["wav_dir"]
        self.txt_dir = config["path"]["txt_dir"]
        self.out_dir = config["path"]["preprocessed_path"]
        self.val_size = config["preprocessing"]["val_size"]
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.skip_len = config["preprocessing"]["audio"]["skip_len"]
        self.trim_top_db = config["preprocessing"]["audio"]["trim_top_db"]
        self.filter_length = config["preprocessing"]["stft"]["filter_length"]
        self.hop_length = config["preprocessing"]["stft"]["hop_length"]
        self.silence_audio_size = config["preprocessing"]["audio"]["silence_audio_size"]
        self.pre_emphasis = config["preprocessing"]["audio"]["pre_emphasis"]
        self.max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
        self.sanity_check = config["preprocessing"]["sanity_check"]
        self.cleaners = config["preprocessing"]["text"]["text_cleaners"]

        self.STFT = Audio.stft.TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )
        self.val_prior = self.val_prior_names(os.path.join(self.out_dir, "val.txt"))
        self.speaker_emb = None
        if config["preprocessing"]["speaker_embedder"] != "none":
            self.speaker_emb = PreDefinedEmbedder(config)

    def val_prior_names(self, val_prior_path):
        val_prior_names = set()
        if os.path.isfile(val_prior_path):
            print("Load pre-defined validation set...")
            with open(val_prior_path, "r", encoding="utf-8") as f:
                for m in f.readlines():
                    val_prior_names.add(m.split("|")[0])
            return list(val_prior_names)
        else:
            return None

    def _init_spker_embeds(self, spkers):
        spker_embeds = dict()
        for spker in spkers:
            spker_embeds[spker] = list()
        return spker_embeds

    def build_from_path(self):
        txt_dir = os.path.join(self.in_dir, self.txt_dir)
        wav_dir = os.path.join(self.in_dir, self.wav_dir)
        embedding_dir = os.path.join(self.out_dir, "spker_embed")
        os.makedirs((os.path.join(self.out_dir, "text")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "mel")), exist_ok=True)
        os.makedirs(embedding_dir, exist_ok=True)

        print("Processing Data ...")
        # out = list()
        train = list()
        val = list()
        n_frames = 0
        mel_min = float('inf')
        mel_max = -float('inf')
        max_seq_len = -float('inf')
        sub_dirs = os.listdir(txt_dir)
        if self.speaker_emb is not None:
            spker_embeds = self._init_spker_embeds(sub_dirs)

        skip_speakers = set()
        preprocessed_dirs = os.listdir(embedding_dir)
        for preprocessed_dir in preprocessed_dirs:
            skip_speakers.add(preprocessed_dir.split("-")[0])

        speakers = {}
        for spker_id, speaker in enumerate(tqdm(sub_dirs)):
            speakers[speaker] = spker_id
            for i, txt_name in enumerate(tqdm(os.listdir(os.path.join(txt_dir, speaker)))):

                basename = txt_name.split(".")[0]
                with open(os.path.join(txt_dir, speaker, txt_name), "r") as f:
                    text = f.readline().strip("\n")

                wav_path = os.path.join(os.path.join(wav_dir, speaker), "{}_{}.flac".format(basename, self.wav_tag))
                if not os.path.isfile(wav_path):
                    print("[Error] No flac file:{}".format(wav_path))
                    continue
                ret = self.process_utterance(text, wav_path, speaker, basename, skip_speakers)
                if ret is None:
                    continue
                else:
                    info, n, m_min, m_max, spker_embed = ret
                # out.append(info)

                if self.val_prior is not None:
                    if basename not in self.val_prior:
                        train.append(info)
                    else:
                        val.append(info)
                else:
                    if i == 0 or i == 1: 
                        val.append(info)
                    else:
                        train.append(info)

                if self.speaker_emb is not None:
                    spker_embeds[speaker].append(spker_embed)

                if mel_min > m_min:
                    mel_min = m_min
                if mel_max < m_max:
                    mel_max = m_max
                if n > max_seq_len:
                    max_seq_len = n

                n_frames += n

            # Calculate and save mean speaker embedding of this speaker
            if self.speaker_emb is not None and speaker not in skip_speakers:
                spker_embed_filename = '{}-spker_embed.npy'.format(speaker)
                np.save(os.path.join(self.out_dir, 'spker_embed', spker_embed_filename), \
                    np.mean(spker_embeds[speaker], axis=0), allow_pickle=False)

        # Save files
        with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
            f.write(json.dumps(speakers))

        with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
            stats = {
                "mel": [
                    float(mel_min),
                    float(mel_max),
                ],
                "max_seq_len": max_seq_len
            }
            f.write(json.dumps(stats))

        print(
            "Total time: {} hours".format(
                n_frames * self.hop_length / self.sampling_rate / 3600
            )
        )

        print("plot speaker embedding...")
        plot_embedding(
            self.out_dir, *self.load_embedding(embedding_dir),
            self.divide_speaker_by_gender(self.in_dir), filename="spker_embed_tsne.png"
        )

        # random.shuffle(out)
        # out = [r for r in out if r is not None]
        if self.val_prior is None:
            random.shuffle(train)
        train = [r for r in train if r is not None]
        val = [r for r in val if r is not None]

        # Write metadata
        with open(os.path.join(self.out_dir, "train.txt"), "w", encoding="utf-8") as f:
            for m in train:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "val.txt"), "w", encoding="utf-8") as f:
            for m in val:
                f.write(m + "\n")

        return train, val #out

    def load_audio(self, wav_path):
        wav_raw, _ = librosa.load(wav_path, self.sampling_rate)
        if len(wav_raw) < self.skip_len:
            return None
        wav = wav_raw / np.abs(wav_raw).max() * 0.999
        wav = librosa.effects.trim(wav, top_db=self.trim_top_db, frame_length=self.filter_length, hop_length=self.hop_length)[0]
        if self.pre_emphasis:
            wav = np.append(wav[0], wav[1:] - 0.97 * wav[:-1])
            wav = wav / np.abs(wav).max() * 0.999
        wav = np.append(wav, [0.] * self.hop_length * self.silence_audio_size)
        wav = wav.astype(np.float32)
        return wav_raw, wav
 
    def process_utterance(self, raw_text, wav_path, speaker, basename, skip_speakers):
        text_filename = "{}-text-{}.npy".format(speaker, basename)
        mel_filename = "{}-mel-{}.npy".format(speaker, basename)

        # Preprocess text
        if not os.path.isfile(os.path.join(self.out_dir, "text", text_filename)):
            text = np.array(text_to_sequence(raw_text, self.cleaners))
            np.save(
                os.path.join(self.out_dir, "text", text_filename),
                text,
            )
        else:
            text = np.load(os.path.join(self.out_dir, "text", text_filename))

        # Load and process wav files
        wav_raw = wav = None
        if not os.path.isfile(os.path.join(self.out_dir, "mel", mel_filename)):

            wav_raw, wav = self.load_audio(wav_path)

            # Compute mel-scale spectrogram
            mel_spectrogram = Audio.tools.get_mel_from_wav(wav, self.STFT)

            # Sanity check
            if self.sanity_check:
                save_mel_and_audio(mel_spectrogram, wav*self.max_wav_value,
                    self.sampling_rate, self.out_dir, basename, tag="processed"
                )
                save_mel_and_audio(Audio.tools.get_mel_from_wav(wav_raw, self.STFT), wav_raw*self.max_wav_value,
                    self.sampling_rate, self.out_dir, basename, tag="raw"
                )
                exit(0) # quit for testing

            np.save(
                os.path.join(self.out_dir, "mel", mel_filename),
                mel_spectrogram.T,
            )
        else:
            mel_spectrogram = np.load(os.path.join(self.out_dir, "mel", mel_filename)).T

        # Speaker embedding
        spker_embed=None
        if self.speaker_emb is not None and speaker not in skip_speakers:
            if wav is not None:
                spker_embed = self.speaker_emb(wav)
            else:
                wav_raw, wav = self.load_audio(wav_path)
                spker_embed = self.speaker_emb(wav)

        return (
            "|".join([basename, speaker, raw_text]),
            mel_spectrogram.shape[1],
            np.min(mel_spectrogram),
            np.max(mel_spectrogram),
            spker_embed,
        )

    def divide_speaker_by_gender(self, in_dir, speaker_path="speaker-info.txt"):
        speakers = dict()
        with open(os.path.join(in_dir, speaker_path), encoding='utf-8') as f:
            for line in tqdm(f):
                if "ID" in line: continue
                parts = [p.strip() for p in re.sub(' +', ' ',(line.strip())).split(' ')]
                spk_id, gender = parts[0], parts[2]
                speakers[str(spk_id)] = gender
        return speakers

    def load_embedding(self, embedding_dir):
        embedding_path_list = [_ for _ in Path(embedding_dir).rglob('*.npy')]
        embedding = None
        embedding_speaker_id = list()
        # Gather data
        for path in tqdm(embedding_path_list):
            embedding = np.concatenate((embedding, np.load(path)), axis=0) \
                                            if embedding is not None else np.load(path)
            embedding_speaker_id.append(str(str(path).split('/')[-1].split('-')[0]))
        return embedding, embedding_speaker_id
