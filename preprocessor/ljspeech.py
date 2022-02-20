import os
import random
import json

import tgt
import librosa
import numpy as np
from tqdm import tqdm

import audio as Audio
from text import text_to_sequence
from utils.tools import save_mel_and_audio

random.seed(1234)


class Preprocessor:
    def __init__(self, config):
        self.dataset = config["dataset"]
        self.in_dir = config["path"]["corpus_path"]
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

    def build_from_path(self):
        os.makedirs((os.path.join(self.out_dir, "text")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "mel")), exist_ok=True)

        print("Processing Data ...")
        out = list()
        train = list()
        val = list()
        n_frames = 0
        mel_min = float('inf')
        mel_max = -float('inf')

        speakers = {self.dataset: 0}
        with open(os.path.join(self.in_dir, "metadata.csv"), encoding="utf-8") as f:
            for line in tqdm(f.readlines()):
                parts = line.strip().split("|")
                basename = parts[0]
                text = parts[2]

                wav_path = os.path.join(self.in_dir, "wavs", "{}.wav".format(basename))

                ret = self.process_utterance(text, wav_path, self.dataset, basename)
                if ret is None:
                    continue
                else:
                    info, n, m_min, m_max = ret

                if self.val_prior is not None:
                    if basename not in self.val_prior:
                        train.append(info)
                    else:
                        val.append(info)
                else:
                    out.append(info)

                if mel_min > m_min:
                    mel_min = m_min
                if mel_max < m_max:
                    mel_max = m_max

                n_frames += n

        # Save files
        with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
            f.write(json.dumps(speakers))

        with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
            stats = {
                "mel": [
                    float(mel_min),
                    float(mel_max),
                ],
            }
            f.write(json.dumps(stats))

        print(
            "Total time: {} hours".format(
                n_frames * self.hop_length / self.sampling_rate / 3600
            )
        )

        if self.val_prior is not None:
            assert len(out) == 0
            random.shuffle(train)
            train = [r for r in train if r is not None]
            val = [r for r in val if r is not None]
        else:
            assert len(train) == 0 and len(val) == 0
            random.shuffle(out)
            out = [r for r in out if r is not None]
            train = out[self.val_size :]
            val = out[: self.val_size]

        # Write metadata
        with open(os.path.join(self.out_dir, "train.txt"), "w", encoding="utf-8") as f:
            for m in train:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "val.txt"), "w", encoding="utf-8") as f:
            for m in val:
                f.write(m + "\n")

        return out

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

    def process_utterance(self, raw_text, wav_path, speaker, basename):

        # Preprocess text
        text = np.array(text_to_sequence(raw_text, self.cleaners))

        # Load and process wav files
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

        # Save files
        text_filename = "{}-text-{}.npy".format(speaker, basename)
        np.save(
            os.path.join(self.out_dir, "text", text_filename),
            text,
        )
        mel_filename = "{}-mel-{}.npy".format(speaker, basename)
        np.save(
            os.path.join(self.out_dir, "mel", mel_filename),
            mel_spectrogram.T,
        )

        return (
            "|".join([basename, speaker, raw_text]),
            mel_spectrogram.shape[1],
            np.min(mel_spectrogram),
            np.max(mel_spectrogram),
        )
