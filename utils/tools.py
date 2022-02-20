import re
import os
import json
import yaml

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
from scipy.io import wavfile
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import audio as Audio


matplotlib.use("Agg")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_configs_of(dataset):
    config_dir = os.path.join("./config", dataset)
    preprocess_config = yaml.load(open(
        os.path.join(config_dir, "preprocess.yaml"), "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(
        os.path.join(config_dir, "model.yaml"), "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(
        os.path.join(config_dir, "train.yaml"), "r"), Loader=yaml.FullLoader)
    return preprocess_config, model_config, train_config


def to_device(data, device, mel_stats=None):
    if len(data) == 12:
        (
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            r_len_pad,
            gates,
            spker_embeds,
        ) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)
        mels = torch.from_numpy(mels).float().to(device)
        if mel_stats is not None:
            mels = Audio.tools.mel_normalize(mels, *mel_stats)
        mel_lens = torch.from_numpy(mel_lens).to(device)
        gates = torch.from_numpy(gates).float().to(device)
        if spker_embeds is not None:
            spker_embeds = torch.from_numpy(spker_embeds).float().to(device)

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            r_len_pad,
            gates,
            spker_embeds,
        )

    if len(data) == 7:
        (ids, raw_texts, speakers, texts, src_lens, max_src_len, spker_embeds) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)
        if spker_embeds is not None:
            spker_embeds = torch.from_numpy(spker_embeds).float().to(device)

        return (ids, raw_texts, speakers, texts, src_lens, max_src_len, spker_embeds)


def log(
    logger, step=None, losses=None, grad_norm=None, fig=None, audio=None, sampling_rate=22050, tag=""
):
    if losses is not None:
        logger.add_scalar("Loss/total_loss", losses[0], step)
        logger.add_scalar("Loss/mel_loss", losses[1], step)
        logger.add_scalar("Loss/gate_loss", losses[2], step)
        logger.add_scalar("Loss/attn_loss", losses[3], step)

    if grad_norm is not None:
        logger.add_scalar("Grad/grad_norm", grad_norm, step)

    if fig is not None:
        logger.add_figure(tag, fig, step)

    if audio is not None:
        logger.add_audio(
            tag,
            audio / max(abs(audio)),
            sample_rate=sampling_rate,
        )


def get_mask_from_lengths(lengths, max_len=-1):
    max_len = max(torch.max(lengths).item(), max_len)
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1))
    return mask


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def synth_one_sample(targets, predictions, vocoder, mel_stats, model_config, preprocess_config, step):

    normalize = preprocess_config["preprocessing"]["mel"]["normalize"]
    n_frames_per_step = model_config["decoder"]["n_frames_per_step"]
    basename = targets[0][0]
    src_len = targets[4][0].item()
    mel_len = targets[7][0].item()
    reduced_mel_len = mel_len // n_frames_per_step
    mel_target = targets[6][0, :mel_len].detach().transpose(0, 1)
    mel_prediction = predictions[1][0, :mel_len].detach().transpose(0, 1)
    attention = predictions[3][0, :reduced_mel_len, :src_len].detach().transpose(0, 1) # [seq_len, mel_len]
    gate_target = targets[10][0].detach()
    gate_prediction = predictions[2][0].detach()

    if normalize:
        mel_target = Audio.tools.mel_denormalize(mel_target, *mel_stats)
        mel_prediction = Audio.tools.mel_denormalize(mel_prediction, *mel_stats)

    fig = plot_mel(
        [
            mel_prediction.cpu().numpy(),
            mel_target.cpu().numpy(),
            attention.cpu().numpy(),
        ],
        ["Synthetized Spectrogram", "Ground-Truth Spectrogram", "Decoder Alignment"],
        attention=True,
    )

    gate_fig, _ = plot_gate_outputs(
        gate_target.cpu().numpy(),
        torch.sigmoid(gate_prediction).cpu().numpy(),
        subtitle="step_{}_{}".format(step, basename)
    )

    if vocoder is not None:
        from .model import vocoder_infer

        wav_reconstruction = vocoder_infer(
            mel_target.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
        wav_prediction = vocoder_infer(
            mel_prediction.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
    else:
        wav_reconstruction = wav_prediction = None

    return fig, gate_fig, wav_reconstruction, wav_prediction, basename


def infer_one_sample(targets, predictions, vocoder, mel_stats, model_config, preprocess_config, path, args):

    normalize = preprocess_config["preprocessing"]["mel"]["normalize"]
    n_frames_per_step = model_config["decoder"]["n_frames_per_step"]
    multi_speaker = model_config["multi_speaker"]
    mel_predictions = predictions[1]
    if normalize:
        mel_predictions = Audio.tools.mel_denormalize(predictions[1], *mel_stats)

    basename = targets[0][0]
    src_len = targets[4][0].item()
    mel_len = mel_predictions[0].shape[0]
    reduced_mel_len = mel_len // n_frames_per_step
    mel_prediction = mel_predictions[0].detach().transpose(0, 1)
    attention = predictions[3][0, :reduced_mel_len, :src_len].detach().transpose(0, 1) # [seq_len, mel_len]

    fig = plot_mel(
        [
            mel_prediction.cpu().numpy(),
            attention.cpu().numpy(),
        ],
        ["Synthetized Spectrogram", "Decoder Alignment"],
        attention=True,
    )
    plt.savefig(os.path.join(
        path, str(args.restore_step), "{}_{}.png".format(basename, args.speaker_id)\
             if multi_speaker and args.mode == "single" else "{}.png".format(basename)))
    plt.close()

    from .model import vocoder_infer

    lengths = mel_len * preprocess_config["preprocessing"]["stft"]["hop_length"]
    wav_predictions = vocoder_infer(
        mel_predictions.transpose(1, 2), vocoder, model_config, preprocess_config, lengths=[lengths]
    )

    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    wavfile.write(os.path.join(
        path, str(args.restore_step), "{}_{}.wav".format(basename, args.speaker_id)\
             if multi_speaker and args.mode == "single" else "{}.wav".format(basename)),
        sampling_rate, wav_predictions[0])


def plot_mel(data, titles, attention=False):
    fig, axes = plt.subplots(len(data), 1, squeeze=False)
    if titles is None:
        titles = [None for i in range(len(data))]

    def add_axis(fig, old_ax):
        ax = fig.add_axes(old_ax.get_position(), anchor="W")
        ax.set_facecolor("None")
        return ax

    for i in range(len(data)):
        if i == len(data)-1 and attention:
            im = axes[i][0].imshow(data[i], origin='lower', aspect='auto')
            axes[i][0].set_xlabel('Audio timestep')
            axes[i][0].set_ylabel('Text timestep')
            axes[i][0].set_xlim(0, data[i].shape[1])
            axes[i][0].set_ylim(0, data[i].shape[0])
            axes[i][0].set_title(titles[i], fontsize="medium")
            axes[i][0].tick_params(labelsize="x-small")
            axes[i][0].set_anchor("W")
            fig.colorbar(im, ax=axes[i][0])
            break
        mel = data[i]
        axes[i][0].imshow(mel, origin="lower")
        axes[i][0].set_aspect(2.5, adjustable="box")
        axes[i][0].set_ylim(0, mel.shape[0])
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[i][0].set_anchor("W")

    return fig


def plot_gate_outputs(gate_targets, gate_outputs, subtitle=None, to_numpy=False):
    data = None
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.scatter(range(len(gate_targets)), gate_targets, alpha=0.5,
               color='green', marker='+', s=1, label='target')
    ax.scatter(range(len(gate_outputs)), gate_outputs, alpha=0.5,
               color='red', marker='.', s=1, label='predicted')

    plt.xlabel("Frames (Green target, Red predicted)")
    plt.ylabel("Gate State")
    plt.tight_layout()

    fig.canvas.draw()
    if subtitle is not None:
        fig.suptitle(subtitle)
    if to_numpy:
        data = save_figure_to_numpy(fig)
    plt.close()
    return fig, data


def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def save_mel_and_audio(spectrogram, audio, sampling_rate, out_dir, basename, tag=None):
    # Save mel
    plt.imshow(spectrogram, origin='lower')
    plt.ylim(0, spectrogram.shape[0])
    plt.tick_params(labelsize='x-small',
                        left=False, labelleft=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '{}_{}.png'.format(basename, tag)\
            if tag is not None else '{}.png'.format(basename)), dpi=200)
    plt.close()

    # Save audio
    wavfile.write(
        os.path.join(out_dir, "{}_{}.wav".format(basename, tag)\
            if tag is not None else '{}.wav'.format(basename)),
        sampling_rate,
        audio.astype(np.int16),
    )


def plot_embedding(out_dir, embedding, embedding_speaker_id, gender_dict, filename='embedding.png'):
    colors = 'r','b'
    labels = 'Female','Male'

    data_x = embedding
    data_y = np.array([gender_dict[spk_id] == 'M' for spk_id in embedding_speaker_id], dtype=np.int)
    tsne_model = TSNE(n_components=2, random_state=0, init='random')
    tsne_all_data = tsne_model.fit_transform(data_x)
    tsne_all_y_data = data_y

    plt.figure(figsize=(10,10))
    for i, (c, label) in enumerate(zip(colors, labels)):
        plt.scatter(tsne_all_data[tsne_all_y_data==i,0], tsne_all_data[tsne_all_y_data==i,1], c=c, label=label, alpha=0.5)

    plt.grid(True)
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename))


def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(
            x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD
        )
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded
