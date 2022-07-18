import argparse
import os

from datetime import datetime
import numpy as np
import torch
import yaml

import datasets
from datasets import preprocessing
from utils import main_utils


def load_model(checkpoint_path):
    class Args:
        rank = -1
        quiet = False
    args = Args()
    cfg = yaml.safe_load(open("configs/main/avid-cma/kinetics/InstX-N1024-PosW-N64-Top32.yaml"))
    logger, tb_writter, model_dir = main_utils.prep_environment(args, cfg)
    cfg['model']['args']['checkpoint'] = checkpoint_path
    model = main_utils.build_model(cfg['model'], logger)
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict({k.replace('module.', ''): state_dict['model'][k] for k in state_dict['model']})
    model.eval()
    return model


def get_dataloader(subset):
    video_transform = preprocessing.VideoPrep_MSC_CJ(
        crop=(224, 224),
        augment=False,
        num_frames=0.5 * 16,
        pad_missing=True,
    )
    audio_transform = [
        preprocessing.AudioPrep(
            trim_pad=True,
            duration=2,
            augment=False,
            missing_as_zero=True),
        preprocessing.LogSpectrogram(
            24000,
            n_fft=512,
            hop_size=1. / 100,
            normalize=True)
    ]
    dataset = datasets.Kinetics(
        subset=subset,
        return_video=True,
        video_clip_duration=0.5,
        video_fps=16,
        video_transform=video_transform,
        return_audio=True,
        audio_clip_duration=2,
        audio_fps=24000,
        audio_fps_out=64,
        audio_transform=audio_transform,
        max_offsync_augm=0,
        return_labels=True,
        return_index=True,
        mode='clip',
        clips_per_video=10,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
        sampler=None
    )
    return dataloader, len(dataset)


def calculate_embeddings(model, dataloader, len_dataset):
    labels_test = []
    codings_video = []
    codings_audio = []
    for sample in dataloader:
        with torch.no_grad():
            logits = model(sample["frames"], sample["audio"])
            codings_video.append(logits[0])
            codings_audio.append(logits[1])
            labels_test += [l.split("?")[-1] for l in sample["label"]]
            print(f"{np.round(100 * ((len(codings_video) * 64) / len_dataset), 2)}%")
    codings_video = torch.vstack(codings_video)
    codings_audio = torch.vstack(codings_audio)
    return codings_video, codings_audio


def save_codings(vid, aud, subset, output_dp):
    vid_path = os.path.join(output_dp, f"{str(datetime.now())[:10]}_video_encoding_{subset}.pt")
    aud_path = os.path.join(output_dp, f"{str(datetime.now())[:10]}_audio_encoding_{subset}.pt")
    torch.save(vid, vid_path)
    torch.save(aud, aud_path)


def main(checkpoint_path, output_dp):
    model = load_model(checkpoint_path)
    for subset in ["validate", "test"]:
        dl, len_dataset = get_dataloader(subset)
        codings_video, codings_audio = calculate_embeddings(model, dl, len_dataset)
        save_codings(vid=codings_video, aud=codings_audio, subset=subset, output_dp=output_dp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dp", type=str, required=True)
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/cluster-polyaxon/users/molesz/checkpoints/avid-cma-kinetics400/checkpoint.pth.tar"
    )
    args = parser.parse_args()
    main(
        checkpoint_path=args.checkpoint_path,
        output_dp=args.output_dp,
    )
