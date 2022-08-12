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


def get_dataloader(subset, batch_size, dataset_name):
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
            missing_as_zero=True
        ),
        preprocessing.LogSpectrogram(
            24000,
            n_fft=512,
            hop_size=1. / 100,
            normalize=True
        )
    ]
    if dataset_name == "kinetics":
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
    elif dataset_name == "ucf":
        dataset = datasets.UCF(
            subset=subset,
            return_video=True,
            video_clip_duration=0.5,
            video_fps=16,
            video_transform=video_transform,
            max_offsync_augm=0,
            return_labels=True,
            mode='clip',
            clips_per_video=10,
        )
    else:
        raise Exception(f"Dataset {dataset_name} not supported")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
        sampler=None
    )
    return dataloader, len(dataset)


def calculate_embeddings(model, dataloader, len_dataset, batch_size):
    labels = []
    codings_video = []
    codings_audio = []
    for sample in dataloader:
        with torch.no_grad():
            if "audio" in sample.keys():
                logits = model(sample["frames"], sample["audio"])
                labels += [l.split("?")[-1] for l in sample["label"]]
            else:
                logits = model(sample["frames"], torch.zeros(size=[8, 1, 200, 257]))
                labels += sample["label"]
            codings_video.append(logits[0])
            codings_audio.append(logits[1])
            print(f"{np.round(100 * ((len(codings_video) * batch_size) / len_dataset), 2)}%")
    codings_video = torch.vstack(codings_video)
    codings_audio = torch.vstack(codings_audio)
    return codings_video, codings_audio, labels


def save_codings(vid, aud, lab, subset, output_dp):
    vid_path = os.path.join(output_dp, f"{str(datetime.now())[:10]}_video_encoding_{subset}.pt")
    aud_path = os.path.join(output_dp, f"{str(datetime.now())[:10]}_audio_encoding_{subset}.pt")
    lab_path = os.path.join(output_dp, f"{str(datetime.now())[:10]}_labels_{subset}.txt")
    torch.save(vid, vid_path)
    torch.save(aud, aud_path)
    with open(lab_path, "w") as f:
        for l in lab:
            f.write("%s\n" % l)


def main(checkpoint_path, output_dp, splits, batch_size, dataset_name):
    model = load_model(checkpoint_path)
    for subset in splits:
        dl, len_dataset = get_dataloader(subset, batch_size, dataset_name)
        with torch.no_grad():
            codings_video, codings_audio, labels = calculate_embeddings(model, dl, len_dataset, batch_size)
        save_codings(vid=codings_video, aud=codings_audio, lab=labels, subset=subset, output_dp=output_dp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dp", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        # default="/cluster-polyaxon/users/molesz/checkpoints/avid-cma-kinetics400/checkpoint.pth.tar",
        default="checkpoints/AVID-CMA/kinetics/Cross-N1024/checkpoint.pth.tar"
    )
    parser.add_argument("--splits", nargs="+", default=["validate", "test"])
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    main(
        checkpoint_path=args.checkpoint_path,
        output_dp=args.output_dp,
        splits=args.splits,
        batch_size=args.batch_size,
        dataset_name=args.dataset_name,
    )
