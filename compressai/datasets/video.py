# video.py
# (permanece exatamente como você enviou, sem alterações)

import random

from pathlib import Path

import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset

from compressai.registry import register_dataset


@register_dataset("VideoFolder")
class VideoFolder(Dataset):
    """Load a video folder database. Training and testing video clips
    are stored in a directory containing many sub-directories like Vimeo90K Dataset:

    .. code-block::

        - rootdir/
            train.list
            test.list
            - sequences/
                - 00010/
                    - 0001.png
                    - 0002.png
                    ...
                - 00011/
                ...

    This class returns a list of `num_frames` frames as tensors.
    """

    def __init__(
        self,
        root,
        rnd_interval=False,
        rnd_temp_order=False,
        transform=None,
        split="train",
        num_frames=5,
        ignore_sequence_folder=False
    ):
        if transform is None:
            raise RuntimeError("Transform must be applied")

        splitfile = Path(f"{root}/{split}.list")
        splitdir = Path(root) if ignore_sequence_folder else Path(f"{root}/sequences")

        if not splitfile.is_file():
            raise RuntimeError(f'Missing file "{splitfile}"')
        if not splitdir.is_dir():
            raise RuntimeError(f'Missing directory "{splitdir}"')

        with open(splitfile, "r") as f_in:
            self.sample_folders = [splitdir / f.strip() for f in f_in]

        self.max_frames = num_frames
        self.rnd_interval = rnd_interval
        self.rnd_temp_order = rnd_temp_order
        self.transform = transform

    def __getitem__(self, index):
        sample_folder = self.sample_folders[index]
        samples = sorted(p for p in sample_folder.iterdir() if p.is_file())

        max_interval = (len(samples) + 2) // self.max_frames
        interval = random.randint(1, max_interval) if self.rnd_interval else 1
        frame_paths = (samples[::interval])[: self.max_frames]

        # load all frames, concatenate along width, then transform and split
        frames = np.concatenate(
            [np.asarray(Image.open(p).convert("RGB")) for p in frame_paths], axis=-1
        )
        frames = torch.chunk(self.transform(frames), self.max_frames)

        if self.rnd_temp_order and random.random() < 0.5:
            return frames[::-1]
        return frames

    def __len__(self):
        return len(self.sample_folders)
