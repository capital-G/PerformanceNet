import math
import os
from pathlib import Path
from typing import List

import lightning as L
import torch
import torch.utils
import torch.utils.data

from performance_net.augmentations import Augmentation, Transposer
from performance_net.transformations import PerformanceVectorFactory, PianoRoll


class MIDIDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        window_size: int = 400,
        batch_size: int = 64,
        file_ext: str = ".mid,.midi",
        num_workers: int = 0,
    ) -> None:
        """Loads MIDI files from a director and transforms them into a performance net vector.

        :param data_dir: Directory which will be scanned recursively for midi files
        :param window_size: Number of prior items to present the network during training
        :param batch_size: batch size of the dataset
        :param file_ext: comma separated lowercase string of file extensions to search for
        :param num_workers: number of threads to spawn which extract and transform data - should be number of cores
        """
        super().__init__()
        self.window_size = window_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.performance_vector_factory = PerformanceVectorFactory()

        if not os.path.isdir(data_dir):
            print(f"'{data_dir}' is not a folder!")
            return

        allowed_file_ext: List[str] = file_ext.split(",")

        self.midi_files: List[Path] = []
        for dir_path, _, files in os.walk(data_dir):
            for file in files:
                if any([file.lower().endswith(ext) for ext in allowed_file_ext]):
                    self.midi_files.append(Path(dir_path).joinpath(file))

        print(f"Found {len(self.midi_files)} MIDI-files")

    def setup(self, stage: str) -> None:
        self.dataset = PerformanceOneHotDataset(
            midi_files=self.midi_files,
            window_size=self.window_size,
            performance_factory=self.performance_vector_factory,
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=64,
            num_workers=self.num_workers,
        )


class PerformanceOneHotDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        midi_files: List[Path],
        performance_factory: PerformanceVectorFactory,
        window_size: int,
    ) -> None:
        super().__init__()
        self.midi_files = midi_files
        self.window_size = window_size
        self._performance_vector_factory = performance_factory
        # self.num_velocities = num_velocities
        # self.num_time_slots = num_time_slots
        self.window_size_out = 1
        # self.include_pedal = include_pedal

        self._performance_vector_factory = PerformanceVectorFactory()

        self.transformations: List[Augmentation] = [
            Transposer(vector_factory=self._performance_vector_factory)
        ]

    def __iter__(self):
        # determine the file offset for each worker
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            iter_start = 0
            iter_end = len(self.midi_files)
        else:
            per_worker = int(
                math.ceil((len(self.midi_files)) / float(worker_info.num_workers))
            )
            iter_start = per_worker * worker_info.id
            iter_end = min(iter_start + per_worker, len(self.midi_files))

        for midi_file_path in self.midi_files[iter_start:iter_end]:
            # print(f"Load {midi_file_path}")
            performance_vector = PianoRoll(midi_file_path).performance_vector_one_hot(
                performance_vector_factory=self._performance_vector_factory
            )
            for i in range(
                0,
                performance_vector.shape[0] - (self.window_size + self.window_size_out),
            ):
                x = performance_vector[i : i + self.window_size]
                y = performance_vector[
                    i + self.window_size : i + self.window_size + self.window_size_out
                ]
                x = torch.Tensor(x)
                y = torch.Tensor(y)

                for transformation in self.transformations:
                    x, y = transformation(x, y)

                yield x, y
