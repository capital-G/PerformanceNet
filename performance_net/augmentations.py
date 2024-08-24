import abc
from typing import Optional, Tuple

import numpy as np
import torch

from performance_net.transformations import PerformanceVectorFactory


class Augmentation(abc.ABC):
    # the augmentation happens on dataloader/batch level so we have a high variety
    # during training and not just when changing files which take some time to iterate through
    # x and y need to have the shape [sequence, features]
    @abc.abstractmethod
    def __call__(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass


class Transposer(Augmentation):
    """A transposer which samples an integer from a normal
    distribution for transposition.
    This allows to create deviations and data augmentation
    while still emphasizing the original key.
    Any values that would clip will be tuned down/up one octave (12 semitones that is)
    """

    def __init__(self, vector_factory: PerformanceVectorFactory, std_dev: float = 4.0):
        """
        :param std_dev: standard deviation used for sampling
        """
        self._vector_factory = vector_factory
        self.std_dev = std_dev
        self.mean = 0.0

    def _transpose_within_boundaries(self, x: torch.Tensor, offset: int):
        # x must have shape of [sequence, features]
        # transpose is happening inplace
        for i in range(x.shape[0]):
            # we skip rows that don't have a relevant value to avoid any messing around
            if x[i, :].sum() == 0.0:
                continue
            note = x[i, :].argmax(dim=-1)
            note_offset = offset
            if note + note_offset < 0:
                note_offset += 12
            elif note + note_offset >= x.shape[-1]:
                note_offset -= 12
            x[i] = x[i].roll(int(note_offset))

    def __call__(
        self, x: torch.Tensor, y: torch.Tensor, offset: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not offset:
            offset = int(np.random.normal(self.mean, self.std_dev))

        self._transpose_within_boundaries(
            self._vector_factory.note_on_vector_view(x, with_pedal=False), offset
        )
        self._transpose_within_boundaries(
            self._vector_factory.note_off_vector_view(x, with_pedal=False), offset
        )

        self._transpose_within_boundaries(
            self._vector_factory.note_on_vector_view(y, with_pedal=False), offset
        )
        self._transpose_within_boundaries(
            self._vector_factory.note_off_vector_view(y, with_pedal=False), offset
        )

        return x, y


class TimerStretcher:
    pass


class VelocityRandomizer:
    pass
