import pytest
import torch

from performance_net.augmentations import Transposer
from performance_net.transformations import PerformanceVectorFactory


@pytest.fixture
def vector_factory():
    return PerformanceVectorFactory()


@pytest.fixture
def transposer(vector_factory):
    return Transposer(
        vector_factory=vector_factory,
    )


@pytest.fixture
def num_dims(vector_factory: PerformanceVectorFactory) -> int:
    # returns a performance vector with time sequence
    num_dims = (
        vector_factory.num_note_on
        + vector_factory.num_note_off
        + vector_factory.num_time
        + vector_factory.num_velocities
    )
    return num_dims


def mock_performance_vector(num_dims, time_steps):
    vec = torch.eye(num_dims)[torch.randint(0, num_dims, size=[time_steps])]
    return vec


def test_mock_performance_vector(num_dims):
    assert all(
        [
            x == 1
            for x in list(mock_performance_vector(num_dims, time_steps=10).sum(axis=-1))
        ]
    )


def test_transpose_in_range(transposer, num_dims):
    vec = torch.eye(num_dims)[torch.Tensor([1]).to(torch.int).repeat(5)]
    assert vec.shape == (5, num_dims)
    assert int(vec.argmax(dim=-1)[0]) == 1

    transposer._transpose_within_boundaries(vec, -2)
    # needs to wrap
    assert int(vec.argmax(dim=-1)[0]) == 11


def test_transpose_note_off(transposer, num_dims):
    vec_offset = transposer._vector_factory._note_off_vector_offset
    vec = torch.eye(num_dims)[torch.Tensor([vec_offset + 1]).to(torch.int).repeat(5)]
    assert vec.shape == (5, num_dims)
    assert int(vec.argmax(dim=-1)[0]) == vec_offset + 1

    transposer._transpose_within_boundaries(
        transposer._vector_factory.note_off_vector_view(vec), -2
    )
    # needs to wrap

    assert int(vec.argmax(dim=-1)[0]) == vec_offset + 11


def test_dont_touch_beyond(transposer, num_dims):
    vec_offset = transposer._vector_factory._note_on_vector_offset
    vec = torch.eye(num_dims)[torch.Tensor([vec_offset + 5]).to(torch.int).repeat(5)]
    assert vec.shape == (5, num_dims)
    assert int(vec.argmax(dim=-1)[0]) == vec_offset + 5

    transposer._transpose_within_boundaries(
        transposer._vector_factory.note_off_vector_view(vec), -2
    )
    assert int(vec.argmax(dim=-1)[0]) == vec_offset + 5


def test_consistent_wrapping(transposer, num_dims):
    vec_offset = transposer._vector_factory._note_on_vector_offset
    x = torch.eye(num_dims)[torch.Tensor([vec_offset + 5]).to(torch.int).repeat(5)]
    y = torch.eye(num_dims)[5].unsqueeze(0)

    assert x.shape == (5, num_dims)
    assert y.shape == (1, num_dims)

    transposer(x, y, 3)
    assert int(x.argmax(dim=-1)[0]) == vec_offset + 5 + 3
    assert int(y.argmax(dim=-1)[0]) == vec_offset + 5 + 3
