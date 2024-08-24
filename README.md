# Performance Net

A PyTorch Lightning implementation of *Performance RNN*.

The reference implementation can be found in the [magenta repository](https://github.com/magenta/magenta/tree/main/magenta/models/performance_rnn) but the setup is getting more and more complicated as the dependencies are not kept up-to-date for the Magenta project.

## Setup

> Tested with python 3.10

Create a virtual environment via

```shell
virtualenv venv
# activate the venv
source venv/bin/activate
```

Use `python cli.py --help` to display the help.

### Train a model

```shell
python cli.py fit \
    --data.data_dir data/
```

Observe the training process via tensorboard on <http://127.0.0.1:6006> via

```shell
tensorboard --logdir runs/some_date_and_time_stamp/tensorboard
```

To continue training from a given checkpoint, use

```shell
python cli.py \
    --data.data_dir "/path/to/my/data" \
    --ckpt_path "/path/to/my/checkpoint.ckpt"
```

## Trained models

## Changes

* The original paper uses all available 128 MIDI pitch values as input data. As we focus on piano, we reduced this to 88 in order to make our model less complex.
* The original paper suggest transposing the material in -5, +6 semitones for data augmentation. As we operate sequentially through MIDI files and therefore can't jump efficiently through (transposed) files, we transpose on the fly. But instead of using an uniform distribution for transposing, we are using a normal-distributed transposing around 0 to pronounce the original tuning.

## Resources

* [Performance RNN Website](https://magenta.tensorflow.org/performance-rnn)
* [Performance RNN reference implementation](https://github.com/magenta/magenta/tree/main/magenta/models/performance_rnn)
* [*This Time with Feeling: Learning Expressive Musical Performance*: Performance RNN Paper on arxiv.org](https://arxiv.org/abs/1808.03715)

## License

AGPL-3.0
