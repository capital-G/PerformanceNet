import os
from datetime import datetime, timedelta

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger


class PerformanceNetTrainer(L.Trainer):
    def __init__(
        self,
        run_dir: str = "runs",
        gradient_clip_val: float = 1.0,
        save_every_n_minutes: int = 30,
        max_epochs: int = 150,
        *args,
        **kwargs
    ):
        run_dir = os.path.join(run_dir, datetime.now().strftime("%Y%m%d_%H%M"))

        callbacks = [
            ModelCheckpoint(
                dirpath=run_dir,
                # save_top_k=20,
                train_time_interval=timedelta(days=save_every_n_minutes),
            )
        ]

        logger = [
            TensorBoardLogger(
                save_dir=os.path.join(run_dir, "tensorboard"),
            )
        ]

        kwargs["callbacks"] = callbacks
        kwargs["logger"] = logger
        kwargs["gradient_clip_val"] = gradient_clip_val
        kwargs["max_epochs"] = max_epochs

        return super().__init__(*args, **kwargs)  # type: ignore
