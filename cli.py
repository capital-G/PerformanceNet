from lightning.pytorch.cli import LightningCLI

from performance_net import MIDIDataModule, PerformanceNet, PerformanceNetTrainer


def cli_main():
    cli = LightningCLI(
        model_class=PerformanceNet,
        datamodule_class=MIDIDataModule,
        trainer_class=PerformanceNetTrainer,
    )


if __name__ == "__main__":
    cli_main()
