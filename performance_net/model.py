import lightning as L
import torch
from torch import nn

DEFAULT_VECTOR_SIZE = 89 + 89 + 125 + 32


class PerformanceNet(L.LightningModule):
    def __init__(
        self,
        input_size: int = DEFAULT_VECTOR_SIZE,
        hidden_size: int = 256,
        output_size: int = DEFAULT_VECTOR_SIZE,
        num_layers: int = 3,
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
    ):
        super().__init__()
        self.learning_rate = learning_rate

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        z, _ = self.lstm(x)
        # only use output from last step of our rnn sequence
        z = self.fc(z[:, -1, :])
        return z

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, torch.argmax(y, dim=2).squeeze(1))
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
