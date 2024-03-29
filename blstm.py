import torch
import torch.nn as nn

class BLSTM(nn.Module):
    def __init__(
        self, input_size, hidden_size, num_layers, num_classes, dropout, device
    ):
        super(BLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)
        self.device = device

    def forward(self, x):
        out, _ = self.lstm(x)

        # decode last time step
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)

        return self.softmax(out)