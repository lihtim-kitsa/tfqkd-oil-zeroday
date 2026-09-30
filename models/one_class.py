import torch
import torch.nn as nn


class DeepSVDD(nn.Module):
    """
    Deep Anomaly Detector (Autoencoder-based).
    Trained ONLY on nominal (Class 0) data.  At inference the anomaly score
    is the per-sample reconstruction error ||x - x̂||².

    Architecture choices
    --------------------
    * Tight bottleneck (rep_dim=2 << input_dim=5) forces information loss,
      so the network can only reconstruct patterns it learned during training.
    * Tanh activations keep encoder outputs bounded, preventing gradient
      blow-up on out-of-distribution inputs.
    """

    def __init__(self, input_dim=5, rep_dim=2):
        super(DeepSVDD, self).__init__()
        self.rep_dim = rep_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, rep_dim),
        )

        # Decoder  (mirror)
        self.decoder = nn.Sequential(
            nn.Linear(rep_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 32),
            nn.Tanh(),
            nn.Linear(32, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
