"""TMR-style shared-latent encoders used by the Lite retriever."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _summary(x: Tensor) -> Tensor:
    """Compact temporal statistics that preserve pose and dynamics."""

    if x.ndim != 3:
        raise ValueError(f"expected (B,T,D), got {tuple(x.shape)}")
    delta = x[:, 1:] - x[:, :-1]
    return torch.cat(
        [
            x.mean(dim=1),
            x.std(dim=1, unbiased=False),
            delta.mean(dim=1),
            delta.std(dim=1, unbiased=False),
        ],
        dim=-1,
    )


class Projection(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(self.net(x), dim=-1)


class SharedRetriever(nn.Module):
    """Motion, music, and text encoders projected into one normalized space."""

    def __init__(
        self,
        motion_dim: int = 319,
        music_dim: int = 35,
        text_dim: int = 256,
        latent_dim: int = 128,
        genre_vocab: Sequence[str] = (),
    ) -> None:
        super().__init__()
        self.motion_dim = int(motion_dim)
        self.music_dim = int(music_dim)
        self.text_dim = int(text_dim)
        self.latent_dim = int(latent_dim)
        self.genre_vocab = list(genre_vocab)
        self.motion_encoder = Projection(self.motion_dim * 4, 512, latent_dim)
        self.music_encoder = Projection(self.music_dim * 4, 256, latent_dim)
        self.text_encoder = Projection(self.text_dim, 256, latent_dim)
        self.genre_head = nn.Linear(latent_dim, len(self.genre_vocab)) if self.genre_vocab else None

    def encode_motion(self, motion: Tensor) -> Tensor:
        return self.motion_encoder(_summary(motion))

    def encode_music(self, music: Tensor) -> Tensor:
        return self.music_encoder(_summary(music))

    def encode_text(self, text: Tensor) -> Tensor:
        return self.text_encoder(text)

    def forward(self, motion: Tensor, music: Tensor, text: Tensor) -> dict[str, Tensor]:
        motion_z = self.encode_motion(motion)
        music_z = self.encode_music(music)
        text_z = self.encode_text(text)
        result = {"motion": motion_z, "music": music_z, "text": text_z}
        if self.genre_head is not None:
            result["genre_logits"] = self.genre_head(motion_z)
        return result
