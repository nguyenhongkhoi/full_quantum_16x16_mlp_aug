"""A quantum graph layer on the pixel grid: every pixel is a qubit, linked to its 8 neighbours.

Each pixel is RY-encoded, then exp(-i gamma Z_i Z_j) acts on every edge of the 8-neighbour grid: one gamma for the
4 horizontal and vertical edges, one for the 4 diagonal edges, the same for every pixel. These gates all commute, so
their order does not matter and the layer treats every direction the same way (it turns with the image by 90
degrees). Then one shared one-qubit gate acts on every pixel.

The layer returns the exact one-qubit state of every pixel after these gates. For pixel i,

    rho_i[0, 1] <- rho_i[0, 1] * prod_{j in N(i)} (cos 2 gamma_ij - i sin 2 gamma_ij <Z_j>),

and the diagonal of rho_i does not change. The tree that follows takes the product of these one-qubit states, so
correlations between pixels made by the layer are dropped: on hardware this needs the Bloch vector of every pixel
to be measured and prepared again. With gamma = 0 and the one-qubit gate at the identity, the layer changes
nothing, so a model with the layer starts exactly as the model without it.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .unitary import LearnedUnitary1Q

ORTHOGONAL = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIAGONAL = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def shifted(grid: torch.Tensor, dy: int, dx: int) -> tuple[torch.Tensor, torch.Tensor]:
    """grid[..., y + dy, x + dx] at every (y, x), and a mask that is False where that neighbour is off the grid."""
    height, width = grid.shape[-2:]
    padded = nn.functional.pad(grid, (1, 1, 1, 1))
    neighbour = padded[..., 1 + dy:1 + dy + height, 1 + dx:1 + dx + width]
    inside = torch.zeros(height + 2, width + 2, dtype=torch.bool, device=grid.device)
    inside[1:-1, 1:-1] = True
    return neighbour, inside[1 + dy:1 + dy + height, 1 + dx:1 + dx + width]


def graph_reduced_states(theta: torch.Tensor, gammas: torch.Tensor) -> torch.Tensor:
    """(batch, H, W) RY angles, gammas = (orthogonal, diagonal) -> (batch, H*W, 2, 2) one-qubit states, row-major."""
    z = torch.cos(theta)
    coherence = torch.ones_like(theta, dtype=torch.complex64)
    for gamma, offsets in zip(gammas, (ORTHOGONAL, DIAGONAL)):
        cosine, sine = torch.cos(2 * gamma), torch.sin(2 * gamma)
        for dy, dx in offsets:
            neighbour, inside = shifted(z, dy, dx)
            factor = torch.complex(cosine.expand_as(neighbour), -sine * neighbour)
            coherence = coherence * torch.where(inside, factor, torch.ones_like(factor))
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    off = (c * s).to(torch.complex64) * coherence
    rho = torch.stack((torch.stack(((c * c).to(torch.complex64), off), -1),
                       torch.stack((off.conj(), (s * s).to(torch.complex64)), -1)), -2)
    return rho.reshape(theta.shape[0], -1, 2, 2)


class GraphZZLayer(nn.Module):
    """10 parameters: two couplings and one shared one-qubit gate. Starts as the identity, and its creation draws no
    random numbers, so the rest of the model starts exactly as without the layer."""

    def __init__(self, height: int = 16, width: int = 16):
        super().__init__()
        self.height, self.width = height, width
        self.param_scale = 1.0  # gammas are used as param_scale * stored (fusion_model.scale_learning_rate)
        self.gammas = nn.Parameter(torch.zeros(2))
        with torch.random.fork_rng(devices=[]):
            self.local = LearnedUnitary1Q(init_scale=0.0)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """(batch, H*W) pixel values in [0, 1], row-major -> (batch, H*W, 2, 2) one-qubit states, row-major."""
        theta = values.clamp(0.0, 1.0).reshape(-1, self.height, self.width) * math.pi
        rho = graph_reduced_states(theta, self.gammas * self.param_scale)
        gate = self.local()
        return gate @ rho @ gate.conj().T
