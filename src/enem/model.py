"""MLP para regressão da nota média do ENEM.

Entrada = [one-hot das categóricas | numéricas | embedding do município].
Uma camada de embedding é matematicamente uma camada linear aplicada ao one-hot
do município (sem viés); usamos o embedding só porque o one-hot teria ~5 mil
colunas quase sempre zeradas.

Com `hidden=[]` o modelo vira uma regressão linear sobre as mesmas entradas,
usada como baseline.
"""
import torch
from torch import nn

ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU, "tanh": nn.Tanh}


class EnemMLP(nn.Module):
    def __init__(self, n_onehot: int, n_num: int, n_muni: int, hidden: list[int],
                 emb_dim: int = 16, activation: str = "relu", dropout: float = 0.0,
                 norm: str = "none"):
        super().__init__()
        self.n_onehot = n_onehot
        self.emb = nn.Embedding(n_muni, emb_dim) if emb_dim > 0 else None
        in_dim = n_onehot + n_num + emb_dim

        layers = []
        for width in hidden:
            layers.append(nn.Linear(in_dim, width))
            if norm == "batch":
                layers.append(nn.BatchNorm1d(width))
            elif norm == "layer":
                layers.append(nn.LayerNorm(width))
            layers.append(ACTIVATIONS[activation]())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = width
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def one_hot(self, cat: torch.Tensor) -> torch.Tensor:
        # cat: (B, n_cols) com a posição global do "1"; -1 = categoria desconhecida,
        # enviada para uma coluna extra que é descartada (vetor de zeros no bloco).
        idx = cat.long()
        idx = torch.where(idx < 0, torch.full_like(idx, self.n_onehot), idx)
        out = torch.zeros(idx.shape[0], self.n_onehot + 1, device=idx.device)
        out.scatter_(1, idx, 1.0)
        return out[:, : self.n_onehot]

    def forward(self, cat, num, muni):
        parts = [self.one_hot(cat), num]
        if self.emb is not None:
            parts.append(self.emb(muni.long()))
        return self.net(torch.cat(parts, dim=1)).squeeze(-1)

    def penalty(self, l1: float, l2: float) -> torch.Tensor:
        """Penalidades L1/L2 explícitas sobre os pesos das camadas lineares
        (não penaliza vieses, parâmetros de normalização nem o embedding)."""
        total = torch.zeros((), device=next(self.parameters()).device)
        for m in self.net:
            if isinstance(m, nn.Linear):
                if l1 > 0:
                    total = total + l1 * m.weight.abs().sum()
                if l2 > 0:
                    total = total + l2 * m.weight.pow(2).sum()
        return total


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
