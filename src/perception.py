"""感知模型的共用部分：參數向量拆解、主觀機率對照表、實際門檻的組成。

05 估計模型並把結果存成 perception.pkl，06（模型化的實際行為）與 07（實際門檻 vs 最佳門檻）
都要用同一套方式把參數向量 th 拆開、算出每顆球的實際門檻，所以集中在這裡。

模型（分守備方 def／打擊方 off 各自估計）
  m   = 真實誤判幅度（英吋，> 0 表示原判錯誤）
  球員看到 m̃ = m + σ_e·ε，ε ~ N(0,1)，σ_e 依最近的好球帶邊（上緣／下緣／左右）不同
  若 m̃ > c 則挑戰，但只有比例 λ 的機會會被「考慮」（注意力）
  → P(挑戰) = λ · Φ((m − c) / σ_e)
  c = β_cell（局數區段 × 剩餘次數）＋ γ·(log G − 平均) ＋ δ·球數虛擬變數 ＋ α_player
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.config import EDGES, G_FLOOR, IG_EDGES, IG_LAB, POST_GRID_ARANGE, PRIOR_CLIP, PRIOR_EDGES_ARANGE

#: 球數虛擬變數：一好球、兩好球、三壞球
N_COUNT_DUMMIES: int = 3
N_EDGES: int = len(EDGES)


def inning_group(inning: pd.Series) -> pd.Series:
    """局數 → 局數區段字串（"1-3"、"4-6"、"7-8"、"9"、"10+"）。"""
    return pd.cut(inning, IG_EDGES, labels=IG_LAB).astype(str)


def count_dummies(df: pd.DataFrame) -> np.ndarray:
    """球數虛擬變數矩陣 [一好球, 兩好球, 三壞球]（投球前球數）。

    放進門檻是為了吸收 G 之外的球數慣性，例如打者多半只在兩好球時挑戰。
    """
    return np.column_stack([df.strikes_pre.eq(1), df.strikes_pre.eq(2), df.balls_pre.eq(3)]).astype(float)


def log_g(g: pd.Series) -> pd.Series:
    """log G，G 先截在 G_FLOOR 以上（避免 log 0）。"""
    return np.log(g.clip(lower=G_FLOOR))


def unpack_theta(th: np.ndarray, nc: int) -> tuple[np.ndarray, float, np.ndarray, float, np.ndarray, np.ndarray]:
    """把 05 的參數向量拆成 (β, γ, logσ, logitλ, δ, α)。

    排列順序：β（nc 個狀態格）、γ、logσ（3 個邊）、logitλ、δ（3 個球數）、α（球員）。
    """
    beta = th[:nc]
    gam = th[nc]
    ls = th[nc + 1 : nc + 1 + N_EDGES]
    ll = th[nc + 1 + N_EDGES]
    delta = th[nc + 2 + N_EDGES : nc + 2 + N_EDGES + N_COUNT_DUMMIES]
    alpha = th[nc + 2 + N_EDGES + N_COUNT_DUMMIES :]
    return beta, gam, ls, ll, delta, alpha


def logistic(x: float | np.ndarray) -> float | np.ndarray:
    """logit 的反函數。"""
    return 1 / (1 + np.exp(-x))


@dataclass
class SideParams:
    """perception.pkl 中一方（"off" 或 "def"）的參數，已拆好方便使用。"""

    cells: list[str]
    players: np.ndarray
    lg_mean: float
    beta: np.ndarray
    gam: float
    log_sigma: np.ndarray
    logit_lam: float
    delta: np.ndarray
    alpha: np.ndarray

    @classmethod
    def from_pickle(cls, side_dict: dict) -> SideParams:
        """由 perception.pkl 的 perc[side] 建立。"""
        cells = side_dict["cells"]
        beta, gam, ls, ll, delta, alpha = unpack_theta(side_dict["th"], len(cells))
        return cls(cells, side_dict["players"], side_dict["lg_mean"], beta, gam, ls, ll, delta, alpha)

    @property
    def lam(self) -> float:
        """注意力 λ。"""
        return 1 / (1 + np.exp(-self.logit_lam))

    @property
    def sigma(self) -> np.ndarray:
        """三個邊的感知誤差 σ（英吋）。"""
        return np.exp(self.log_sigma)

    def cell_index(self, ig: np.ndarray, k: np.ndarray | int) -> np.ndarray:
        """狀態格 "局數區段|k剩餘次數" 的索引，不存在的格回傳 −1。"""
        cmap = {c: i for i, c in enumerate(self.cells)}
        if np.ndim(k) == 0:
            return np.array([cmap.get(f"{g}|k{k}", -1) for g in ig])
        return np.array([cmap.get(f"{g}|k{x}", -1) for g, x in zip(ig, k, strict=False)])

    def player_effect(self, player: np.ndarray) -> np.ndarray:
        """球員效果 α，估計樣本中沒出現的球員給 0。"""
        pidx = np.clip(np.searchsorted(self.players, player), 0, len(self.players) - 1)
        return np.where(self.players[pidx] == player, self.alpha[pidx], 0.0)


_DEFAULT_GRID: np.ndarray = np.arange(*POST_GRID_ARANGE)


def posterior_table(
    m_prior: np.ndarray,
    sigma: float,
    grid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """主觀機率對照表 p̃(m̃) = P(m > 0 | m̃)。

    先驗為實際機會中 m 的分布（0.05 英吋直方圖，|m| 截在 30 英吋內），
    觀察值 m̃ = m + σ·ε。對每個網格點 m̃，以貝氏定理算原判錯誤（m > 0）的後驗機率。
    """
    if grid is None:
        grid = _DEFAULT_GRID  # 共用同一個陣列物件：pickle 時只存一份（與原版 perception.pkl 位元相同）
    edges = np.arange(*PRIOR_EDGES_ARANGE)
    h, _ = np.histogram(np.clip(m_prior, -PRIOR_CLIP, PRIOR_CLIP), edges)
    mids = (edges[:-1] + edges[1:]) / 2
    w = norm.pdf((grid[:, None] - mids[None, :]) / sigma) * h[None, :]
    post = (w * (mids > 0)).sum(1) / w.sum(1)
    return grid, post
