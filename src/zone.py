"""ABS 好球帶重判的幾何計算：帶號距離、最近的邊、壘況代碼、誤判幅度。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import BALL_RADIUS_FT, EDGES, INCHES_PER_FOOT, PLATE_HALF_WIDTH_FT


def bases_code(df: pd.DataFrame) -> pd.Series:
    """壘況代碼 0–7：一壘 = 1、二壘 = 2、三壘 = 4 相加（Statcast on_1b／on_2b／on_3b 是跑者 ID 或空值）。"""
    return df.on_1b.notna().astype(int) + 2 * df.on_2b.notna().astype(int) + 4 * df.on_3b.notna().astype(int)


def signed_dist(x: np.ndarray, z: np.ndarray, top: np.ndarray, bot: np.ndarray) -> np.ndarray:
    """球心到「好球帶外擴一個球半徑」矩形的帶號距離（英吋）。

    > 0 表示 ABS 壞球（在擴大矩形外，值為到矩形的歐氏距離），
    ≤ 0 表示 ABS 好球（在矩形內，值為到最近一邊距離的負值）。

    為什麼外擴一個球半徑：ABS 判定「球的任何部分碰到好球帶」就是好球，
    等價於「球心落在好球帶外擴一個球半徑的區域內」。這裡用矩形近似（四角不做圓角），
    以被挑戰球檢驗，分類正確率 99.71%。
    參數單位都是英尺（Statcast plate_x、plate_z、sz_top、sz_bot）。
    """
    xo = PLATE_HALF_WIDTH_FT + BALL_RADIUS_FT
    zt, zb = top + BALL_RADIUS_FT, bot - BALL_RADIUS_FT
    dx = np.maximum(np.abs(x) - xo, 0)
    dz = np.maximum.reduce([z - zt, zb - z, np.zeros_like(z)])
    outside = np.sqrt(dx**2 + dz**2)
    inside = -np.minimum.reduce([xo - np.abs(x), zt - z, z - zb])
    return np.where((dx > 0) | (dz > 0), outside, inside) * INCHES_PER_FOOT


def nearest_edge(plate_x: pd.Series, plate_z: pd.Series, sz_top: pd.Series, sz_bot: pd.Series) -> np.ndarray:
    """回傳每顆球最接近的好球帶邊（"side"／"top"／"bottom"）。

    以「超出擴大矩形各邊的距離」比大小：左右超出最多 → side，否則上緣超出較多 → top，其餘 → bottom。
    感知誤差 σ 依這三個邊分開估計（球員判斷上下緣與左右的難度不同）。
    """
    xo = PLATE_HALF_WIDTH_FT + BALL_RADIUS_FT
    dx = np.abs(plate_x) - xo
    dt = plate_z - (sz_top + BALL_RADIUS_FT)
    db = (sz_bot - BALL_RADIUS_FT) - plate_z
    return np.select([(dx >= dt) & (dx >= db), dt >= db], ["side", "top"], "bottom")


def edge_index(edge: pd.Series) -> np.ndarray:
    """把邊的名稱轉成 EDGES 的索引（0 = bottom、1 = side、2 = top）。"""
    return edge.map({e: i for i, e in enumerate(EDGES)}).values


def misjudgment(orig_strike: np.ndarray | pd.Series, signed_dist_in: np.ndarray | pd.Series) -> np.ndarray:
    """真實誤判幅度 m（英吋）：> 0 表示原判錯誤。

    原判好球時，球在擴大矩形外越遠（signed_dist_in 越大）誤判越明顯，
    原判壞球時方向相反，所以取負號。
    """
    return np.where(orig_strike, signed_dist_in, -signed_dist_in)
