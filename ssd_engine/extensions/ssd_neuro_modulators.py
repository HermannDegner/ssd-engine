# -*- coding: utf-8 -*-
"""
Neuromodulator layer for SSD Core Engine
- Dopamine (D1/D2), Noradrenaline (α/β), Serotonin (5HT1/2)
- Receptor-specific, saturating gains
- Non-destructive parameter modulation (returns a copy)
"""

from dataclasses import dataclass, replace
from typing import Optional, Dict
import numpy as np

# -------- Utilities --------
def sat(x, xmin=0.0, xmax=1.0):
    return max(xmin, min(xmax, x))

def s_curve(x, k=1.0):
    # smooth saturating nonlinearity in [0,1] → [0,1]
    x = sat(x, 0.0, 1.0)
    return x**(1/(1e-6 + k)) / (x**(1/(1e-6 + k)) + (1-x)**(1/(1e-6 + k)) + 1e-9)

# -------- Neuro state (normalized 0..1) --------
@dataclass
class NeuroState:
    D1: float = 0.3   # Dopamine D1-like (促進)
    D2: float = 0.3   # Dopamine D2-like (抑制)
    NE: float = 0.3   # Noradrenaline (覚醒/ゲイン)
    _5HT: float = 0.3  # Serotonin（制御・抑制/安定化）
    ACh: float = 0.3  # Acetylcholine（注意・感度）
    # 拡張可：GABA, Glu など

# -------- Modulation config (strengths) --------
@dataclass
class NeuroConfig:
    # 感覚ゲイン（Log-Alignment alpha0 の感じやすさ）
    k_sense_D1: float = 0.30
    k_sense_NE: float  = 0.20
    k_sense_5HT: float = -0.10  # 5HTは過感度を抑える方向

    # LEAP閾値調整（Theta_values を縮める/広げる）
    k_theta_D1: float = -0.25  # D1↑でLEAPしやすい（閾値↓）
    k_theta_D2: float = 0.20   # D2↑でLEAP抑制（閾値↑）

    # エネルギー生成（gamma_values 活動性）
    k_gamma_D1: float = 0.15   # D1↑でエネルギー生成↑
    k_gamma_NE: float = 0.10   # NE↑で活動性↑

    # エネルギー減衰（beta_values 安定化）
    k_beta_5HT: float = 0.15   # 5HT↑で減衰↑（安定化）
    k_beta_D2: float = 0.10    # D2↑で減衰↑（抑制）

    # 探索温度/ノイズ
    k_temp_NE: float = 0.20       # NEで探索温度↑
    k_noise_5HT: float = -0.10    # 5HTでノイズ↓（安定）
    k_noise_D1: float = 0.08      # D1でノイズ微増（冒険）

    # 学習（eta_values 可塑性）
    k_eta_ACh: float = 0.20    # AChで学習率↑（注意）
    k_eta_D1: float = 0.10     # D1で学習促進

    # 導電性（G0, g オーム則パラメータ）
    k_conductance_NE: float = 0.15  # NEで導電性↑（反応性）
    k_conductance_5HT: float = -0.10 # 5HTで導電性↓（制御）

    # クリップ域
    min_gamma_mult: float = 0.50
    max_gamma_mult: float = 2.00
    min_beta_mult: float = 0.50
    max_beta_mult: float = 3.00
    min_theta_mult: float = 0.30
    max_theta_mult: float = 3.00

# -------- Modulation entry-point --------
def modulate_params(core_params, neuro: NeuroState, cfg: Optional[NeuroConfig] = None):
    """
    Returns a COPY of core_params with neuromodulator-aware tweaks.
    core_params: SSDCoreParams (from ssd_core_engine_log.py)
    neuro: NeuroState (0..1 normalized levels)
    """
    if cfg is None:
        cfg = NeuroConfig()

    p = core_params  # shorthand
    q = replace(p)   # copy (dataclass replace)

    # --- 1) 感覚ゲイン（Log-Alignment alpha0 の感じやすさ） ---
    sense_gain = 1.0 + cfg.k_sense_D1 * s_curve(neuro.D1) \
                     + cfg.k_sense_NE  * s_curve(neuro.NE) \
                     + cfg.k_sense_5HT * s_curve(neuro._5HT)
    q.alpha0 = max(1e-3, p.alpha0 * sense_gain)

    # --- 2) LEAP閾値調整（Theta_values を神経変調） ---
    theta_mult = 1.0 + cfg.k_theta_D1 * s_curve(neuro.D1) \
                     + cfg.k_theta_D2 * s_curve(neuro.D2)
    theta_mult = sat(theta_mult, cfg.min_theta_mult, cfg.max_theta_mult)
    q.Theta_values = [theta * theta_mult for theta in p.Theta_values]

    # --- 3) エネルギー生成（gamma_values 活動性調整） ---
    gamma_mult = 1.0 + cfg.k_gamma_D1 * s_curve(neuro.D1) \
                     + cfg.k_gamma_NE * s_curve(neuro.NE)
    gamma_mult = sat(gamma_mult, cfg.min_gamma_mult, cfg.max_gamma_mult)
    q.gamma_values = [gamma * gamma_mult for gamma in p.gamma_values]

    # --- 4) エネルギー減衰（beta_values 安定化調整） ---
    beta_mult = 1.0 + cfg.k_beta_5HT * s_curve(neuro._5HT) \
                    + cfg.k_beta_D2 * s_curve(neuro.D2)
    beta_mult = sat(beta_mult, cfg.min_beta_mult, cfg.max_beta_mult)
    q.beta_values = [beta * beta_mult for beta in p.beta_values]

    # --- 5) 学習可塑性（eta_values 注意・学習） ---
    eta_mult = 1.0 + cfg.k_eta_ACh * s_curve(neuro.ACh) \
                   + cfg.k_eta_D1 * s_curve(neuro.D1)
    eta_mult = max(0.1, eta_mult)
    q.eta_values = [eta * eta_mult for eta in p.eta_values]

    # --- 6) 導電性（G0, g オーム則パラメータ） ---
    conductance_mult = 1.0 + cfg.k_conductance_NE * s_curve(neuro.NE) \
                           + cfg.k_conductance_5HT * s_curve(neuro._5HT)
    conductance_mult = max(0.1, conductance_mult)
    q.G0 = max(1e-6, p.G0 * conductance_mult)
    q.g = max(1e-6, p.g * conductance_mult)

    # --- 7) 探索温度/ノイズ ---
    q.temperature_T = max(0.0, p.temperature_T * (1.0 + cfg.k_temp_NE * s_curve(neuro.NE)))
    
    # ノイズレベル調整
    noise_mult = 1.0 + cfg.k_noise_5HT * s_curve(neuro._5HT) + cfg.k_noise_D1 * s_curve(neuro.D1)
    noise_mult = max(0.1, noise_mult)
    q.epsilon_noise = max(1e-6, p.epsilon_noise * noise_mult)

    return q

# Optional: simple policy helper
def neuro_preset(name: str) -> NeuroState:
    name = name.lower()
    if name in ("focus", "集中"):
        return NeuroState(D1=0.4, D2=0.3, NE=0.5, _5HT=0.5, ACh=0.6)
    if name in ("explore", "探索"):
        return NeuroState(D1=0.7, D2=0.2, NE=0.7, _5HT=0.2, ACh=0.4)
    if name in ("calm", "鎮静"):
        return NeuroState(D1=0.2, D2=0.5, NE=0.2, _5HT=0.7, ACh=0.5)
    return NeuroState()
