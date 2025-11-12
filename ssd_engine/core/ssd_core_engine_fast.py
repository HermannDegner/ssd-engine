"""
SSD Core Engine - Numba加速版
=============================

ssd_core_engineの計算集約部分をNumbaでJITコンパイル。
Numbaがない環境では自動的に通常版にフォールバック。
"""

import numpy as np

# Numbaの利用可否を判定
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # ダミーデコレータ
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


@jit(nopython=True, cache=True)
def compute_log_alignment(p: float, alpha_t: float, log_base: float) -> float:
    """対数整合の計算（高速版）"""
    if p == 0.0:
        return 0.0
    sign_p = 1.0 if p > 0 else -1.0
    abs_p = abs(p)
    return sign_p * np.log(1.0 + alpha_t * abs_p) / np.log(log_base)


@jit(nopython=True, cache=True)
def compute_ohm_flow(p_hat: float, G0: float, g: float, kappa: float) -> float:
    """Ohmの法則による流れ計算（高速版）"""
    conductance = G0 + g * kappa
    return conductance * p_hat


@jit(nopython=True, cache=True)
def compute_energy_update(E: float, residual: float, gamma: float, beta: float) -> float:
    """エネルギー更新（高速版）"""
    dE = gamma * residual - beta * E
    return max(0.0, E + dE)


@jit(nopython=True, cache=True)
def compute_inertia_update(kappa: float, usage: float, eta: float, lambda_: float, kappa_min: float) -> float:
    """慣性更新（高速版）"""
    d_kappa = eta * usage - lambda_ * kappa
    new_kappa = kappa + d_kappa
    return max(kappa_min, new_kappa)


@jit(nopython=True, cache=True)
def compute_step_fast(
    p_array: np.ndarray,
    E_array: np.ndarray,
    kappa_array: np.ndarray,
    alpha_t: float,
    log_base: float,
    G0: float,
    g: float,
    gamma_array: np.ndarray,
    beta_array: np.ndarray,
    eta_array: np.ndarray,
    lambda_array: np.ndarray,
    kappa_min_array: np.ndarray,
    Theta_array: np.ndarray
) -> tuple:
    """
    SSDコアステップの高速計算
    
    Returns:
        (j_array, E_new_array, kappa_new_array, leap_flags)
    """
    num_layers = len(p_array)
    j_array = np.zeros(num_layers)
    E_new_array = np.zeros(num_layers)
    kappa_new_array = np.zeros(num_layers)
    leap_flags = np.zeros(num_layers, dtype=np.int32)
    
    for i in range(num_layers):
        # Log-Alignment
        p_hat = compute_log_alignment(p_array[i], alpha_t, log_base)
        
        # Ohm's law
        j = compute_ohm_flow(p_hat, G0, g, kappa_array[i])
        j_array[i] = j
        
        # Residual
        residual = p_hat - j
        
        # Energy update
        E_new = compute_energy_update(E_array[i], residual, gamma_array[i], beta_array[i])
        E_new_array[i] = E_new
        
        # Leap detection
        if E_new >= Theta_array[i]:
            leap_flags[i] = 1
            E_new_array[i] = 0.0  # リセット
        
        # Inertia update
        usage = abs(j)
        kappa_new = compute_inertia_update(
            kappa_array[i], usage, eta_array[i], lambda_array[i], kappa_min_array[i]
        )
        kappa_new_array[i] = kappa_new
    
    return j_array, E_new_array, kappa_new_array, leap_flags


# モジュールレベルで利用可否を公開
__all__ = [
    'NUMBA_AVAILABLE',
    'compute_log_alignment',
    'compute_ohm_flow',
    'compute_energy_update',
    'compute_inertia_update',
    'compute_step_fast'
]
