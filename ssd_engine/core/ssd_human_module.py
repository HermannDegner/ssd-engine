"""
SSD Human Module - 人間心理特化モジュール（新コア対応版）
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, List, Tuple, Union

from .ssd_core_engine import SSDCoreEngine, SSDCoreParams, SSDCoreState, LeapType, create_default_state
from .ssd_nonlinear_transfer import NonlinearInterlayerTransfer


class HumanLayer(Enum):
    """人間モジュールの四層構造"""
    PHYSICAL = 0
    BASE = 1
    CORE = 2
    UPPER = 3


@dataclass
class HumanParams:
    """人間モジュール特化パラメータ"""
    R_physical: float = 1000.0
    R_base: float = 100.0
    R_core: float = 10.0
    R_upper: float = 1.0
    
    gamma_physical: float = 0.12
    gamma_base: float = 0.08
    gamma_core: float = 0.06
    gamma_upper: float = 0.04
    
    beta_physical: float = 0.001
    beta_base: float = 0.01
    beta_core: float = 0.05
    beta_upper: float = 0.1
    
    eta_physical: float = 0.9
    eta_base: float = 0.5
    eta_core: float = 0.3
    eta_upper: float = 0.2
    
    lambda_physical: float = 0.001
    lambda_base: float = 0.01
    lambda_core: float = 0.02
    lambda_upper: float = 0.05
    
    kappa_min_physical: float = 0.9
    kappa_min_base: float = 0.8
    kappa_min_core: float = 0.5
    kappa_min_upper: float = 0.3
    
    Theta_physical: float = 200.0
    Theta_base: float = 100.0
    Theta_core: float = 50.0
    Theta_upper: float = 30.0
    
    enable_dynamic_theta: bool = True
    theta_sensitivity: float = 0.3
    log_base: float = np.e
    alpha_t: float = 1.0
    
    def to_core_params(self) -> SSDCoreParams:
        """新コアエンジンパラメータに変換"""
        return SSDCoreParams(
            num_layers=4,
            R_values=[self.R_physical, self.R_base, self.R_core, self.R_upper],
            gamma_values=[self.gamma_physical, self.gamma_base, self.gamma_core, self.gamma_upper],
            beta_values=[self.beta_physical, self.beta_base, self.beta_core, self.beta_upper],
            eta_values=[self.eta_physical, self.eta_base, self.eta_core, self.eta_upper],
            lambda_values=[self.lambda_physical, self.lambda_base, self.lambda_core, self.lambda_upper],
            kappa_min_values=[self.kappa_min_physical, self.kappa_min_base, self.kappa_min_core, self.kappa_min_upper],
            Theta_values=[self.Theta_physical, self.Theta_base, self.Theta_core, self.Theta_upper],
            enable_dynamic_theta=self.enable_dynamic_theta,
            theta_sensitivity=self.theta_sensitivity,
            log_base=self.log_base,
            alpha0=self.alpha_t
        )


@dataclass
class HumanPressure:
    """人間特化の意味圧入力"""
    physical: float = 0.0
    base: float = 0.0
    core: float = 0.0
    upper: float = 0.0
    
    def to_vector(self) -> np.ndarray:
        return np.array([self.physical, self.base, self.core, self.upper])


class HumanAgent:
    """人間エージェント（新コア統合版）"""
    
    def __init__(self, params: Optional[HumanParams] = None, agent_id: str = "Human", enable_nonlinear_transfer: bool = True):
        self.params = params or HumanParams()
        self.agent_id = agent_id
        
        core_params = self.params.to_core_params()
        self.engine = SSDCoreEngine(core_params)
        self.state = create_default_state(num_layers=4)
        
        # 非線形転送システム統合
        self.enable_nonlinear_transfer = enable_nonlinear_transfer
        if enable_nonlinear_transfer:
            self.nonlinear_transfer = NonlinearInterlayerTransfer()
        
    def step(self, pressure: Union[HumanPressure, np.ndarray], dt: float = 0.1):
        """1ステップ実行（非線形転送対応）"""
        if isinstance(pressure, np.ndarray):
            p_vector = pressure
        else:
            p_vector = pressure.to_vector()
        
        # 非線形転送計算
        interlayer_transfer = None
        if self.enable_nonlinear_transfer:
            interlayer_transfer = self.nonlinear_transfer.compute_transfer(
                self.state.E, self.state.kappa, log_aligned=True
            )
        
        # 新コアエンジンで状態更新（非線形転送統合）
        self.state = self.engine.step(self.state, p_vector, dt=dt, interlayer_transfer=interlayer_transfer)
        return self.state
    
    def get_dominant_layer(self) -> HumanLayer:
        """最も影響力の高い層を返す"""
        p_dummy = np.ones(4)
        dominant_idx = self.engine.get_dominant_layer(self.state, p_dummy)
        return HumanLayer(dominant_idx)
    
    def set_nonlinear_strength(self, strength: float):
        """非線形転送強度を設定"""
        if self.enable_nonlinear_transfer:
            self.nonlinear_transfer.set_global_strength(strength)
    
    def __repr__(self):
        dominant = self.get_dominant_layer()
        nonlinear_status = "NonLinear" if self.enable_nonlinear_transfer else "Linear"
        return (f"HumanAgent({self.agent_id}, {nonlinear_status}) - Dominant: {dominant.name}\n"
                f"  E=[{self.state.E[0]:.1f}, {self.state.E[1]:.1f}, {self.state.E[2]:.1f}, {self.state.E[3]:.1f}]")


def demo_human_with_nonlinear():
    """非線形転送統合人間モジュールのデモ"""
    print("=== 人間心理特化モジュール + 非線形転送デモ ===")
    
    # 線形版と非線形版を比較
    human_linear = HumanAgent(agent_id="Linear", enable_nonlinear_transfer=False)
    human_nonlinear = HumanAgent(agent_id="NonLinear", enable_nonlinear_transfer=True)
    
    # 同じ圧力を投入
    pressure = HumanPressure(physical=50, base=70, core=30, upper=20)
    
    # 複数ステップ実行
    for i in range(3):
        state_linear = human_linear.step(pressure, dt=0.1)
        state_nonlinear = human_nonlinear.step(pressure, dt=0.1)
        
        print(f"\nステップ {i+1}:")
        print(f"  線形版:   {human_linear}")
        print(f"  非線形版: {human_nonlinear}")
        
        # エネルギー差を計算
        energy_diff = np.sum(np.abs(state_nonlinear.E - state_linear.E))
        print(f"  エネルギー差: {energy_diff:.2f}")
    
    print("\n非線形転送効果により、より複雑で現実的な心理動態を実現！")


if __name__ == "__main__":
    demo_human_with_nonlinear()
