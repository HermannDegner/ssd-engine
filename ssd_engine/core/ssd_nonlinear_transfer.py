"""
非線形層間転送システム - 新コア対応版
=========================================

Phase 3の理論的整合を高めるための、非線形層間転送実装。
新Log-Alignmentコアエンジンに統合対応。
"""

from dataclasses import dataclass
from typing import Callable, Optional, Dict, List
from enum import Enum
import numpy as np


class StructuralLayer(Enum):
    """構造層（新コア対応版）"""
    PHYSICAL = 0
    BASE = 1
    CORE = 2
    UPPER = 3


@dataclass
class NonlinearTransferFunction:
    """非線形転送関数の定義"""
    source_layer: StructuralLayer
    target_layer: StructuralLayer
    base_coefficient: float
    transfer_function: Callable[[float, float, float, float], float]
    description: str
    log_align_adaptation: float = 1.0


class NonlinearInterlayerTransfer:
    """非線形層間転送システム（新コア対応版）"""
    
    def __init__(self):
        self.transfer_functions: List[NonlinearTransferFunction] = []
        self._register_default_functions()
        self.global_strength: float = 1.0
        self.log_align_compensation: bool = True
        
    def _register_default_functions(self):
        """デフォルトの非線形転送関数を登録"""
        
        # 1. UPPER → BASE: 理念による本能の抑制
        self.transfer_functions.append(
            NonlinearTransferFunction(
                source_layer=StructuralLayer.UPPER,
                target_layer=StructuralLayer.BASE,
                base_coefficient=-0.12,
                transfer_function=self._saturating_suppression,
                description="理念による本能の抑制",
                log_align_adaptation=1.2
            )
        )
        
        # 2. BASE → CORE: 本能から規範への転送
        self.transfer_functions.append(
            NonlinearTransferFunction(
                source_layer=StructuralLayer.BASE,
                target_layer=StructuralLayer.CORE,
                base_coefficient=0.08,
                transfer_function=self._saturating_transfer,
                description="本能→規範転送",
                log_align_adaptation=0.9
            )
        )
        
        # 3. CORE → BASE: 規範による本能の抑制
        self.transfer_functions.append(
            NonlinearTransferFunction(
                source_layer=StructuralLayer.CORE,
                target_layer=StructuralLayer.BASE,
                base_coefficient=-0.08,
                transfer_function=self._kappa_weighted_suppression,
                description="規範による本能の抑制",
                log_align_adaptation=1.1
            )
        )
        
        # 4. PHYSICAL → BASE: 身体から本能への転送
        self.transfer_functions.append(
            NonlinearTransferFunction(
                source_layer=StructuralLayer.PHYSICAL,
                target_layer=StructuralLayer.BASE,
                base_coefficient=0.15,
                transfer_function=self._pain_transfer,
                description="身体→本能転送（痛覚）",
                log_align_adaptation=1.4
            )
        )
    
    def compute_transfer(self, E: np.ndarray, kappa: np.ndarray, log_aligned: bool = True) -> np.ndarray:
        """非線形層間転送を計算"""
        transfer = np.zeros(4)
        
        for func_def in self.transfer_functions:
            src_idx = func_def.source_layer.value
            tgt_idx = func_def.target_layer.value
            
            E_source = E[src_idx]
            E_target = E[tgt_idx]
            kappa_source = kappa[src_idx]
            kappa_target = kappa[tgt_idx]
            
            # 非線形転送量を計算
            transfer_amount = func_def.transfer_function(
                E_source, E_target, kappa_source, kappa_target
            )
            
            # 基本係数適用
            base_transfer = func_def.base_coefficient * transfer_amount
            
            # Log-Alignment補償
            if log_aligned and self.log_align_compensation:
                base_transfer *= func_def.log_align_adaptation
            
            # 全体強度適用
            base_transfer *= self.global_strength
            
            transfer[tgt_idx] += base_transfer
        
        return transfer
    
    def _saturating_suppression(self, E_source: float, E_target: float, kappa_source: float, kappa_target: float) -> float:
        """飽和抑制関数"""
        suppression_power = E_source * kappa_source * 0.8
        resistance = 1.0 + E_target / 8.0
        return suppression_power / resistance
    
    def _saturating_transfer(self, E_source: float, E_target: float, kappa_source: float, kappa_target: float) -> float:
        """飽和転送関数"""
        transfer_power = E_source * 0.9
        saturation_factor = 1.0 / (1.0 + E_target / 40.0)
        receptivity = kappa_target / 2.2
        return transfer_power * saturation_factor * receptivity
    
    def _kappa_weighted_suppression(self, E_source: float, E_target: float, kappa_source: float, kappa_target: float) -> float:
        """κ重み付き抑制関数"""
        suppression_strength = E_source * (kappa_source / 1.8)
        resistance = 1.0 + E_target / 15.0
        return suppression_strength / resistance
    
    def _pain_transfer(self, E_source: float, E_target: float, kappa_source: float, kappa_target: float) -> float:
        """痛覚転送関数"""
        pain_intensity = E_source * 0.9
        pain_sensitivity = 1.8 / (kappa_source + 1.0)
        return pain_intensity * pain_sensitivity
    
    def set_global_strength(self, strength: float):
        """全体的な転送強度を設定"""
        self.global_strength = max(0.0, min(2.0, strength))
    
    def get_transfer_description(self) -> str:
        """転送関数の説明を取得"""
        desc = "非線形層間転送関数（新コア対応版）:\n"
        desc += f"  全体強度: {self.global_strength:.2f}\n"
        desc += f"  Log-Alignment補償: {'有効' if self.log_align_compensation else '無効'}\n\n"
        
        for i, func in enumerate(self.transfer_functions, 1):
            desc += f"  {i}. {func.source_layer.name} → {func.target_layer.name}\n"
            desc += f"     係数: {func.base_coefficient:+.3f}\n"
            desc += f"     説明: {func.description}\n\n"
        return desc


def create_default_nonlinear_transfer() -> NonlinearInterlayerTransfer:
    """デフォルト非線形転送システムを作成"""
    return NonlinearInterlayerTransfer()


def demo_nonlinear_transfer():
    """非線形転送システムのデモ"""
    print("=== 非線形層間転送システムデモ（新コア対応版） ===")
    
    transfer_system = create_default_nonlinear_transfer()
    
    print(transfer_system.get_transfer_description())
    
    # テストデータ
    E = np.array([50.0, 80.0, 40.0, 30.0])
    kappa = np.array([0.9, 0.7, 0.6, 0.5])
    
    print("テスト入力:")
    print(f"  E = {E}")
    print(f"  κ = {kappa}")
    
    # 転送計算
    transfer = transfer_system.compute_transfer(E, kappa, log_aligned=True)
    
    print(f"\n転送結果:")
    print(f"  転送量 = {transfer}")
    print(f"  転送後予想E = {E + transfer * 0.1}")


if __name__ == "__main__":
    demo_nonlinear_transfer()
