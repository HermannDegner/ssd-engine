"""
SSD Multidimensional Pressure - 多次元意味圧モジュール（新コア対応版）
===============================================================================

構造主観力学の「意味圧」を多次元的に計算し、四層構造別に集計するシステム。
新しいLog-Alignment対応コアエンジンに統合。

核心概念:
- 意味圧は単一ではなく、複数の次元から構成される
- 各次元は特定の構造層（PHYSICAL/BASE/CORE/UPPER）に作用する
- 層ごとに集計された圧力が、その層の「未処理圧（E）」の生成源となる

新コア統合:
- Log-Alignment対応の圧力前処理
- 神経変調システムとの連携
- SS型（感覚過敏）対応の圧力感度調整

理論的意義:
----------
1. 内的葛藤のモデル化
   - BASE圧高（危険） × UPPER圧高（理念） → 「逃げるべきか、理念を貫くか」
   - 層間の葛藤を定量化可能

2. 層別反応ロジック
   - BASE層圧力 → 本能的・衝動的跳躍
   - CORE層圧力 → 規範的・効率的整合
   - UPPER層圧力 → 理念的・自己犠牲的跳躍

3. 動かしにくさの再現
   - R値階層（PHYSICAL > BASE > CORE > UPPER）に基づく支配構造
"""

import numpy as np
from typing import Dict, Callable, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum, auto


# -------- 構造層定義 --------
class StructuralLayer(Enum):
    """構造層（四層モデル）"""
    PHYSICAL = 0  # 物理層：生理・反射
    BASE = 1      # 基層：本能・衝動・感情
    CORE = 2      # 中核層：価値観・規範・アイデンティティ
    UPPER = 3     # 上層：理念・理想・超越的価値


# -------- 圧力次元定義 --------
@dataclass
class PressureDimension:
    """
    圧力次元定義
    
    意味圧の一つの構成要素。特定の現象・状況・感情などが
    どの構造層にどの程度の圧力を生成するかを定義。
    """
    name: str                                    # 次元名
    description: str = ""                        # 説明
    target_layers: Dict[StructuralLayer, float] = field(default_factory=dict)  # 層別重み
    base_intensity: float = 1.0                 # 基本強度
    sensitivity_factor: float = 1.0             # 感度係数（SS型等で調整）
    temporal_decay: float = 0.95                # 時間減衰率
    
    def __post_init__(self):
        """デフォルト層重みの設定"""
        if not self.target_layers:
            # デフォルト：全層に均等
            self.target_layers = {
                StructuralLayer.PHYSICAL: 0.25,
                StructuralLayer.BASE: 0.25,
                StructuralLayer.CORE: 0.25,
                StructuralLayer.UPPER: 0.25
            }


# -------- 圧力計算エンジン --------
@dataclass
class PressureCalculationResult:
    """圧力計算結果"""
    layer_pressures: np.ndarray = field(default_factory=lambda: np.zeros(4))  # 層別圧力
    total_pressure: float = 0.0                   # 総圧力
    dominant_layer: StructuralLayer = StructuralLayer.BASE  # 支配的層
    dimension_contributions: Dict[str, float] = field(default_factory=dict)  # 次元別寄与
    
    # Log-Alignment前処理結果
    aligned_pressures: np.ndarray = field(default_factory=lambda: np.zeros(4))  # 整合済み圧力
    pressure_magnitude: float = 0.0              # 圧力ノルム


class MultidimensionalPressureEngine:
    """
    多次元意味圧計算エンジン
    
    複数の圧力次元から層別意味圧を計算し、
    新コアエンジンに適合した形で出力する。
    """
    
    def __init__(self):
        self.dimensions: Dict[str, PressureDimension] = {}
        self.dimension_values: Dict[str, float] = {}  # 各次元の現在値
        self.history: List[PressureCalculationResult] = []
        
        # Log-Alignment用パラメータ
        self.log_base: float = np.e
        self.eps_log: float = 1e-6
        
        # SS型連携用
        self.ss_sensitivity_modifier: float = 1.0
        
    def add_dimension(self, dimension: PressureDimension):
        """圧力次元を追加"""
        self.dimensions[dimension.name] = dimension
        self.dimension_values[dimension.name] = 0.0
    
    def set_dimension_value(self, name: str, value: float):
        """次元値を設定"""
        if name in self.dimensions:
            self.dimension_values[name] = value
        else:
            raise ValueError(f"Unknown dimension: {name}")
    
    def update_dimension_values(self, values: Dict[str, float]):
        """複数次元値を一括更新"""
        for name, value in values.items():
            self.set_dimension_value(name, value)
    
    def calculate_layer_pressures(self, 
                                use_log_alignment: bool = True,
                                alpha_t: float = 1.0) -> PressureCalculationResult:
        """
        層別圧力を計算
        
        Args:
            use_log_alignment: Log-Alignment前処理を使用するか
            alpha_t: Log-Alignment適応ゲイン
        """
        result = PressureCalculationResult()
        
        # 1) 各次元の寄与を計算
        layer_contributions = {layer: 0.0 for layer in StructuralLayer}
        
        for dim_name, dimension in self.dimensions.items():
            if dim_name not in self.dimension_values:
                continue
                
            dim_value = self.dimension_values[dim_name]
            
            # SS型感度調整
            adjusted_value = dim_value * dimension.sensitivity_factor * self.ss_sensitivity_modifier
            
            # 各層への寄与計算
            for layer, weight in dimension.target_layers.items():
                contribution = adjusted_value * weight * dimension.base_intensity
                layer_contributions[layer] += contribution
                
            # 次元別寄与記録
            result.dimension_contributions[dim_name] = adjusted_value
        
        # 2) 層別圧力配列に変換
        result.layer_pressures = np.array([
            layer_contributions[StructuralLayer.PHYSICAL],
            layer_contributions[StructuralLayer.BASE],
            layer_contributions[StructuralLayer.CORE], 
            layer_contributions[StructuralLayer.UPPER]
        ])
        
        # 3) 総圧力・支配層計算
        result.total_pressure = np.sum(result.layer_pressures)
        if result.total_pressure > self.eps_log:
            dominant_idx = np.argmax(result.layer_pressures)
            result.dominant_layer = StructuralLayer(dominant_idx)
        
        # 4) Log-Alignment前処理
        if use_log_alignment:
            result.aligned_pressures = self._apply_log_alignment(
                result.layer_pressures, alpha_t
            )
            result.pressure_magnitude = np.linalg.norm(result.aligned_pressures)
        else:
            result.aligned_pressures = result.layer_pressures.copy()
            result.pressure_magnitude = np.linalg.norm(result.layer_pressures)
        
        # 5) 履歴記録
        self.history.append(result)
        if len(self.history) > 1000:  # 履歴上限
            self.history.pop(0)
            
        return result
    
    def _apply_log_alignment(self, pressures: np.ndarray, alpha_t: float) -> np.ndarray:
        """Log-Alignment前処理を適用"""
        aligned = np.zeros_like(pressures)
        
        for i, p in enumerate(pressures):
            if abs(p) < self.eps_log:
                aligned[i] = 0.0
            else:
                # p̂ = sign(p) * log(1 + α_t * |p|) / log(b)
                sign_p = np.sign(p)
                abs_p = abs(p)
                log_term = np.log(1.0 + alpha_t * abs_p) / np.log(self.log_base)
                aligned[i] = sign_p * log_term
                
        return aligned
    
    def get_pressure_for_core_engine(self, 
                                   use_log_alignment: bool = True,
                                   alpha_t: float = 1.0) -> np.ndarray:
        """
        新コアエンジン用の圧力配列を取得
        
        Returns:
            4層分の圧力配列（Log-Alignment済み）
        """
        result = self.calculate_layer_pressures(use_log_alignment, alpha_t)
        return result.aligned_pressures
    
    def apply_ss_sensitivity(self, ss_level: float, ss_profile_context_dependency: float = 0.7):
        """SS型（感覚過敏）感度調整を適用"""
        # 基本感度倍率
        base_multiplier = 1.0 + ss_level * 0.5
        
        # 文脈依存度による微調整
        context_modifier = 1.0 + ss_profile_context_dependency * 0.3
        
        self.ss_sensitivity_modifier = base_multiplier * context_modifier
        
        # 各次元の感度も個別調整
        for dimension in self.dimensions.values():
            if ss_level > 0.5:
                dimension.sensitivity_factor *= (1.0 + ss_level * 0.2)


# -------- プリセット圧力次元 --------
def create_kaiji_pressure_dimensions() -> Dict[str, PressureDimension]:
    """カイジ借金地獄用の圧力次元セット"""
    return {
        "debt_pressure": PressureDimension(
            name="debt_pressure",
            description="借金・金銭的困窮による圧力",
            target_layers={
                StructuralLayer.PHYSICAL: 0.4,  # 生存脅威
                StructuralLayer.BASE: 0.4,      # 不安・恐怖
                StructuralLayer.CORE: 0.1,      # 価値観への脅威
                StructuralLayer.UPPER: 0.1      # 理想への影響
            },
            base_intensity=1.2
        ),
        
        "gambling_temptation": PressureDimension(
            name="gambling_temptation",
            description="一発逆転への誘惑",
            target_layers={
                StructuralLayer.PHYSICAL: 0.1,
                StructuralLayer.BASE: 0.6,      # 欲望・衝動
                StructuralLayer.CORE: 0.2,      # 希望・期待
                StructuralLayer.UPPER: 0.1
            },
            base_intensity=1.0
        ),
        
        "social_shame": PressureDimension(
            name="social_shame",
            description="社会的恥辱・体面の失墜",
            target_layers={
                StructuralLayer.PHYSICAL: 0.1,
                StructuralLayer.BASE: 0.2,
                StructuralLayer.CORE: 0.5,      # アイデンティティ脅威
                StructuralLayer.UPPER: 0.2      # 理想からの乖離
            },
            base_intensity=0.8
        ),
        
        "time_pressure": PressureDimension(
            name="time_pressure", 
            description="時間切迫・決断強要",
            target_layers={
                StructuralLayer.PHYSICAL: 0.3,  # ストレス反応
                StructuralLayer.BASE: 0.4,      # 焦燥感
                StructuralLayer.CORE: 0.2,      # 判断力への圧迫
                StructuralLayer.UPPER: 0.1
            },
            base_intensity=1.1,
            temporal_decay=0.85  # 急速減衰
        )
    }


def create_ss_type_pressure_dimensions() -> Dict[str, PressureDimension]:
    """SS型（感覚過敏）用の圧力次元セット"""
    return {
        "sensory_overload": PressureDimension(
            name="sensory_overload",
            description="感覚過負荷・刺激過多",
            target_layers={
                StructuralLayer.PHYSICAL: 0.6,  # 生理的負荷
                StructuralLayer.BASE: 0.3,      # 不快感・疲労
                StructuralLayer.CORE: 0.1,
                StructuralLayer.UPPER: 0.0
            },
            base_intensity=1.5,
            sensitivity_factor=2.0  # SS型で高感度
        ),
        
        "social_context_pressure": PressureDimension(
            name="social_context_pressure",
            description="場の空気・文脈読み取り負荷",
            target_layers={
                StructuralLayer.PHYSICAL: 0.1,
                StructuralLayer.BASE: 0.2,
                StructuralLayer.CORE: 0.5,      # 認知負荷
                StructuralLayer.UPPER: 0.2      # 適応努力
            },
            base_intensity=1.0,
            sensitivity_factor=1.8  # SS型で高負荷
        ),
        
        "perfectionism_pressure": PressureDimension(
            name="perfectionism_pressure",
            description="完璧主義・細部へのこだわり",
            target_layers={
                StructuralLayer.PHYSICAL: 0.1,
                StructuralLayer.BASE: 0.2,
                StructuralLayer.CORE: 0.4,      # 価値基準
                StructuralLayer.UPPER: 0.3      # 理想追求
            },
            base_intensity=0.9,
            sensitivity_factor=1.6
        ),
        
        "threat_hypervigilance": PressureDimension(
            name="threat_hypervigilance",
            description="脅威過敏・警戒状態維持",
            target_layers={
                StructuralLayer.PHYSICAL: 0.4,  # 警戒反応
                StructuralLayer.BASE: 0.5,      # 不安・恐怖
                StructuralLayer.CORE: 0.1,
                StructuralLayer.UPPER: 0.0
            },
            base_intensity=1.3,
            sensitivity_factor=2.2  # SS型で極めて高感度
        )
    }


# -------- 統合ヘルパー関数 --------
def create_pressure_engine_for_scenario(scenario: str = "kaiji") -> MultidimensionalPressureEngine:
    """シナリオ別圧力エンジンを作成"""
    engine = MultidimensionalPressureEngine()
    
    if scenario == "kaiji":
        dimensions = create_kaiji_pressure_dimensions()
    elif scenario == "ss_type":
        dimensions = create_ss_type_pressure_dimensions()
    elif scenario == "combined":
        # カイジ + SS型統合
        dimensions = {}
        dimensions.update(create_kaiji_pressure_dimensions())
        dimensions.update(create_ss_type_pressure_dimensions())
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    
    for dimension in dimensions.values():
        engine.add_dimension(dimension)
    
    return engine


def demo_pressure_system():
    """圧力システムのデモンストレーション"""
    print("=" * 80)
    print("多次元意味圧システムデモ（新コア対応版）")
    print("=" * 80)
    
    # カイジシナリオエンジン
    kaiji_engine = create_pressure_engine_for_scenario("kaiji")
    
    print("\nカイジ借金地獄シナリオ:")
    print("-" * 50)
    
    # 段階的圧力変化
    stages = [
        ("平常時", {"debt_pressure": 20, "gambling_temptation": 10, "social_shame": 5, "time_pressure": 5}),
        ("借金発覚", {"debt_pressure": 60, "gambling_temptation": 30, "social_shame": 40, "time_pressure": 20}),
        ("ギャンブル中", {"debt_pressure": 80, "gambling_temptation": 90, "social_shame": 60, "time_pressure": 70}),
        ("破産寸前", {"debt_pressure": 100, "gambling_temptation": 95, "social_shame": 90, "time_pressure": 95})
    ]
    
    for stage_name, pressures in stages:
        print(f"\n[{stage_name}]:")
        kaiji_engine.update_dimension_values(pressures)
        result = kaiji_engine.calculate_layer_pressures()
        
        print(f"  層別圧力: P={result.layer_pressures[0]:.1f}, B={result.layer_pressures[1]:.1f}, "
              f"C={result.layer_pressures[2]:.1f}, U={result.layer_pressures[3]:.1f}")
        print(f"  支配層: {result.dominant_layer.name}, 総圧力: {result.total_pressure:.1f}")
        print(f"  Log整合後: {result.aligned_pressures}")
    
    # SS型統合テスト
    print("\nSS型（感覚過敏）統合テスト:")
    print("-" * 50)
    
    ss_engine = create_pressure_engine_for_scenario("ss_type")
    ss_engine.apply_ss_sensitivity(ss_level=0.8, ss_profile_context_dependency=0.9)
    
    ss_pressures = {
        "sensory_overload": 40,
        "social_context_pressure": 60,
        "perfectionism_pressure": 50,
        "threat_hypervigilance": 70
    }
    
    ss_engine.update_dimension_values(ss_pressures)
    ss_result = ss_engine.calculate_layer_pressures()
    
    print(f"  SS型圧力分布: P={ss_result.layer_pressures[0]:.1f}, B={ss_result.layer_pressures[1]:.1f}, "
          f"C={ss_result.layer_pressures[2]:.1f}, U={ss_result.layer_pressures[3]:.1f}")
    print(f"  支配層: {ss_result.dominant_layer.name}")
    print(f"  新コア用配列: {ss_engine.get_pressure_for_core_engine()}")


if __name__ == "__main__":
    demo_pressure_system()
