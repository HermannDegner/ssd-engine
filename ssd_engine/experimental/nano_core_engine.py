"""
SSD Nano Core Engine v8.0 (Based on v8 SubjectiveSociety Logic)
================================================================

構造主観力学（SSD）の v8 主観的社会システムに基づく、
リアルタイム・パフォーマンス重視の軽量実装。

設計思想 (nano_ssd.md):
- クラス呼び出しや抽象化レイヤーを最小限に抑える
- すべての計算をインライン化し、NumPy配列の直接操作に最適化
- 理論の核心（完全四層構造、層間変換、主観的社会圧力）は維持
- 群衆AIや能動NPCなど、多数のエージェントの同時実行を目的とする

v8 (リファレンス実装) との違い:
- 抽象化の排除: HumanAgent, SubjectiveSociety, SignalGenerator クラスを廃止
- インライン化: 全てのロジックを NanoCoreEngine クラス内の関数に集約
- パフォーマンス: 辞書参照やクラスインスタンス生成を排除
- 理論的整合性: v8の「主観的観測→解釈→自己変化」ロジックを維持

理論的位置づけ:
- ssd_core_engine.py + ssd_human_module.py: 理論の「完全性」と「抽象性」
- nano_core_engine.py: 理論の「核心」と「物理層の制約」との整合
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from enum import IntEnum


# ========================================
# 層定義（v8互換）
# ========================================

class Layer(IntEnum):
    """層インデックス定義（理論的整合性のため明記）"""
    PHYSICAL = 0
    BASE = 1
    CORE = 2
    UPPER = 3


# ========================================
# 状態とパラメータ
# ========================================

@dataclass
class NanoState:
    """
    SSD Nano 状態ベクトル（NumPy配列ベース）
    
    v8との対応:
    - E[i]: HumanAgent.state.E[i]
    - kappa[i]: HumanAgent.state.kappa[i]
    - visible_signals[j]: ObservableSignal強度（7種類）
    """
    # エネルギー [PHYSICAL, BASE, CORE, UPPER]
    E: np.ndarray = field(default_factory=lambda: np.zeros(4))
    
    # 整合慣性 [PHYSICAL, BASE, CORE, UPPER]
    kappa: np.ndarray = field(default_factory=lambda: np.ones(4))
    
    # 観測可能なシグナル（7種類）
    # 0:FEAR, 1:ANGER, 2:COOP, 3:AGGR, 4:IDEOLOGY, 5:NORM_VIOL, 6:NORM_ADHER
    visible_signals: np.ndarray = field(default_factory=lambda: np.zeros(7))
    
    # 時間
    t: float = 0.0
    
    # 診断用（オプション）
    last_pressure: np.ndarray = field(default_factory=lambda: np.zeros(4))
    last_leap_layer: int = -1  # -1 = NO_LEAP


@dataclass
class NanoParams:
    """
    SSD Nano パラメータセット（v8ロジックベース）
    
    NumPy配列として保持し、高速な要素積アクセスを可能にする
    """
    # --- R値（動かしにくさ） [PHYSICAL, BASE, CORE, UPPER] ---
    R_values: np.ndarray = field(default_factory=lambda: np.array([1000.0, 100.0, 10.0, 1.0]))
    
    # --- エネルギー生成・減衰 ---
    gamma: np.ndarray = field(default_factory=lambda: np.array([0.15, 0.1, 0.08, 0.05]))
    beta: np.ndarray = field(default_factory=lambda: np.array([0.001, 0.01, 0.05, 0.1]))
    
    # --- κ（整合慣性）学習・減衰 ---
    eta: np.ndarray = field(default_factory=lambda: np.array([0.9, 0.5, 0.3, 0.2]))
    lambda_kappa: np.ndarray = field(default_factory=lambda: np.array([0.001, 0.01, 0.02, 0.05]))
    kappa_min: np.ndarray = field(default_factory=lambda: np.array([0.9, 0.8, 0.5, 0.3]))
    
    # --- 跳躍（Leap） ---
    Theta_base: np.ndarray = field(default_factory=lambda: np.array([200.0, 100.0, 50.0, 30.0]))
    enable_dynamic_theta: bool = True
    theta_sensitivity: float = 0.3
    
    # --- 整合流（Ohm's law） ---
    G0: float = 0.5
    g: float = 0.7
    
    # --- 層間非線形変換 (Phase 7) ---
    # 4x4 マトリクス: [target_layer, source_layer] = source→targetへの係数
    # v8では飽和効果を含む非線形関数を使用
    interlayer_base_coeffs: np.ndarray = field(default_factory=lambda: np.array([
        # to: PHYSICAL, BASE, CORE, UPPER
        [ 0.0,  0.0,  0.0,  0.0],    # from PHYSICAL
        [ 0.20, 0.0, -0.03, -0.08],  # from BASE
        [-0.05, 0.03, 0.0, -0.05],   # from CORE
        [-0.10, 0.15, 0.06, 0.0]     # from UPPER
    ]))
    # 飽和パラメータ
    saturation_E_threshold: float = 100.0
    saturation_kappa_threshold: float = 2.5
    
    # --- シグナル生成閾値 (v8 SignalGenerator) ---
    # [E_base, E_base, E_core, E_base, E_upper, E_core, kappa_core]
    signal_thresholds: np.ndarray = field(default_factory=lambda: np.array([
        0.3,   # 0: FEAR_EXPRESSION (E_base)
        1.0,   # 1: ANGER_EXPRESSION (E_base)
        1.5,   # 2: COOPERATIVE_ACT (kappa_core, E_core < 3.0)
        5.0,   # 3: AGGRESSIVE_ACT (E_base, E_upper < 1.0)
        1.0,   # 4: VERBAL_IDEOLOGY (E_upper)
        3.0,   # 5: NORM_VIOLATION (E_core, kappa_core < 1.2)
        1.8    # 6: NORM_ADHERENCE (kappa_core)
    ]))
    
    # --- 主観的社会圧力 (Phase 6/8) ---
    # 各シグナルタイプからの圧力係数 [7シグナル x 4層]
    # [FEAR, ANGER, COOP, AGGR, IDEOLOGY, NORM_VIOL, NORM_ADHER] x [PHYS, BASE, CORE, UPPER]
    signal_pressure_coeffs: np.ndarray = field(default_factory=lambda: np.array([
        [0.0, 0.80, 0.0, 0.0],   # FEAR → BASE
        [0.0, 0.60, 0.30, 0.0],  # ANGER → BASE, CORE
        [0.0, 0.0, 0.50, 0.20],  # COOP → CORE, UPPER
        [0.0, 0.50, 0.0, 0.0],   # AGGR → BASE
        [0.0, 0.0, 0.30, 0.50],  # IDEOLOGY → CORE, UPPER
        [0.0, 0.0, 0.40, 0.0],   # NORM_VIOL → CORE
        [0.0, 0.0, 0.30, 0.0]    # NORM_ADHER → CORE
    ]))
    
    # 関係性・距離による減衰
    relationship_amplification: float = 1.5  # 協力関係での増幅
    competition_suppression: float = -0.5    # 競争関係での抑制
    distance_decay: float = 0.5              # 距離による減衰


class NanoCoreEngine:
    """
    SSD Nano Core Engine v8.0
    
    v8のリファレンスロジック（主観的社会システム）を
    インライン化・高速化したプロダクションエンジン
    
    核心的特徴:
    1. 主観的観測: 他者の内部状態（E, κ）は観測不可能
    2. シグナル生成: 内部状態→観測可能なシグナル変換
    3. 主観的解釈: シグナル→自己の圧力への変換
    4. 自己変化: 圧力による自己の E, κ 更新
    """
    
    def __init__(self, params: Optional[NanoParams] = None):
        self.params = params if params else NanoParams()
        self.num_layers = 4
        self.num_signals = 7
    
    def generate_signals(self, state: NanoState) -> np.ndarray:
        """
        内部状態から観測可能なシグナルを生成（v8 SignalGenerator）
        
        Args:
            state: NanoState
            
        Returns:
            signals: np.ndarray (7,) - 各シグナルタイプの強度 [0.0, 1.0]
        """
        signals = np.zeros(7)
        E = state.E
        kappa = state.kappa
        thresholds = self.params.signal_thresholds
        
        # 0: FEAR_EXPRESSION (E_base > 0.3)
        if E[Layer.BASE] > thresholds[0]:
            signals[0] = min(E[Layer.BASE] / 10.0, 1.0)
        
        # 1: ANGER_EXPRESSION (E_base > 1.0 and E_core > 0.8)
        if E[Layer.BASE] > thresholds[1] and E[Layer.CORE] > 0.8:
            signals[1] = min((E[Layer.BASE] + E[Layer.CORE]) / 15.0, 1.0)
        
        # 2: COOPERATIVE_ACT (kappa_core > 1.5 and E_core < 3.0)
        if kappa[Layer.CORE] > thresholds[2] and E[Layer.CORE] < 3.0:
            signals[2] = min((kappa[Layer.CORE] - 1.0) / 2.0, 1.0)
        
        # 3: AGGRESSIVE_ACT (E_base > 5.0 and E_upper < 1.0)
        if E[Layer.BASE] > thresholds[3] and E[Layer.UPPER] < 1.0:
            signals[3] = min(E[Layer.BASE] / 10.0, 1.0)
        
        # 4: VERBAL_IDEOLOGY (E_upper > 1.0)
        if E[Layer.UPPER] > thresholds[4]:
            signals[4] = min(E[Layer.UPPER] / 8.0, 1.0)
        
        # 5: NORM_VIOLATION (E_core > 3.0 and kappa_core < 1.2)
        if E[Layer.CORE] > thresholds[5] and kappa[Layer.CORE] < 1.2:
            signals[5] = min(E[Layer.CORE] / 8.0, 1.0)
        
        # 6: NORM_ADHERENCE (kappa_core > 1.8)
        if kappa[Layer.CORE] > thresholds[6]:
            signals[6] = min((kappa[Layer.CORE] - 1.0) / 3.0, 1.0)
        
        return signals
    
    def interpret_signals(
        self,
        observer_state: NanoState,
        target_signals: np.ndarray,
        relationship: float,
        distance: float
    ) -> np.ndarray:
        """
        観測したシグナルを主観的に解釈し、自己への圧力に変換
        （v8 SubjectiveSocialPressureCalculator）
        
        Args:
            observer_state: 観測者の状態
            target_signals: 対象の観測可能なシグナル (7,)
            relationship: 関係性 [-1.0, 1.0]
            distance: 距離 [0.0, 1.0]
            
        Returns:
            pressure: np.ndarray (4,) - 各層への圧力
        """
        # 基本圧力: シグナル強度 × 圧力係数行列
        # (7,) × (7, 4) = (4,)
        base_pressure = np.dot(target_signals, self.params.signal_pressure_coeffs)
        
        # 関係性による調整
        if relationship > 0.5:  # 協力関係
            relationship_factor = 1.0 + relationship * self.params.relationship_amplification
        elif relationship < -0.5:  # 競争関係
            relationship_factor = 1.0 + abs(relationship) * self.params.competition_suppression
        else:  # 中立
            relationship_factor = 1.0
        
        # 距離による減衰
        distance_factor = 1.0 - distance * self.params.distance_decay
        
        # 最終圧力
        pressure = base_pressure * relationship_factor * distance_factor
        
        return np.maximum(0.0, pressure)
    
    def compute_nonlinear_transfer(self, state: NanoState) -> np.ndarray:
        """
        非線形層間変換の計算（v8 Phase 7）
        
        飽和効果を考慮:
        - E_source が大きすぎる場合、変換効率が低下
        - kappa_target が大きすぎる場合、受け入れ抵抗が増大
        
        Args:
            state: NanoState
            
        Returns:
            transfer: np.ndarray (4,) - 各層への変換量
        """
        E = state.E
        kappa = state.kappa
        base_coeffs = self.params.interlayer_base_coeffs
        
        # 飽和係数の計算
        E_saturation = 1.0 / (1.0 + E / self.params.saturation_E_threshold)
        kappa_saturation = 1.0 / (1.0 + kappa / self.params.saturation_kappa_threshold)
        
        # 非線形転送行列の構築
        # base_coeffs[i, j] × E_saturation[j] × kappa_saturation[i]
        nonlinear_coeffs = base_coeffs * E_saturation[np.newaxis, :] * kappa_saturation[:, np.newaxis]
        
        # 転送量の計算
        transfer = np.dot(nonlinear_coeffs, E)
        
        return transfer
    
    def step_single(
        self,
        state: NanoState,
        pressure: np.ndarray,
        dt: float = 0.1
    ) -> None:
        """
        単一エージェントの状態更新（インプレース）
        
        Args:
            state: NanoState（更新される）
            pressure: 外部圧力 (4,)
            dt: 時間刻み
        """
        E = state.E
        kappa = state.kappa
        params = self.params
        
        # 診断用
        state.last_pressure = pressure.copy()
        
        # --- 1. 整合流（Ohm's law） ---
        conductance = params.G0 + params.g * kappa
        j = conductance * pressure
        
        # --- 2. 跳躍判定 ---
        if params.enable_dynamic_theta:
            power = pressure * E * kappa * params.R_values
            influence = power / (kappa * params.R_values + 1e-6)
            dynamic_Theta = params.Theta_base * (1.0 - params.theta_sensitivity * np.clip(influence, 0, 1))
            dynamic_Theta = np.maximum(params.Theta_base * 0.3, dynamic_Theta)
        else:
            dynamic_Theta = params.Theta_base
        
        # 跳躍検出
        critical_layers = np.where(E >= dynamic_Theta)[0]
        state.last_leap_layer = -1
        
        if len(critical_layers) > 0:
            # 最大エネルギーの層が跳躍
            leap_layer = critical_layers[np.argmax(E[critical_layers])]
            state.last_leap_layer = leap_layer
            
            # エネルギーリセット
            E[leap_layer] *= 0.1
            # κ学習
            kappa[leap_layer] += 0.1
        
        # --- 3. エネルギー更新 ---
        # 生成: gamma * |p| / R
        energy_generation = params.gamma * np.abs(pressure) / params.R_values
        
        # 減衰
        energy_decay = params.beta * E
        
        # 層間非線形転送
        interlayer_transfer = self.compute_nonlinear_transfer(state)
        
        # dE/dt
        dE = energy_generation - energy_decay + interlayer_transfer
        state.E = np.maximum(0.0, E + dE * dt)
        
        # --- 4. κ更新 ---
        # 使用強化
        usage_factor = np.abs(j) / (np.abs(j) + 1.0)
        kappa_generation = params.eta * usage_factor
        
        # 減衰
        kappa_decay = params.lambda_kappa * kappa
        
        # dκ/dt
        dkappa = kappa_generation - kappa_decay
        state.kappa = np.maximum(params.kappa_min, kappa + dkappa * dt)
        
        # --- 5. 時間更新 ---
        state.t += dt
        
        # --- 6. シグナル更新 ---
        state.visible_signals = self.generate_signals(state)
    
    def step_batch(
        self,
        states: List[NanoState],
        external_pressures: List[np.ndarray],
        relationships: Optional[np.ndarray] = None,
        distances: Optional[np.ndarray] = None,
        dt: float = 0.1
    ) -> None:
        """
        複数エージェントのバッチ処理（v8主観的社会システム）
        
        プロセス:
        1. 全エージェントのシグナル生成
        2. 各エージェントが他者を観測→解釈→社会的圧力を計算
        3. 外部圧力 + 社会的圧力で状態更新
        
        Args:
            states: NanoStateのリスト（インプレース更新）
            external_pressures: 各エージェントの外部圧力（環境・タスク等）
            relationships: 関係性マトリクス (N, N) [-1.0, 1.0]
            distances: 距離マトリクス (N, N) [0.0, 1.0]
            dt: 時間刻み
        """
        num_agents = len(states)
        
        if len(external_pressures) != num_agents:
            raise ValueError("external_pressures の数が states と一致しません")
        
        # デフォルト値
        if relationships is None:
            relationships = np.zeros((num_agents, num_agents))
        if distances is None:
            distances = np.zeros((num_agents, num_agents))
        
        # --- フェーズ1: シグナル生成 ---
        all_signals = np.zeros((num_agents, self.num_signals))
        for i, state in enumerate(states):
            all_signals[i] = self.generate_signals(state)
        
        # --- フェーズ2: 主観的観測→解釈→社会的圧力計算 ---
        social_pressures = np.zeros((num_agents, self.num_layers))
        
        for i, observer_state in enumerate(states):
            for j in range(num_agents):
                if i == j:
                    continue  # 自分自身は観測しない
                
                # 対象のシグナル
                target_signals = all_signals[j]
                
                # シグナルが微弱なら無視
                if np.max(target_signals) < 0.01:
                    continue
                
                # 主観的解釈
                pressure = self.interpret_signals(
                    observer_state,
                    target_signals,
                    relationships[i, j],
                    distances[i, j]
                )
                
                # 累積
                social_pressures[i] += pressure
        
        # --- フェーズ3: 個体更新 ---
        for i, state in enumerate(states):
            # 総圧力 = 外部圧力 + 社会的圧力
            total_pressure = external_pressures[i] + social_pressures[i]
            
            # 個体ステップ
            self.step_single(state, total_pressure, dt)


# ========================================
# デモ
# ========================================

def nano_demo_v8():
    """Nano Core Engine v8.0 のデモ"""
    print("=" * 80)
    print(" " * 20 + "SSD Nano Core Engine v8.0 - デモ")
    print("=" * 80)
    
    print("\n設計思想:")
    print("  - リファレンス実装（ssd_core_engine + ssd_human_module）: 理論の完全性")
    print("  - Nano実装（nano_core_engine）: 物理層制約との整合（パフォーマンス）")
    print("  - v8理論の核心（主観的観測→解釈→自己変化）を維持")
    
    # --- パラメータとエンジン ---
    params = NanoParams()
    engine = NanoCoreEngine(params)
    
    # --- エージェント準備 ---
    print("\n" + "=" * 80)
    print("[シナリオ] 主観的恐怖伝染（v8ロジック）")
    print("=" * 80)
    
    num_agents = 3
    states = [NanoState() for _ in range(num_agents)]
    
    # Agent_0: 高い恐怖（E_base）
    states[0].E[Layer.BASE] = 0.9
    
    # 関係性: 全員協力
    relationships = np.array([
        [0.0, 0.8, 0.8],
        [0.8, 0.0, 0.8],
        [0.8, 0.8, 0.0]
    ])
    
    # 距離: 全員近接
    distances = np.zeros((num_agents, num_agents))
    
    print("\n初期状態:")
    for i, state in enumerate(states):
        print(f"  Agent_{i} E_base: {state.E[Layer.BASE]:.6f}")
    
    # Agent_0のシグナル確認
    signals_0 = engine.generate_signals(states[0])
    print(f"\nAgent_0 の観測可能なシグナル:")
    signal_names = ["FEAR", "ANGER", "COOP", "AGGR", "IDEOLOGY", "NORM_VIOL", "NORM_ADHER"]
    for i, name in enumerate(signal_names):
        if signals_0[i] > 0:
            print(f"  {name}: {signals_0[i]:.4f}")
    
    # --- シミュレーション ---
    print(f"\nシミュレーション実行（100ステップ）...")
    
    external_pressures = [np.zeros(4) for _ in range(num_agents)]
    
    for step in range(100):
        engine.step_batch(states, external_pressures, relationships, distances, dt=0.1)
    
    print(f"\n最終状態（Step 100）:")
    for i, state in enumerate(states):
        print(f"  Agent_{i} E_base: {state.E[Layer.BASE]:.6f}")
    
    # 結果検証
    e1_final = states[1].E[Layer.BASE]
    e2_final = states[2].E[Layer.BASE]
    
    print(f"\n結果:")
    if e1_final > 0.0001 and e2_final > 0.0001:
        print(f"  ✅ 恐怖伝染成功！（主観的観測→解釈→自己変化）")
        print(f"     Agent_1: 0.0 → {e1_final:.6f}")
        print(f"     Agent_2: 0.0 → {e2_final:.6f}")
    else:
        print(f"  ⚠️ 恐怖伝染が観測されませんでした")
    
    # --- パフォーマンス比較 ---
    print("\n" + "=" * 80)
    print("[パフォーマンス指標]")
    print("=" * 80)
    
    import time
    
    # 大規模シミュレーション
    num_agents_large = 100
    states_large = [NanoState() for _ in range(num_agents_large)]
    states_large[0].E[Layer.BASE] = 1.0  # 1人が恐怖
    
    relationships_large = np.random.rand(num_agents_large, num_agents_large) * 0.5
    np.fill_diagonal(relationships_large, 0.0)
    distances_large = np.random.rand(num_agents_large, num_agents_large) * 0.3
    external_pressures_large = [np.zeros(4) for _ in range(num_agents_large)]
    
    print(f"\n{num_agents_large}エージェント × 100ステップ のベンチマーク...")
    
    start_time = time.time()
    for step in range(100):
        engine.step_batch(
            states_large,
            external_pressures_large,
            relationships_large,
            distances_large,
            dt=0.1
        )
    elapsed = time.time() - start_time
    
    print(f"\n実行時間: {elapsed:.3f} 秒")
    print(f"スループット: {num_agents_large * 100 / elapsed:.1f} agent-steps/sec")
    print(f"平均フレーム時間: {elapsed / 100 * 1000:.2f} ms/frame")
    
    print("\n" + "=" * 80)
    print("理論的整合性: v8「主観的社会システム」100%維持")
    print("パフォーマンス: リアルタイムAI・群衆シミュレーション対応")
    print("=" * 80)


if __name__ == "__main__":
    nano_demo_v8()
