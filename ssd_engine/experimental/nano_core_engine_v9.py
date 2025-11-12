"""
SSD v9.0: Nano Core Engine with Dynamic Interpretation
動的解釈構造のパフォーマンス最適化実装

理論的背景:
- Reference実装 (ssd_dynamic_interpretation.py): 理論完全性
- Nano実装 (このファイル): パフォーマンス最適化

最適化戦略:
1. 固定長メモリバッファ（動的リストの代わり）
2. Numba JITコンパイル
3. ベクトル化された記憶検索
4. インライン計算

目標パフォーマンス:
- 100 agents × 100 steps < 2.5s (v8: 1.667s)
- メモリオーバーヘッド < 30%

作成日: 2025年11月7日
バージョン: 9.0
"""

import numpy as np
from numba import njit, prange
from typing import Tuple
from dataclasses import dataclass
import time


@dataclass
class NanoStateV9:
    """
    Nanoエージェントの状態（v9: メモリ付き）
    
    Attributes:
        E: エネルギー [4] - PHYSICAL, BASE, CORE, UPPER
        kappa: 整合慣性 [4]
        visible_signals: 可視シグナル [7]
        memory: 記憶バッファ [max_memories, memory_dim]
                memory_dim = 7(signal) + 1(layer) + 1(outcome) + 1(timestamp)
        memory_count: 実際に使用されている記憶数
    """
    E: np.ndarray  # [4]
    kappa: np.ndarray  # [4]
    visible_signals: np.ndarray  # [7]
    memory: np.ndarray  # [max_memories, 10]
    memory_count: int


@dataclass
class NanoParamsV9:
    """
    Nanoエンジンのパラメータ（v9: 動的解釈用）
    
    Attributes:
        base_signal_pressure_coeffs: 基本解釈係数 [4, 7]
        learning_rate: 学習強度
        tau_memory: 記憶の減衰時定数
        signal_generation_coeffs: E→シグナル変換係数 [7, 4]
        energy_decay: エネルギー減衰率 [4]
        kappa_growth: κ成長率 [4]
        R_values: R値 [4]
    """
    base_signal_pressure_coeffs: np.ndarray  # [4, 7]
    learning_rate: float
    tau_memory: float
    signal_generation_coeffs: np.ndarray  # [7, 4]
    energy_decay: np.ndarray  # [4]
    kappa_growth: np.ndarray  # [4]
    R_values: np.ndarray  # [4]


# ================================================================================
# Numba JIT コンパイル関数
# ================================================================================

@njit
def compute_dynamic_coeffs_batch(
    base_coeffs: np.ndarray,  # [4, 7]
    kappa: np.ndarray,  # [n_agents, 4]
    memories: np.ndarray,  # [n_agents, max_memories, 10]
    memory_counts: np.ndarray,  # [n_agents]
    t_now: float,
    learning_rate: float,
    tau_memory: float
) -> np.ndarray:
    """
    バッチ処理で動的解釈係数を計算（Numba最適化）
    
    Args:
        base_coeffs: 基本係数 [4, 7]
        kappa: 各エージェントのκ [n_agents, 4]
        memories: 記憶バッファ [n_agents, max_memories, 10]
                  memory[i, j] = [sig0...sig6, layer, outcome, timestamp]
        memory_counts: 各エージェントの記憶数 [n_agents]
        t_now: 現在時刻
        learning_rate: 学習強度
        tau_memory: 記憶減衰
    
    Returns:
        動的係数 [n_agents, 4, 7]
    """
    n_agents = kappa.shape[0]
    n_layers = 4
    n_signals = 7
    
    # 出力バッファ
    dynamic_coeffs = np.zeros((n_agents, n_layers, n_signals), dtype=np.float64)
    
    # 各エージェント
    for agent_idx in prange(n_agents):
        # 基本係数をコピー
        for layer in range(n_layers):
            for sig in range(n_signals):
                dynamic_coeffs[agent_idx, layer, sig] = base_coeffs[layer, sig]
        
        # 記憶から学習項を計算
        n_mems = int(memory_counts[agent_idx])
        
        if n_mems > 0:
            for layer in range(n_layers):
                for sig in range(n_signals):
                    learning_term = 0.0
                    
                    # 関連する記憶を探索
                    for mem_idx in range(n_mems):
                        mem_layer = int(memories[agent_idx, mem_idx, 7])
                        
                        # 同じ層の記憶のみ
                        if mem_layer == layer:
                            # シグナル強度
                            signal_strength = memories[agent_idx, mem_idx, sig]
                            
                            # 結果の影響（悪い結果ほど警戒↑）
                            outcome = memories[agent_idx, mem_idx, 8]
                            impact = -outcome
                            
                            # 時間減衰
                            timestamp = memories[agent_idx, mem_idx, 9]
                            age = t_now - timestamp
                            decay = np.exp(-age / tau_memory)
                            
                            # 累積
                            learning_term += signal_strength * impact * decay
                    
                    # κによる定着度
                    learned_value = (
                        base_coeffs[layer, sig] + 
                        kappa[agent_idx, layer] * learning_rate * learning_term
                    )
                    
                    # クリップ（スカラー用）
                    if learned_value < 0.0:
                        learned_value = 0.0
                    elif learned_value > 1.0:
                        learned_value = 1.0
                    
                    dynamic_coeffs[agent_idx, layer, sig] = learned_value
    
    return dynamic_coeffs


@njit
def interpret_signals_batch_dynamic(
    observer_states: np.ndarray,  # [n_observers, 4] (kappa)
    target_signals: np.ndarray,  # [n_targets, 7]
    dynamic_coeffs: np.ndarray,  # [n_observers, 4, 7]
) -> np.ndarray:
    """
    バッチ動的解釈: 全観測者×全対象の圧力を計算
    
    Args:
        observer_states: 観測者のκ [n_observers, 4]
        target_signals: 対象のシグナル [n_targets, 7]
        dynamic_coeffs: 動的解釈係数 [n_observers, 4, 7]
    
    Returns:
        圧力テンソル [n_observers, n_targets, 4]
    """
    n_observers = observer_states.shape[0]
    n_targets = target_signals.shape[0]
    n_layers = 4
    
    pressure = np.zeros((n_observers, n_targets, n_layers), dtype=np.float64)
    
    for obs_idx in prange(n_observers):
        for tgt_idx in range(n_targets):
            for layer in range(n_layers):
                # 内積: coeffs[layer] · signals
                pressure[obs_idx, tgt_idx, layer] = np.dot(
                    dynamic_coeffs[obs_idx, layer],
                    target_signals[tgt_idx]
                )
    
    return pressure


@njit
def add_memory_batch(
    memories: np.ndarray,  # [n_agents, max_memories, 10]
    memory_counts: np.ndarray,  # [n_agents]
    agent_indices: np.ndarray,  # [n_events]
    signal_patterns: np.ndarray,  # [n_events, 7]
    layers: np.ndarray,  # [n_events]
    outcomes: np.ndarray,  # [n_events]
    timestamp: float,
    max_memories: int
) -> np.ndarray:
    """
    バッチで記憶を追加
    
    Args:
        memories: 記憶バッファ [n_agents, max_memories, 10]
        memory_counts: 記憶数 [n_agents]
        agent_indices: イベント対象のエージェント [n_events]
        signal_patterns: シグナル [n_events, 7]
        layers: 層 [n_events]
        outcomes: 結果 [n_events]
        timestamp: 時刻
        max_memories: 最大記憶数
    
    Returns:
        更新された記憶数 [n_agents]
    """
    n_events = agent_indices.shape[0]
    new_counts = memory_counts.copy()
    
    for i in range(n_events):
        agent_idx = agent_indices[i]
        count = int(new_counts[agent_idx])
        
        if count < max_memories:
            # 新しい記憶を追加
            mem_idx = count
        else:
            # 古い記憶を上書き（FIFO）
            mem_idx = count % max_memories
        
        # 記憶を書き込み
        for sig in range(7):
            memories[agent_idx, mem_idx, sig] = signal_patterns[i, sig]
        
        memories[agent_idx, mem_idx, 7] = layers[i]
        memories[agent_idx, mem_idx, 8] = outcomes[i]
        memories[agent_idx, mem_idx, 9] = timestamp
        
        # カウント更新
        if count < max_memories:
            new_counts[agent_idx] += 1
    
    return new_counts


@njit
def generate_signals_batch(
    E: np.ndarray,  # [n_agents, 4]
    kappa: np.ndarray,  # [n_agents, 4]
    coeffs: np.ndarray  # [7, 4]
) -> np.ndarray:
    """
    バッチでシグナルを生成
    
    Args:
        E: エネルギー [n_agents, 4]
        kappa: κ [n_agents, 4]
        coeffs: 生成係数 [7, 4]
    
    Returns:
        シグナル [n_agents, 7]
    """
    n_agents = E.shape[0]
    signals = np.zeros((n_agents, 7), dtype=np.float64)
    
    for i in prange(n_agents):
        for sig in range(7):
            # 線形結合: E[0]*c[0] + E[1]*c[1] + ...
            for layer in range(4):
                signals[i, sig] += E[i, layer] * coeffs[sig, layer]
            
            # κによる安定化
            avg_kappa = np.mean(kappa[i])
            signals[i, sig] *= (1.0 + avg_kappa * 0.2)
    
    return signals


@njit
def step_batch_v9(
    E: np.ndarray,  # [n_agents, 4]
    kappa: np.ndarray,  # [n_agents, 4]
    signals: np.ndarray,  # [n_agents, 7]
    pressures: np.ndarray,  # [n_agents, 4] (社会圧力の総和)
    params: Tuple,  # (energy_decay, kappa_growth, R_values)
    dt: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    v9バッチステップ: E, κの時間発展
    
    Args:
        E: エネルギー [n_agents, 4]
        kappa: κ [n_agents, 4]
        signals: シグナル [n_agents, 7]
        pressures: 社会圧力 [n_agents, 4]
        params: (energy_decay, kappa_growth, R_values)
        dt: 時間刻み
    
    Returns:
        (new_E, new_kappa)
    """
    energy_decay, kappa_growth, R_values = params
    
    n_agents = E.shape[0]
    new_E = E.copy()
    new_kappa = kappa.copy()
    
    for i in prange(n_agents):
        for layer in range(4):
            # 構造的影響力
            structural_influence = pressures[i, layer] * E[i, layer] * kappa[i, layer] * R_values[layer]
            
            # エネルギー更新
            dE = -energy_decay[layer] * E[i, layer] + structural_influence
            new_E[i, layer] = E[i, layer] + dE * dt
            
            # 跳躍判定（簡易版）
            if abs(structural_influence) > 1.0:
                leap = np.sign(structural_influence) * 0.5
                new_E[i, layer] += leap
            
            # κ更新
            dkappa = kappa_growth[layer] * abs(structural_influence)
            new_kappa[i, layer] = kappa[i, layer] + dkappa * dt
            
            # クリップ（スカラー用）
            if new_E[i, layer] < -2.0:
                new_E[i, layer] = -2.0
            elif new_E[i, layer] > 2.0:
                new_E[i, layer] = 2.0
            
            if new_kappa[i, layer] < 0.0:
                new_kappa[i, layer] = 0.0
            elif new_kappa[i, layer] > 1.0:
                new_kappa[i, layer] = 1.0
    
    return new_E, new_kappa


# ================================================================================
# NanoCoreEngineV9 クラス
# ================================================================================

class NanoCoreEngineV9:
    """
    Nano Core Engine v9.0: 動的解釈のパフォーマンス最適化実装
    """
    
    @staticmethod
    def create_default_params() -> NanoParamsV9:
        """デフォルトパラメータを作成"""
        
        # 基本解釈係数
        base_coeffs = np.array([
            [0.1, 0.1, 0.2, 0.3, 0.2, 0.1, 0.0],  # PHYSICAL
            [0.2, 0.3, 0.5, 0.4, 0.3, 0.2, 0.1],  # BASE
            [0.1, 0.2, 0.3, 0.5, 0.4, 0.4, 0.2],  # CORE
            [0.0, 0.1, 0.2, 0.3, 0.3, 0.5, 0.6],  # UPPER
        ], dtype=np.float64)
        
        # シグナル生成係数
        signal_gen = np.array([
            [0.3, 0.2, 0.1, 0.0],  # signal 0 ← E
            [0.2, 0.4, 0.2, 0.1],  # signal 1
            [0.1, 0.5, 0.3, 0.2],  # signal 2
            [0.1, 0.3, 0.5, 0.3],  # signal 3
            [0.1, 0.2, 0.4, 0.4],  # signal 4
            [0.0, 0.1, 0.3, 0.5],  # signal 5
            [0.0, 0.1, 0.2, 0.6],  # signal 6
        ], dtype=np.float64)
        
        return NanoParamsV9(
            base_signal_pressure_coeffs=base_coeffs,
            learning_rate=0.6,
            tau_memory=50.0,
            signal_generation_coeffs=signal_gen,
            energy_decay=np.array([0.05, 0.03, 0.02, 0.01], dtype=np.float64),
            kappa_growth=np.array([0.01, 0.02, 0.03, 0.02], dtype=np.float64),
            R_values=np.array([1e10, 100.0, 10.0, 1.0], dtype=np.float64)
        )
    
    @staticmethod
    def initialize_states(n_agents: int, max_memories: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        エージェント状態を初期化
        
        Returns:
            (E, kappa, memories, memory_counts)
        """
        E = np.random.rand(n_agents, 4).astype(np.float64) * 0.5 + 0.5
        kappa = np.random.rand(n_agents, 4).astype(np.float64) * 0.3 + 0.2
        memories = np.zeros((n_agents, max_memories, 10), dtype=np.float64)
        memory_counts = np.zeros(n_agents, dtype=np.int32)
        
        return E, kappa, memories, memory_counts
    
    @staticmethod
    def step_society(
        E: np.ndarray,  # [n_agents, 4]
        kappa: np.ndarray,  # [n_agents, 4]
        memories: np.ndarray,  # [n_agents, max_memories, 10]
        memory_counts: np.ndarray,  # [n_agents]
        params: NanoParamsV9,
        t_now: float,
        dt: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        社会全体の1ステップ更新
        
        Returns:
            (new_E, new_kappa)
        """
        n_agents = E.shape[0]
        
        # Phase 1: シグナル生成
        signals = generate_signals_batch(E, kappa, params.signal_generation_coeffs)
        
        # Phase 2: 動的解釈係数の計算
        dynamic_coeffs = compute_dynamic_coeffs_batch(
            params.base_signal_pressure_coeffs,
            kappa,
            memories,
            memory_counts,
            t_now,
            params.learning_rate,
            params.tau_memory
        )
        
        # Phase 3: 社会圧力の計算
        pressure_tensor = interpret_signals_batch_dynamic(
            kappa,  # observer states
            signals,  # target signals
            dynamic_coeffs
        )
        
        # 各エージェントへの総圧力（自分以外から）
        total_pressure = np.zeros((n_agents, 4), dtype=np.float64)
        for i in range(n_agents):
            for j in range(n_agents):
                if i != j:
                    total_pressure[i] += pressure_tensor[i, j]
        
        # Phase 4: E, κの更新
        params_tuple = (params.energy_decay, params.kappa_growth, params.R_values)
        new_E, new_kappa = step_batch_v9(
            E, kappa, signals, total_pressure, params_tuple, dt
        )
        
        return new_E, new_kappa


# ================================================================================
# ベンチマーク
# ================================================================================

def benchmark_v9():
    """v9のベンチマーク"""
    print("="*70)
    print("Nano Core Engine v9.0 - ベンチマーク")
    print("="*70)
    
    params = NanoCoreEngineV9.create_default_params()
    
    # 小規模（7 agents）
    print("\n【小規模: 7 agents × 100 steps】")
    n_agents = 7
    n_steps = 100
    max_memories = 50
    
    E, kappa, memories, memory_counts = NanoCoreEngineV9.initialize_states(n_agents, max_memories)
    
    start = time.time()
    for step in range(n_steps):
        t_now = step * 0.1
        E, kappa = NanoCoreEngineV9.step_society(
            E, kappa, memories, memory_counts, params, t_now
        )
        
        # ランダムに記憶を追加（学習シミュレーション）
        if step % 10 == 0 and step > 0:
            n_events = np.random.randint(1, 4)
            agent_indices = np.random.randint(0, n_agents, n_events)
            signal_patterns = np.random.rand(n_events, 7)
            layers = np.random.randint(0, 4, n_events)
            outcomes = np.random.randn(n_events) * 0.5
            
            memory_counts = add_memory_batch(
                memories, memory_counts, agent_indices,
                signal_patterns, layers, outcomes, t_now, max_memories
            )
    
    elapsed = time.time() - start
    print(f"実行時間: {elapsed*1000:.2f} ms")
    print(f"最終記憶数: 平均 {np.mean(memory_counts):.1f}")
    
    # 中規模（20 agents）
    print("\n【中規模: 20 agents × 100 steps】")
    n_agents = 20
    E, kappa, memories, memory_counts = NanoCoreEngineV9.initialize_states(n_agents, max_memories)
    
    start = time.time()
    for step in range(n_steps):
        t_now = step * 0.1
        E, kappa = NanoCoreEngineV9.step_society(
            E, kappa, memories, memory_counts, params, t_now
        )
        
        if step % 10 == 0 and step > 0:
            n_events = np.random.randint(2, 6)
            agent_indices = np.random.randint(0, n_agents, n_events)
            signal_patterns = np.random.rand(n_events, 7)
            layers = np.random.randint(0, 4, n_events)
            outcomes = np.random.randn(n_events) * 0.5
            
            memory_counts = add_memory_batch(
                memories, memory_counts, agent_indices,
                signal_patterns, layers, outcomes, t_now, max_memories
            )
    
    elapsed = time.time() - start
    print(f"実行時間: {elapsed*1000:.2f} ms")
    
    # 大規模（100 agents）
    print("\n【大規模: 100 agents × 100 steps】")
    n_agents = 100
    E, kappa, memories, memory_counts = NanoCoreEngineV9.initialize_states(n_agents, max_memories)
    
    start = time.time()
    for step in range(n_steps):
        t_now = step * 0.1
        E, kappa = NanoCoreEngineV9.step_society(
            E, kappa, memories, memory_counts, params, t_now
        )
        
        if step % 10 == 0 and step > 0:
            n_events = np.random.randint(5, 15)
            agent_indices = np.random.randint(0, n_agents, n_events)
            signal_patterns = np.random.rand(n_events, 7)
            layers = np.random.randint(0, 4, n_events)
            outcomes = np.random.randn(n_events) * 0.5
            
            memory_counts = add_memory_batch(
                memories, memory_counts, agent_indices,
                signal_patterns, layers, outcomes, t_now, max_memories
            )
    
    elapsed = time.time() - start
    print(f"実行時間: {elapsed:.3f} s")
    print(f"スループット: {n_agents * n_steps / elapsed:.0f} agent-steps/sec")
    print(f"最終記憶数: 平均 {np.mean(memory_counts):.1f}")
    
    print("\n" + "="*70)
    print("✅ v9.0ベンチマーク完了")
    print("="*70)


if __name__ == "__main__":
    benchmark_v9()
