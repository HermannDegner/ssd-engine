"""
SSD v9.0: Dynamic Interpretation Module
動的解釈構造 - κと記憶に基づく主観の可塑性

理論的背景:
- v8.5までの限界: 解釈係数が静的（固定）
- v9.0の革新: κ（整合慣性）と記憶に基づき、解釈構造自体が変化
- 哲学的意義: フッサールの「地平（Horizont）」の変容を数理的にモデル化

主要クラス:
- MemoryTrace: 単一の経験記録
- MemoryStore: 経験の蓄積・検索
- DynamicInterpretationMatrix: κ依存的解釈係数
- DynamicInterpretationModule: 統合モジュール

作成日: 2025年11月7日
バージョン: 9.0
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import time


@dataclass
class MemoryTrace:
    """
    単一の経験記録
    
    Attributes:
        signal_pattern: 観測されたシグナルベクトル [7]
        layer: 影響を受けた層 (0=PHYSICAL, 1=BASE, 2=CORE, 3=UPPER)
        interpreted_pressure: 当時の解釈による圧力値
        outcome: 結果の評価 (-1.0=最悪, 0.0=中立, +1.0=最良)
        timestamp: 経験の時刻
        context: 追加のコンテキスト情報（オプション）
    """
    signal_pattern: np.ndarray  # [7]
    layer: int  # 0-3
    interpreted_pressure: float
    outcome: float  # -1.0 to +1.0
    timestamp: float
    context: Optional[Dict] = None
    
    def get_signal_strength(self, signal_idx: int) -> float:
        """特定シグナルの強度を取得"""
        return self.signal_pattern[signal_idx]
    
    def age(self, current_time: float) -> float:
        """記憶の経過時間"""
        return current_time - self.timestamp
    
    def importance(self) -> float:
        """記憶の重要度（結果の絶対値）"""
        return abs(self.outcome)


class MemoryStore:
    """
    経験の蓄積・検索システム
    
    記憶は時間とともに減衰し、重要度の低い記憶は自動的に削除される。
    """
    
    def __init__(self, 
                 max_memories: int = 1000,
                 importance_threshold: float = 0.1,
                 tau_decay: float = 100.0):
        """
        Args:
            max_memories: 保持する最大記憶数
            importance_threshold: この重要度以下の記憶は削除対象
            tau_decay: 記憶の減衰時定数
        """
        self.memories: List[MemoryTrace] = []
        self.max_memories = max_memories
        self.importance_threshold = importance_threshold
        self.tau_decay = tau_decay
    
    def add_memory(self, memory: MemoryTrace):
        """新しい記憶を追加"""
        self.memories.append(memory)
        
        # メモリ管理: 古くて重要度の低い記憶を削除
        if len(self.memories) > self.max_memories:
            self._prune_memories(memory.timestamp)
    
    def _prune_memories(self, current_time: float):
        """重要度の低い記憶を削除"""
        # 各記憶の「保持価値」を計算
        scores = []
        for mem in self.memories:
            age = current_time - mem.timestamp
            decay = np.exp(-age / self.tau_decay)
            score = mem.importance() * decay
            scores.append(score)
        
        # 保持価値の高い順にソート
        indices = np.argsort(scores)[::-1]
        self.memories = [self.memories[i] for i in indices[:self.max_memories]]
    
    def query_by_signal(self, 
                       signal_idx: int, 
                       layer: int, 
                       current_time: float,
                       min_strength: float = 0.1) -> List[MemoryTrace]:
        """
        特定のシグナルと層に関連する記憶を検索
        
        Args:
            signal_idx: シグナルインデックス (0-6)
            layer: 層インデックス (0-3)
            current_time: 現在時刻
            min_strength: 最小シグナル強度
        
        Returns:
            関連する記憶のリスト（新しい順）
        """
        relevant = []
        for mem in self.memories:
            if (mem.layer == layer and 
                mem.get_signal_strength(signal_idx) >= min_strength):
                relevant.append(mem)
        
        # 新しい順にソート
        relevant.sort(key=lambda m: m.timestamp, reverse=True)
        return relevant
    
    def get_all_memories(self) -> List[MemoryTrace]:
        """全記憶を取得"""
        return self.memories.copy()
    
    def clear(self):
        """全記憶を消去"""
        self.memories.clear()
    
    def __len__(self) -> int:
        return len(self.memories)


class DynamicInterpretationMatrix:
    """
    動的解釈行列
    
    基本係数 + κ依存的学習項 → 経験によって変化する主観
    """
    
    def __init__(self, 
                 base_coeffs: np.ndarray,
                 learning_rate: float = 0.5,
                 tau_memory: float = 100.0):
        """
        Args:
            base_coeffs: 基本解釈係数 [4 layers, 7 signals]
            learning_rate: 学習の強度（κとの結合係数）
            tau_memory: 記憶の影響の減衰時定数
        """
        self.base_coeffs = base_coeffs.copy()  # [4, 7]
        self.learning_rate = learning_rate
        self.tau_memory = tau_memory
        
        # 現在の動的係数（初期値=基本係数）
        self.current_coeffs = base_coeffs.copy()
    
    def compute_learning_term(self,
                             layer: int,
                             signal_idx: int,
                             memories: List[MemoryTrace],
                             current_time: float) -> float:
        """
        記憶から学習項を計算
        
        学習項 = Σ (シグナル強度 × 結果の影響 × 時間減衰)
        
        Args:
            layer: 対象の層
            signal_idx: 対象のシグナル
            memories: 関連する記憶のリスト
            current_time: 現在時刻
        
        Returns:
            学習項（係数への追加値）
        """
        learning_term = 0.0
        
        for mem in memories:
            # シグナルの強度（このシグナルがどれだけ強かったか）
            signal_strength = mem.get_signal_strength(signal_idx)
            
            # 結果の影響（悪い結果ほど警戒を強化）
            # outcome が -1.0 (最悪) → impact = +1.0 (係数を上げる)
            # outcome が +1.0 (最良) → impact = -1.0 (係数を下げる)
            impact = -mem.outcome
            
            # 時間減衰（古い記憶ほど影響が小さい）
            age = current_time - mem.timestamp
            decay = np.exp(-age / self.tau_memory)
            
            # 累積
            learning_term += signal_strength * impact * decay
        
        return learning_term
    
    def update_matrix(self,
                     kappa: np.ndarray,
                     memory_store: MemoryStore,
                     current_time: float):
        """
        κと記憶に基づき、解釈行列を更新
        
        I[layer, signal] = base + κ[layer] × learning_rate × learning_term
        
        Args:
            kappa: 各層のκ値 [4]
            memory_store: 記憶ストア
            current_time: 現在時刻
        """
        for layer in range(4):
            for signal_idx in range(7):
                # この層・シグナルに関連する記憶を検索
                relevant_memories = memory_store.query_by_signal(
                    signal_idx, layer, current_time
                )
                
                # 学習項を計算
                learning_term = self.compute_learning_term(
                    layer, signal_idx, relevant_memories, current_time
                )
                
                # κによる定着度を考慮して係数を更新
                # κが高い = 経験をよく覚えている = 学習が強く反映される
                learned_coeff = (
                    self.base_coeffs[layer, signal_idx] + 
                    kappa[layer] * self.learning_rate * learning_term
                )
                
                # [0, 1]にクリップ
                self.current_coeffs[layer, signal_idx] = np.clip(
                    learned_coeff, 0.0, 1.0
                )
    
    def get_coeffs(self) -> np.ndarray:
        """現在の解釈係数を取得"""
        return self.current_coeffs.copy()
    
    def interpret_signals(self, signals: np.ndarray) -> np.ndarray:
        """
        シグナルを解釈して層別圧力を計算
        
        Args:
            signals: シグナルベクトル [7]
        
        Returns:
            層別圧力 [4]
        """
        pressure = np.zeros(4)
        for layer in range(4):
            pressure[layer] = np.dot(self.current_coeffs[layer], signals)
        return pressure


class DynamicInterpretationModule:
    """
    動的解釈モジュール（統合インターフェース）
    
    記憶の蓄積、κに基づく学習、動的解釈の一元管理
    """
    
    def __init__(self,
                 base_coeffs: np.ndarray,
                 learning_rate: float = 0.5,
                 tau_memory: float = 100.0,
                 max_memories: int = 1000):
        """
        Args:
            base_coeffs: 基本解釈係数 [4, 7]
            learning_rate: 学習強度
            tau_memory: 記憶の減衰時定数
            max_memories: 最大記憶数
        """
        self.memory_store = MemoryStore(
            max_memories=max_memories,
            tau_decay=tau_memory
        )
        
        self.interpretation_matrix = DynamicInterpretationMatrix(
            base_coeffs=base_coeffs,
            learning_rate=learning_rate,
            tau_memory=tau_memory
        )
        
        self.current_time = 0.0
    
    def interpret_signals(self,
                         signals: np.ndarray,
                         kappa: np.ndarray,
                         update_matrix: bool = True) -> np.ndarray:
        """
        動的解釈によるシグナル→圧力変換
        
        Args:
            signals: シグナルベクトル [7]
            kappa: 各層のκ値 [4]
            update_matrix: 解釈行列を更新するか
        
        Returns:
            層別圧力 [4]
        """
        # 必要に応じて解釈行列を更新
        if update_matrix:
            self.interpretation_matrix.update_matrix(
                kappa, self.memory_store, self.current_time
            )
        
        # 解釈
        pressure = self.interpretation_matrix.interpret_signals(signals)
        return pressure
    
    def record_experience(self,
                         signal_pattern: np.ndarray,
                         layer: int,
                         interpreted_pressure: float,
                         outcome: float,
                         context: Optional[Dict] = None):
        """
        経験を記憶として記録
        
        Args:
            signal_pattern: 観測されたシグナル [7]
            layer: 影響を受けた層
            interpreted_pressure: 当時の解釈による圧力
            outcome: 結果の評価 (-1.0 ~ +1.0)
            context: 追加情報
        """
        memory = MemoryTrace(
            signal_pattern=signal_pattern.copy(),
            layer=layer,
            interpreted_pressure=interpreted_pressure,
            outcome=outcome,
            timestamp=self.current_time,
            context=context
        )
        self.memory_store.add_memory(memory)
    
    def advance_time(self, dt: float = 1.0):
        """時間を進める"""
        self.current_time += dt
    
    def get_current_coeffs(self) -> np.ndarray:
        """現在の解釈係数を取得"""
        return self.interpretation_matrix.get_coeffs()
    
    def get_memory_count(self) -> int:
        """記憶数を取得"""
        return len(self.memory_store)
    
    def get_memories(self) -> List[MemoryTrace]:
        """全記憶を取得"""
        return self.memory_store.get_all_memories()
    
    def clear_memories(self):
        """全記憶を消去"""
        self.memory_store.clear()
        self.interpretation_matrix.current_coeffs = \
            self.interpretation_matrix.base_coeffs.copy()


# ================================================================================
# デモ・テスト用関数
# ================================================================================

def demo_learning_cycle():
    """
    学習サイクルのデモンストレーション
    
    シナリオ:
    1. エージェントがシグナル3（攻撃的行動）を受ける
    2. 初回: 低い警戒（係数0.3）で解釈 → 悪い結果（outcome=-1.0）
    3. 学習後: 高い警戒（係数上昇）で解釈 → 防衛成功
    """
    print("="*70)
    print("動的解釈モジュール - 学習サイクルのデモ")
    print("="*70)
    
    # 基本解釈係数（初期値）
    base_coeffs = np.array([
        [0.1, 0.1, 0.2, 0.3, 0.2, 0.1, 0.1],  # PHYSICAL層
        [0.2, 0.3, 0.5, 0.3, 0.4, 0.2, 0.1],  # BASE層（生存本能）
        [0.1, 0.2, 0.3, 0.5, 0.6, 0.3, 0.2],  # CORE層（社会的価値）
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.4],  # UPPER層（理念）
    ])
    
    # モジュールの初期化
    module = DynamicInterpretationModule(
        base_coeffs=base_coeffs,
        learning_rate=0.8,
        tau_memory=50.0,
        max_memories=100
    )
    
    # 初期κ（経験が浅い）
    kappa = np.array([0.3, 0.3, 0.3, 0.3])
    
    print("\n【初期状態】")
    print(f"κ: {kappa}")
    print(f"BASE層のシグナル3への係数: {base_coeffs[1, 3]:.3f}")
    
    # ===== 第1ラウンド: 学習前 =====
    print("\n" + "-"*70)
    print("【第1ラウンド: 攻撃的行動を受ける】")
    
    # 攻撃的シグナル（シグナル3が強い）
    aggressive_signal = np.array([0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0])
    
    # 解釈
    pressure_1 = module.interpret_signals(aggressive_signal, kappa)
    print(f"観測シグナル: {aggressive_signal}")
    print(f"解釈された圧力: {pressure_1}")
    print(f"BASE層への圧力: {pressure_1[1]:.3f}")
    
    # 悪い結果（処刑された）
    outcome_1 = -1.0
    print(f"結果: outcome = {outcome_1} (処刑)")
    
    # 経験を記録
    module.record_experience(
        signal_pattern=aggressive_signal,
        layer=1,  # BASE層
        interpreted_pressure=pressure_1[1],
        outcome=outcome_1,
        context={'event': 'execution', 'round': 1}
    )
    
    print(f"記憶数: {module.get_memory_count()}")
    
    # 時間経過とκの上昇（学習）
    module.advance_time(10.0)
    kappa[1] += 0.3  # BASE層で学習
    print(f"学習後のκ: {kappa}")
    
    # ===== 第2ラウンド: 学習後 =====
    print("\n" + "-"*70)
    print("【第2ラウンド: 同じ攻撃的行動を受ける】")
    
    # 同じシグナル
    pressure_2 = module.interpret_signals(aggressive_signal, kappa, update_matrix=True)
    
    # 学習後の係数
    learned_coeffs = module.get_current_coeffs()
    print(f"観測シグナル: {aggressive_signal}")
    print(f"学習後のBASE層シグナル3係数: {learned_coeffs[1, 3]:.3f} (初期: {base_coeffs[1, 3]:.3f})")
    print(f"解釈された圧力: {pressure_2}")
    print(f"BASE層への圧力: {pressure_2[1]:.3f} (第1ラウンド: {pressure_1[1]:.3f})")
    
    # 今度は成功（防衛できた）
    outcome_2 = +0.5
    print(f"結果: outcome = {outcome_2} (防衛成功)")
    
    module.record_experience(
        signal_pattern=aggressive_signal,
        layer=1,
        interpreted_pressure=pressure_2[1],
        outcome=outcome_2,
        context={'event': 'defended', 'round': 2}
    )
    
    # ===== 第3ラウンド: さらなる学習 =====
    print("\n" + "-"*70)
    print("【第3ラウンド: さらに経験を積む】")
    
    module.advance_time(10.0)
    kappa[1] += 0.2
    
    pressure_3 = module.interpret_signals(aggressive_signal, kappa, update_matrix=True)
    learned_coeffs = module.get_current_coeffs()
    
    print(f"κ: {kappa}")
    print(f"BASE層シグナル3係数: {learned_coeffs[1, 3]:.3f}")
    print(f"BASE層への圧力: {pressure_3[1]:.3f}")
    
    # ===== 結果の比較 =====
    print("\n" + "="*70)
    print("【学習の効果】")
    print("="*70)
    print(f"第1ラウンド (学習前): 係数={base_coeffs[1, 3]:.3f}, 圧力={pressure_1[1]:.3f} → 処刑")
    print(f"第2ラウンド (学習後): 係数={learned_coeffs[1, 3]:.3f}, 圧力={pressure_2[1]:.3f} → 防衛成功")
    
    improvement = (pressure_2[1] - pressure_1[1]) / pressure_1[1] * 100
    print(f"\n警戒レベルの向上: {improvement:.1f}%")
    print("\n✅ エージェントは経験から学び、同じシグナルに対する「見方」を変えた！")
    
    return module


def demo_signal_differentiation():
    """
    シグナル別学習のデモ
    
    異なるシグナルに対して、独立に学習できることを示す
    """
    print("\n\n" + "="*70)
    print("シグナル別学習のデモ")
    print("="*70)
    
    base_coeffs = np.array([
        [0.1, 0.1, 0.2, 0.3, 0.2, 0.1, 0.1],
        [0.2, 0.3, 0.5, 0.3, 0.4, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.5, 0.6, 0.3, 0.2],
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.4],
    ])
    
    module = DynamicInterpretationModule(base_coeffs=base_coeffs)
    kappa = np.array([0.5, 0.5, 0.5, 0.5])
    
    # シグナル3: 攻撃的行動 → 悪い結果
    signal_3 = np.array([0, 0, 0, 0.9, 0, 0, 0])
    pressure_3 = module.interpret_signals(signal_3, kappa, update_matrix=False)
    module.record_experience(signal_3, 1, pressure_3[1], outcome=-0.9)
    module.advance_time(5.0)
    
    # シグナル5: 協調的行動 → 良い結果
    signal_5 = np.array([0, 0, 0, 0, 0, 0.8, 0])
    pressure_5 = module.interpret_signals(signal_5, kappa, update_matrix=False)
    module.record_experience(signal_5, 2, pressure_5[2], outcome=+0.8)
    module.advance_time(5.0)
    
    # 学習後
    module.interpret_signals(signal_3, kappa, update_matrix=True)
    learned_coeffs = module.get_current_coeffs()
    
    print("\n【学習結果】")
    print(f"シグナル3（攻撃的）への係数:")
    print(f"  BASE層: {base_coeffs[1, 3]:.3f} → {learned_coeffs[1, 3]:.3f} (警戒↑)")
    print(f"\nシグナル5（協調的）への係数:")
    print(f"  CORE層: {base_coeffs[2, 5]:.3f} → {learned_coeffs[2, 5]:.3f} (信頼↓)")
    
    print("\n✅ 異なるシグナルを独立に学習！")


if __name__ == "__main__":
    # デモ実行
    module = demo_learning_cycle()
    demo_signal_differentiation()
    
    print("\n\n" + "="*70)
    print("動的解釈モジュール - テスト完了")
    print("="*70)
