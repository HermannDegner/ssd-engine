"""
主観的社会システム (Subjective Society System)
==============================================

Phase 4の理論的整合を完全に達成した、主観的社会ダイナミクス。

核心原理:
- 「神の視点」を完全廃止
- エージェントは他者の内部状態（E, κ）を直接観測できない
- 他者の外的シグナル（表情・行動・発言）を観測
- 自己の構造で主観的に解釈
- その結果、自己の内部状態が変化

v5/v6との決定的な違い:
- v5/v6 Society: 神の視点で全エージェントの E を直接操作
- v8 SubjectiveSociety: 各エージェントが主観的に観測→解釈→自己変化

理論的意義:
- SSDの「主観力学」の本質に完全整合
- 社会的層間変換の実現（他者のUPPER → 自己のBASE）
- 創発的な社会ダイナミクス
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Callable
from enum import Enum
import numpy as np

from ssd_engine.core.ssd_human_module import HumanAgent, HumanPressure, HumanLayer
from .ssd_subjective_social_pressure import (
    SubjectiveSocialPressureCalculator,
    ObservableSignal,
    ObservationContext
)


@dataclass
class AgentState:
    """エージェントの外的に観測可能な状態
    
    内部状態（E, κ）は直接観測できない。
    観測可能なのは、これらの外的表現のみ。
    """
    agent_id: str
    
    # 観測可能な外的シグナル
    visible_signals: Dict[ObservableSignal, float]  # {FEAR_EXPRESSION: 0.8, ...}
    
    # 最近の行動履歴（文脈情報）
    recent_actions: List[str]
    
    # 発言内容（イデオロギー的内容を含む）
    verbal_content: Optional[str] = None
    ideology_alignment_hint: float = 0.0  # -1.0 ~ 1.0


class SignalGenerator:
    """内部状態から観測可能なシグナルを生成
    
    エージェントの内部状態（E, κ）から、
    外的に観測可能なシグナル（表情・行動）を生成する。
    """
    
    def __init__(self):
        """初期化"""
        pass
    
    def generate_signals(self, agent: HumanAgent) -> Dict[ObservableSignal, float]:
        """エージェントの内部状態から観測可能なシグナルを生成
        
        Args:
            agent: HumanAgent
            
        Returns:
            観測可能なシグナル {FEAR_EXPRESSION: 0.8, ANGER_EXPRESSION: 0.3, ...}
        """
        signals = {}
        
        # エネルギーとκを取得
        E_physical = agent.state.E[HumanLayer.PHYSICAL.value]
        E_base = agent.state.E[HumanLayer.BASE.value]
        E_core = agent.state.E[HumanLayer.CORE.value]
        E_upper = agent.state.E[HumanLayer.UPPER.value]
        
        kappa_base = agent.state.kappa[HumanLayer.BASE.value]
        kappa_core = agent.state.kappa[HumanLayer.CORE.value]
        kappa_upper = agent.state.kappa[HumanLayer.UPPER.value]
        
        # 恐怖表情（BASE層の高エネルギー）
        if E_base > 0.3:  # 閾値を現実的に調整（0.3以上で表情に現れる）
            fear_intensity = min(E_base / 10.0, 1.0)
            signals[ObservableSignal.FEAR_EXPRESSION] = fear_intensity
        
        # 怒り表情（BASE層 + CORE層の高エネルギー）
        if E_base > 1.0 and E_core > 0.8:
            anger_intensity = min((E_base + E_core) / 15.0, 1.0)
            signals[ObservableSignal.ANGER_EXPRESSION] = anger_intensity
        
        # 協力的行動（CORE層のκが高い）
        if kappa_core > 1.5 and E_core < 3.0:  # 規範意識が高く、葛藤が少ない
            cooperative_intensity = min((kappa_core - 1.0) / 2.0, 1.0)
            signals[ObservableSignal.COOPERATIVE_ACT] = cooperative_intensity
        
        # 攻撃的行動（BASE層が高く、UPPER層の抑制が効いていない）
        if E_base > 5.0 and E_upper < 1.0:
            aggressive_intensity = min(E_base / 10.0, 1.0)
            signals[ObservableSignal.AGGRESSIVE_ACT] = aggressive_intensity
        
        # イデオロギー的発言（UPPER層が高い）
        if E_upper > 1.0:  # 閾値を現実的に調整
            ideology_intensity = min(E_upper / 8.0, 1.0)
            signals[ObservableSignal.VERBAL_IDEOLOGY] = ideology_intensity
        
        # 規範遵守（CORE層のκが高い）
        if kappa_core > 1.8:
            adherence_intensity = min((kappa_core - 1.0) / 3.0, 1.0)
            signals[ObservableSignal.NORM_ADHERENCE] = adherence_intensity
        
        # 規範違反（CORE層のエネルギーが高く、κが低い）
        if E_core > 3.0 and kappa_core < 1.2:
            violation_intensity = min(E_core / 8.0, 1.0)
            signals[ObservableSignal.NORM_VIOLATION] = violation_intensity
        
        return signals
    
    def get_ideology_alignment(
        self,
        agent: HumanAgent,
        reference_agent: HumanAgent
    ) -> float:
        """2つのエージェント間のイデオロギー一致度を推定
        
        Args:
            agent: 評価対象エージェント
            reference_agent: 基準エージェント
            
        Returns:
            -1.0 (完全対立) ~ +1.0 (完全一致)
        """
        # UPPER層のκの差異から推定
        kappa_diff = abs(
            agent.state.kappa[HumanLayer.UPPER.value] -
            reference_agent.state.kappa[HumanLayer.UPPER.value]
        )
        
        # UPPER層のEの相関から推定
        E_upper_agent = agent.state.E[HumanLayer.UPPER.value]
        E_upper_ref = reference_agent.state.E[HumanLayer.UPPER.value]
        
        # 簡易的な一致度計算
        if kappa_diff < 0.2:  # κが近い = 似た価値観
            if abs(E_upper_agent - E_upper_ref) < 10.0:
                return 0.8  # 高い一致
            else:
                return 0.4  # 中程度の一致
        elif kappa_diff > 0.8:  # κが大きく異なる
            return -0.7  # 対立
        else:
            return 0.0  # 中立


class RelationshipMatrix:
    """関係性マトリクス
    
    エージェント間の関係性（協力/競争）を管理。
    ただし、これは「客観的な関係性」ではなく、
    各エージェントの「主観的な認識」を集約したもの。
    """
    
    def __init__(self, num_agents: int):
        """初期化
        
        Args:
            num_agents: エージェント数
        """
        # matrix[i][j]: エージェントiのjに対する関係性認識
        self.matrix = np.zeros((num_agents, num_agents))
        
        # デフォルトは中立（0.0）
        for i in range(num_agents):
            self.matrix[i, i] = 0.0  # 自己との関係は中立
    
    def set(self, agent_i: int, agent_j: int, relationship: float):
        """関係性を設定
        
        Args:
            agent_i: エージェントiのインデックス
            agent_j: エージェントjのインデックス
            relationship: -1.0 (敵対) ~ +1.0 (親密)
        """
        self.matrix[agent_i, agent_j] = np.clip(relationship, -1.0, 1.0)
    
    def get(self, agent_i: int, agent_j: int) -> float:
        """関係性を取得
        
        Args:
            agent_i: エージェントiのインデックス
            agent_j: エージェントjのインデックス
            
        Returns:
            -1.0 (敵対) ~ +1.0 (親密)
        """
        return self.matrix[agent_i, agent_j]
    
    def update_from_interaction(
        self,
        agent_i: int,
        agent_j: int,
        interaction_quality: float,
        learning_rate: float = 0.1
    ):
        """相互作用の結果から関係性を更新
        
        Args:
            agent_i: エージェントiのインデックス
            agent_j: エージェントjのインデックス
            interaction_quality: -1.0 (悪化) ~ +1.0 (改善)
            learning_rate: 学習率
        """
        current = self.matrix[agent_i, agent_j]
        delta = learning_rate * interaction_quality
        self.matrix[agent_i, agent_j] = np.clip(current + delta, -1.0, 1.0)


class SubjectiveSociety:
    """主観的社会システム
    
    Phase 4の理論的整合を完全達成。
    - 神の視点を廃止
    - 各エージェントが主観的に観測→解釈→自己変化
    - 社会的層間変換の実現
    """
    
    def __init__(
        self,
        agents: List[HumanAgent],
        initial_relationships: Optional[np.ndarray] = None
    ):
        """初期化
        
        Args:
            agents: HumanAgentのリスト
            initial_relationships: 初期関係性マトリクス（オプション）
        """
        self.agents = agents
        self.num_agents = len(agents)
        
        # 関係性マトリクス
        self.relationships = RelationshipMatrix(self.num_agents)
        if initial_relationships is not None:
            self.relationships.matrix = initial_relationships
        
        # シグナル生成器
        self.signal_generator = SignalGenerator()
        
        # 主観的圧力計算器
        self.pressure_calculator = SubjectiveSocialPressureCalculator()
        
        # 距離マトリクス（物理的/心理的距離）
        self.distance_matrix = np.zeros((self.num_agents, self.num_agents))
        for i in range(self.num_agents):
            for j in range(self.num_agents):
                if i != j:
                    self.distance_matrix[i, j] = 0.0  # デフォルトは近距離
    
    def step(self, dt: float = 0.1):
        """主観的社会ダイナミクスの1ステップ
        
        Args:
            dt: 時間刻み
        """
        # フェーズ1: 全エージェントのシグナル生成
        agent_signals = self._generate_all_signals()
        
        # フェーズ2: 各エージェントが他者を観測→解釈→自己変化
        for i, observer in enumerate(self.agents):
            self._process_subjective_observations(i, observer, agent_signals, dt)
    
    def _generate_all_signals(self) -> List[Dict[ObservableSignal, float]]:
        """全エージェントの観測可能なシグナルを生成
        
        Returns:
            各エージェントのシグナル辞書のリスト
        """
        signals = []
        for agent in self.agents:
            agent_signal = self.signal_generator.generate_signals(agent)
            signals.append(agent_signal)
        return signals
    
    def _process_subjective_observations(
        self,
        observer_idx: int,
        observer: HumanAgent,
        all_signals: List[Dict[ObservableSignal, float]],
        dt: float
    ):
        """観測者が他者を主観的に観測し、内部状態を更新
        
        Args:
            observer_idx: 観測者のインデックス
            observer: 観測者エージェント
            all_signals: 全エージェントのシグナル
            dt: 時間刻み
        """
        # 全ての他者を観測
        total_pressure = {
            HumanLayer.PHYSICAL: 0.0,
            HumanLayer.BASE: 0.0,
            HumanLayer.CORE: 0.0,
            HumanLayer.UPPER: 0.0
        }
        
        for target_idx, target_signals in enumerate(all_signals):
            if target_idx == observer_idx:
                continue  # 自分自身は観測しない
            
            # 各シグナルを観測
            for signal_type, signal_intensity in target_signals.items():
                if signal_intensity > 0.01:  # 閾値以上のシグナルのみ
                    # 観測コンテキスト作成
                    observation = self._create_observation_context(
                        observer_idx,
                        target_idx,
                        signal_type,
                        signal_intensity
                    )
                    
                    # 主観的解釈
                    social_pressure = self.pressure_calculator.calculate_pressure(
                        observer, observation
                    )
                    
                    # 圧力を累積
                    for layer, pressure in social_pressure.items():
                        total_pressure[layer] += pressure
        
        # 累積された社会的圧力を適用
        human_pressure = HumanPressure(
            physical=total_pressure[HumanLayer.PHYSICAL],
            base=total_pressure[HumanLayer.BASE],
            core=total_pressure[HumanLayer.CORE],
            upper=total_pressure[HumanLayer.UPPER]
        )
        
        # エージェントの内部状態が主観的に変化
        observer.step(human_pressure, dt)
    
    def _create_observation_context(
        self,
        observer_idx: int,
        target_idx: int,
        signal_type: ObservableSignal,
        signal_intensity: float
    ) -> ObservationContext:
        """観測コンテキストを作成
        
        Args:
            observer_idx: 観測者インデックス
            target_idx: 対象インデックス
            signal_type: シグナルタイプ
            signal_intensity: シグナル強度
            
        Returns:
            ObservationContext
        """
        observer = self.agents[observer_idx]
        target = self.agents[target_idx]
        
        # 関係性と距離
        relationship = self.relationships.get(observer_idx, target_idx)
        distance = self.distance_matrix[observer_idx, target_idx]
        
        # イデオロギー一致度（VERBAL_IDEOLOGYの場合）
        context_data = None
        if signal_type == ObservableSignal.VERBAL_IDEOLOGY:
            alignment = self.signal_generator.get_ideology_alignment(
                target, observer
            )
            context_data = {'ideology_alignment': alignment}
        
        return ObservationContext(
            observer_id=observer.agent_id,
            target_id=target.agent_id,
            signal_type=signal_type,
            signal_intensity=signal_intensity,
            relationship=relationship,
            distance=distance,
            context_data=context_data
        )
    
    def set_relationship(self, agent_i: str, agent_j: str, relationship: float):
        """関係性を設定（エージェントID指定）
        
        Args:
            agent_i: エージェントiのID
            agent_j: エージェントjのID
            relationship: -1.0 (敵対) ~ +1.0 (親密)
        """
        idx_i = next((i for i, a in enumerate(self.agents) if a.agent_id == agent_i), None)
        idx_j = next((i for i, a in enumerate(self.agents) if a.agent_id == agent_j), None)
        
        if idx_i is not None and idx_j is not None:
            self.relationships.set(idx_i, idx_j, relationship)
    
    def get_dominant_layers_distribution(self) -> Dict[HumanLayer, int]:
        """各層が支配的なエージェントの数を取得
        
        Returns:
            {PHYSICAL: 2, BASE: 3, CORE: 1, UPPER: 4}
        """
        distribution = {layer: 0 for layer in HumanLayer}
        
        for agent in self.agents:
            dominant = agent.get_dominant_layer()
            distribution[dominant] += 1
        
        return distribution
    
    def visualize_state(self):
        """社会状態の可視化"""
        print("\n=== Subjective Society State ===")
        print(f"Time: {self.agents[0].state.t:.1f}")
        print(f"Agents: {self.num_agents}")
        
        print("\nDominant Layers:")
        dist = self.get_dominant_layers_distribution()
        for layer, count in dist.items():
            print(f"  {layer.name}: {count} agents")
        
        print("\nRelationship Matrix (sample):")
        print(self.relationships.matrix[:min(5, self.num_agents), :min(5, self.num_agents)])
        
        print("\nAgent States (sample):")
        for agent in self.agents[:min(3, self.num_agents)]:
            print(f"\n{agent}")


# ========================================
# シナリオヘルパー関数
# ========================================

def create_subjective_fear_contagion_scenario(num_agents: int = 5) -> SubjectiveSociety:
    """主観的恐怖伝染シナリオ
    
    全員が協力関係。1人が恐怖を持つと、他者がそれを観測し、
    主観的に解釈して、自分も恐怖を感じる。
    """
    agents = [HumanAgent(agent_id=f"Agent_{i}") for i in range(num_agents)]
    
    # 協力関係
    relationships = np.ones((num_agents, num_agents)) * 0.8
    np.fill_diagonal(relationships, 0.0)
    
    # Societyを先に作成
    society = SubjectiveSociety(agents, relationships)
    
    # Agent_0に恐怖を注入（複数ステップで蓄積）
    for _ in range(200):  # 200ステップに増やして十分なE_baseを確保
        agents[0].step(HumanPressure(base=150.0), dt=0.1)
    
    return society


def create_subjective_ideology_conflict_scenario(num_agents: int = 6) -> SubjectiveSociety:
    """主観的イデオロギー対立シナリオ
    
    2グループに分かれ、各グループ内は協力、グループ間は競争。
    各グループがイデオロギーを持ち、対立する。
    """
    agents = [HumanAgent(agent_id=f"Agent_{i}") for i in range(num_agents)]
    
    mid = num_agents // 2
    
    # グループAにイデオロギーA注入（複数ステップで蓄積）
    for i in range(mid):
        agents[i].state.kappa[HumanLayer.UPPER.value] = 2.0
        for _ in range(50):
            agents[i].step(HumanPressure(upper=50.0), dt=0.1)
    
    # グループBにイデオロギーB注入（複数ステップで蓄積）
    for i in range(mid, num_agents):
        agents[i].state.kappa[HumanLayer.UPPER.value] = 1.2
        for _ in range(50):
            agents[i].step(HumanPressure(upper=45.0), dt=0.1)
    
    # 関係性マトリクス
    relationships = np.zeros((num_agents, num_agents))
    for i in range(num_agents):
        for j in range(num_agents):
            if i == j:
                relationships[i, j] = 0.0
            elif (i < mid and j < mid) or (i >= mid and j >= mid):
                relationships[i, j] = 0.8  # グループ内協力
            else:
                relationships[i, j] = -0.7  # グループ間競争
    
    return SubjectiveSociety(agents, relationships)
    return SubjectiveSociety(agents, relationships)
