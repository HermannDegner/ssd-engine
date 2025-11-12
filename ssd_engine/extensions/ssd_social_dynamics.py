"""
SSD Social Dynamics - 社会的相互作用モジュール
===============================================

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
理論的位置づけ: ミクロ（個人）とマクロ（社会）のシームレスな統合
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

本モジュールは、SSD基本理論（構造主観力学：シンプル版）の直接的な応用であり、
個人レベルのE/κダイナミクスを社会レベルへ拡張したものである。

【理論的基盤】

1. **シンプル版**: E/κダイナミクスの基本原理
   - E（未処理圧）の蓄積と放出
   - κ（整合慣性）の形成と安定化
   - 跳躍（Leap）による状態遷移

2. **人間モジュール コア**: 四層構造と個体の内部力学
   - 個体維持 vs 社会維持の二重性
   - 四層（PHYSICAL, BASE, CORE, UPPER）の相互作用

3. **本モジュール**: 個体間相互作用の力学
   - 個人のE/κが他者に伝播する
   - 関係性（協力/競争）によって伝播の方向が変わる
   - ミクロ（個人心理）→マクロ（集団現象）がシームレスに創発

【ミクロ-マクロ統合の仕組み】

```
個人A: E, κ（個人の内部状態）
   ↓ ζ（エネルギー伝播）、ξ（κ伝播）
個人B: E, κ（影響を受ける）
   ↓
集団: 全体のE/κ分布から集団現象が創発
   → 革命、パニック、規範形成、文化伝播
```

同一の理論的原理（E/κダイナミクス）で、
個人の心理も、社会の歴史も、統一的に記述できる。

【統計力学的性質と臨界現象】

**平時（通常状態）**: 集合的平均化が支配
- 多数のエージェントの相互作用により、小さな揺らぎは誤差として吸収される
- 個人の怒り、不満 → 社会全体では統計的にキャンセルされる
- 慣性の法則（社会の安定性）

**危機時（臨界状態）**: バタフライエフェクトが支配
- 系全体のEが閾値近傍（E ≈ E_threshold）の時
- 1人の小さな跳躍 → 連鎖的伝播 → 全体の革命
- 「最後の一滴」が歴史を動かす（相転移）

これは物理の相転移（水→氷）と同じ力学:
- 0.01℃の差は通常無視できる（平均化）
- しかし0℃付近では0.01℃が固体/液体を分ける（臨界）

社会も同様に、臨界点では個の揺らぎが全体を支配する。

【技術革新と臨界点の変化】

スマホ/SNSの登場は、社会の伝播係数ζを劇的に変化させた:

```
前スマホ時代（20世紀）:
  ζ_base ≈ 0.01  # 感情伝播は遅い（対面、電話のみ）
  → 小さな揺らぎは地域で収束
  → 革命には物理的な集会が必要

スマホ時代（21世紀）:
  ζ_base ≈ 0.3〜0.5  # 感情が瞬時に世界中へ伝播
  → 1つの動画、1つのツイートが世界を動かす
  → アラブの春、#MeToo、炎上現象
```

たった1つの道具（スマホ）が、社会の臨界点を劇的に下げた。
同じ不満・怒りでも、伝播速度が100倍になれば、
かつては消えていた揺らぎが、今は革命を起こす。

これは「人間が変わった」のではなく、
「パラメータζが変わった」だけである（SSD的解釈）。

【主要機能】

1. エネルギー伝播（zeta係数）: 感情・恐怖・興奮の伝染
2. κ伝播（xi係数）: 規範・技術・価値観の学習
3. 競合による増幅/抑制（omega係数）: 協力関係では共感、競争関係では対立
4. 関係性マトリクス: 協力/中立/競争の構造

【原典理論】

- 構造主観力学：シンプル版
  https://github.com/HermannDegner/Structural-Subjectivity-Dynamics/blob/main/構造主観力学：シンプル版.md

- 人間モジュール　コア
  https://github.com/HermannDegner/Structural-Subjectivity-Dynamics/blob/main/Human_Module/人間モジュール　コア.md
  「基層構造の内部における二重性: 個体維持 vs 社会維持」

- 整合の数理モデル
  https://github.com/HermannDegner/Structural-Subjectivity-Dynamics/blob/main/数理モデル/整合の数理モデル.md

【実装における力学式】

個人iのエネルギー変化（社会的影響）:
```
ΔE_i = Σⱼ ζ * relationship(i,j) * (E_j - E_i)
```

競争関係の場合（relationship < 0）:
```
ΔE_i *= (1 + ω * |relationship|)  # 競争による増幅
```

これにより、協力関係では共感・同期が、
競争関係では対立・増幅が、同じ式から創発する。
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum

import sys
import os
# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# coreモジュールのパスを追加
core_path = os.path.join(project_root, 'core')
if core_path not in sys.path:
    sys.path.insert(0, core_path)

from ssd_engine.core.ssd_human_module import HumanAgent, HumanParams, HumanPressure, HumanLayer


class RelationType(Enum):
    """関係性タイプ"""
    COOPERATION = 1    # 協力関係
    NEUTRAL = 0        # 中立
    COMPETITION = -1   # 競争関係


@dataclass
class SocialCouplingParams:
    """
    社会的カップリングパラメータ
    
    【理論的意義】
    個人間のE/κ伝播を制御するパラメータ群。
    
    人間モジュール コアの「社会維持の原理」に基づき、
    各層で異なる伝播特性を持つ:
    
    - PHYSICAL層: 疲労の伝染は弱い（ζ=0.02）
    - BASE層: 感情・恐怖の伝染は強い（ζ=0.08）
    - CORE層: 規範・価値観の伝播は中程度（ζ=0.05）
    - UPPER層: 抽象的理念の伝播は弱い（ζ=0.03）
    
    【協力 vs 競争】
    - 協力関係: E/κが共感的に伝播（同期）
    - 競争関係: Eが増幅され対立が激化（omega < 0による増幅）
    """
    # エネルギー伝播係数（zeta）
    zeta_physical: float = 0.02   # 物理層（疲労の伝染は弱い）
    zeta_base: float = 0.08       # 基層（感情の伝染は強い）
    zeta_core: float = 0.05       # 中核層（規範の伝播は中程度）
    zeta_upper: float = 0.03      # 上層（理念の伝播は弱い）
    
    # κ伝播係数（xi）
    xi_physical: float = 0.01     # 物理層（身体技術の伝達は遅い）
    xi_base: float = 0.04         # 基層（本能的学習は中速）
    xi_core: float = 0.06         # 中核層（規範学習は速い）
    xi_upper: float = 0.05        # 上層（理念学習は速い）
    
    # 競合増幅係数（omega、負値）
    # 競争関係でEを増幅させる（対立の激化）
    omega_physical: float = -0.01  # 物理層（競合は弱い）
    omega_base: float = -0.06      # 基層（本能的競合は強い）
    omega_core: float = -0.03      # 中核層（規範的競合は中程度）
    omega_upper: float = -0.02     # 上層（理念的競合は弱い）
    
    # 協力閾値
    cooperation_threshold: float = 0.5   # relation > 0.5で協力
    competition_threshold: float = -0.5  # relation < -0.5で競争
    
    @classmethod
    def create_pre_digital_era(cls):
        """
        前デジタル時代（20世紀以前）のパラメータ
        
        伝播は遅く、地域的。革命には物理的集会が必須。
        """
        return cls(
            zeta_base=0.01,    # 感情伝播は非常に遅い（対面のみ）
            zeta_core=0.02,    # 規範も地域的
            xi_core=0.02,      # 文化伝播は遅い
        )
    
    @classmethod
    def create_digital_era(cls):
        """
        デジタル時代（21世紀、スマホ/SNS普及後）のパラメータ
        
        伝播は瞬時、グローバル。1つのツイートが世界を動かす。
        """
        return cls(
            zeta_base=0.3,     # 感情が瞬時に伝播（SNS、動画）
            zeta_core=0.15,    # 価値観も急速に拡散
            xi_core=0.12,      # 文化がグローバルに同期
            omega_base=-0.15,  # 炎上、極性化が激化
        )
    
    @classmethod
    def create_hyper_connected_future(cls):
        """
        超接続時代（未来予測、脳直結SNSなど）のパラメータ
        
        思考・感情が直接共有される世界。
        個の境界が曖昧になる可能性。
        """
        return cls(
            zeta_base=0.7,     # 感情がほぼ直接共有
            zeta_core=0.5,     # 価値観も即座に同期
            xi_base=0.3,       # 本能レベルで学習
            xi_core=0.4,       # 規範が瞬時に伝播
        )


@dataclass
class RelationshipMatrix:
    """
    関係性マトリクス
    
    【理論的意義】
    個体間の関係性構造を表現する行列。
    
    SSD基本理論では、個人は独立したE/κシステムだが、
    社会的存在として他者と結びついている（人間モジュール コア：社会維持）。
    
    この行列が、ミクロ（個人）とマクロ（社会）を繋ぐ架け橋となる。
    
    【値の意味】
    - agents[i]のagents[j]に対する関係性
    - 値域: [-1.0, 1.0]
      * +1.0: 完全な協力（E/κが共感的に伝播、共同体）
      *  0.0: 中立（伝播は弱い、無関心）
      * -1.0: 完全な競争（Eが増幅、敵対関係）
    
    【創発する社会現象】
    - 協力的ネットワーク → 規範の共有、文化の形成
    - 競争的ネットワーク → 階級闘争、革命、分極化
    - 混合ネットワーク → 派閥の形成、社会の階層化
    """
    matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    
    def __post_init__(self):
        """対角成分を0にする（自己との関係は無し）"""
        np.fill_diagonal(self.matrix, 0.0)
    
    @classmethod
    def create_random(cls, num_agents: int, cooperation_bias: float = 0.0):
        """
        ランダム関係性マトリクスの生成
        
        Args:
            num_agents: エージェント数
            cooperation_bias: 協力バイアス（-1.0〜1.0）
        """
        matrix = np.random.uniform(-1.0, 1.0, (num_agents, num_agents))
        matrix += cooperation_bias
        matrix = np.clip(matrix, -1.0, 1.0)
        np.fill_diagonal(matrix, 0.0)
        return cls(matrix=matrix)
    
    @classmethod
    def create_cooperative(cls, num_agents: int):
        """完全協力関係"""
        matrix = np.ones((num_agents, num_agents))
        np.fill_diagonal(matrix, 0.0)
        return cls(matrix=matrix)
    
    @classmethod
    def create_competitive(cls, num_agents: int):
        """完全競争関係"""
        matrix = -np.ones((num_agents, num_agents))
        np.fill_diagonal(matrix, 0.0)
        return cls(matrix=matrix)
    
    def get_relation(self, i: int, j: int) -> float:
        """i→jの関係性を取得"""
        return self.matrix[i, j]
    
    def set_relation(self, i: int, j: int, value: float):
        """i→jの関係性を設定"""
        self.matrix[i, j] = np.clip(value, -1.0, 1.0)


class Society:
    """
    社会システム（多エージェント）
    
    【理論的意義】
    SSD理論におけるミクロ-マクロ統合の実装。
    
    複数の個人（HumanAgent）が、関係性マトリクスを介して
    E/κを相互に伝播させることで、集団レベルの現象が創発する。
    
    【統計力学的性質】
    
    多数のエージェントの相互作用により、個別の揺らぎは
    **集合的平均化**によって吸収される（統計力学の法則）。
    
    - 小さな揺らぎ → 誤差として平均化され消える
    - 安定状態は統計的に保たれる（社会の慣性）
    
【バタフライエフェクト（臨界現象）】

    しかし、系が臨界状態（E ≈ E_threshold）にある時、
    **小さな揺らぎが巨大な変化を引き起こす**場合がある:
    
    1. 相転移の直前
       - 革命前夜: 平民全体のEが閾値近傍
       - 1人の跳躍（暴動）→ 連鎖的伝播 → 全体の革命
       
    2. ネットワーク構造の影響
       - ハブ的存在（影響力の強い個人）の揺らぎ
       - 関係性が密な場合、伝播が加速
       
    3. 共鳴現象
       - 複数の層で同時にE ≈ E_thresholdの場合
       - BASE層（恐怖）+ CORE層（不正義感）の共鳴
       - 跳躍が多層で同期 → 社会変革
    
    4. 技術革新による構造変化（現代的例）
       - スマホ/SNSの登場 → ζ（伝播係数）が劇的に増加
       - かつて: 噂の伝播は遅い（対面のみ、ζ_base ≈ 0.01）
       - 現在: 感情が瞬時に拡散（SNS、ζ_base ≈ 0.5）
       - 結果: 炎上、集団極性化、フェイクニュースの爆発的伝播
       - 「たった1つの道具（スマホ）」が社会の臨界点を下げた    【通常状態 vs 臨界状態】
    
    - **通常時**: 平均化が支配的
      * 小さな不満、個人的怒り → 社会全体では無視できる
      * 統計的安定性（慣性の法則）
      
    - **臨界時**: バタフライエフェクトが支配的
      * 「最後の一滴」が革命を引き起こす
      * 歴史の転換点（非線形ダイナミクス）
    
    【創発する社会現象の例】
    
    1. 恐怖伝染（フランス革命のテロル）
       - 1人の高いE（恐怖） → BASE層のζで伝播
       - 協力的ネットワークで全体に拡散
       - 集団パニック、魔女狩り
    
    2. 規範の形成（文化の創発）
       - あるκパターンの共有 → CORE層のξで学習
       - 協力関係で規範が定着
       - 共同体の価値観、タブー
    
    3. 階級闘争（競争的カップリング）
       - 貴族 vs 平民（競争関係）
       - 平民の不満E → ωで増幅
       - 革命の跳躍（臨界現象）
    
    4. 派閥の形成
       - 内部は協力、外部は競争
       - κの同期（内集団）+ Eの増幅（外集団）
       - 社会の分極化
    
    同じE/κダイナミクスで、これらすべてが自然に創発する。
    外部から「パニックを起こせ」と指示する必要はない。
    
    平時は統計的安定、臨界時は個の揺らぎが歴史を動かす。
    """
    
    def __init__(
        self,
        num_agents: int = 10,
        human_params: Optional[HumanParams] = None,
        social_params: Optional[SocialCouplingParams] = None,
        relationships: Optional[RelationshipMatrix] = None
    ):
        self.num_agents = num_agents
        self.social_params = social_params or SocialCouplingParams()
        
        # エージェント生成
        self.agents: List[HumanAgent] = []
        for i in range(num_agents):
            agent = HumanAgent(
                params=human_params or HumanParams(),
                agent_id=f"Agent_{i}"
            )
            self.agents.append(agent)
        
        # 関係性マトリクス
        if relationships is None:
            self.relationships = RelationshipMatrix.create_random(num_agents)
        else:
            self.relationships = relationships
        
        # 時間
        self.t = 0.0
    
    def _compute_social_coupling_for_agent(
        self,
        agent_idx: int
    ) -> Dict[str, np.ndarray]:
        """
        特定エージェントへの社会的カップリングを計算
        
        【理論的説明】
        個人iが他者jから受けるE/κの影響を計算。
        
        エネルギー伝播:
            ΔE_i = Σⱼ ζ * relation(i,j) * (E_j - E_i)
        
        競争時の増幅:
            ΔE_i *= (1 + ω * |relation|)  if relation < 0
        
        Returns:
            {
                'energy_coupling': 各層へのエネルギー影響（4次元）,
                'kappa_coupling': 各層へのκ影響（4次元）
            }
        """
        agent = self.agents[agent_idx]
        
        # 各層へのカップリング
        energy_coupling = np.zeros(4)
        kappa_coupling = np.zeros(4)
        
        # 全ての他エージェントからの影響を集計
        for other_idx in range(self.num_agents):
            if other_idx == agent_idx:
                continue
            
            other_agent = self.agents[other_idx]
            relation = self.relationships.get_relation(agent_idx, other_idx)
            
            # 関係性タイプ判定
            if relation > self.social_params.cooperation_threshold:
                relation_type = RelationType.COOPERATION
                relation_factor = relation  # 正の係数
            elif relation < self.social_params.competition_threshold:
                relation_type = RelationType.COMPETITION
                relation_factor = abs(relation)  # 負の関係を正に変換
            else:
                relation_type = RelationType.NEUTRAL
                relation_factor = 0.0
            
            if relation_type == RelationType.NEUTRAL:
                continue
            
            # 各層のカップリング計算
            for layer_idx in range(4):
                # エネルギー伝播
                E_self = agent.state.E[layer_idx]
                E_other = other_agent.state.E[layer_idx]
                
                # κ伝播
                kappa_self = agent.state.kappa[layer_idx]
                kappa_other = other_agent.state.kappa[layer_idx]
                
                # 層別係数
                if layer_idx == HumanLayer.PHYSICAL.value:
                    zeta = self.social_params.zeta_physical
                    xi = self.social_params.xi_physical
                    omega = self.social_params.omega_physical
                elif layer_idx == HumanLayer.BASE.value:
                    zeta = self.social_params.zeta_base
                    xi = self.social_params.xi_base
                    omega = self.social_params.omega_base
                elif layer_idx == HumanLayer.CORE.value:
                    zeta = self.social_params.zeta_core
                    xi = self.social_params.xi_core
                    omega = self.social_params.omega_core
                else:  # UPPER
                    zeta = self.social_params.zeta_upper
                    xi = self.social_params.xi_upper
                    omega = self.social_params.omega_upper
                
                # 協力関係の場合
                if relation_type == RelationType.COOPERATION:
                    # エネルギー伝播（差分に比例）
                    delta_E = (E_other - E_self) * zeta * relation_factor
                    energy_coupling[layer_idx] += delta_E
                    
                    # κ伝播（高い方が低い方を引き上げる）
                    if kappa_other > kappa_self:
                        delta_kappa = (kappa_other - kappa_self) * xi * relation_factor
                        kappa_coupling[layer_idx] += delta_kappa
                
                # 競争関係の場合
                elif relation_type == RelationType.COMPETITION:
                    # 競合抑制（相手のエネルギーが自分を抑制）
                    suppression = omega * E_other * relation_factor
                    energy_coupling[layer_idx] += suppression
        
        return {
            'energy_coupling': energy_coupling,
            'kappa_coupling': kappa_coupling
        }
    
    def step(self, pressures: Optional[List[HumanPressure]] = None, dt: float = 0.1):
        """
        社会全体の1ステップ更新
        
        Args:
            pressures: 各エージェントへの外部圧力（Noneの場合はゼロ圧力）
            dt: 時間刻み
        """
        if pressures is None:
            pressures = [HumanPressure() for _ in range(self.num_agents)]
        
        # 各エージェントの社会的カップリングを計算
        social_couplings = []
        for i in range(self.num_agents):
            coupling = self._compute_social_coupling_for_agent(i)
            social_couplings.append(coupling)
        
        # 各エージェントを更新（社会的カップリングを反映）
        for i, agent in enumerate(self.agents):
            # 基本ステップ
            agent.step(pressures[i], dt=dt)
            
            # 社会的カップリングを状態に加算
            coupling = social_couplings[i]
            agent.state.E += coupling['energy_coupling'] * dt
            agent.state.kappa += coupling['kappa_coupling'] * dt
            
            # κの範囲制約
            kappa_min = np.array([
                agent.params.kappa_min_physical,
                agent.params.kappa_min_base,
                agent.params.kappa_min_core,
                agent.params.kappa_min_upper
            ])
            agent.state.kappa = np.maximum(kappa_min, agent.state.kappa)
        
        self.t += dt
    
    def get_social_network_state(self) -> Dict:
        """
        社会ネットワークの状態を取得
        
        Returns:
            エージェント状態と関係性の辞書
        """
        agent_states = []
        for agent in self.agents:
            agent_states.append(agent.get_psychological_state())
        
        return {
            "time": self.t,
            "num_agents": self.num_agents,
            "agents": agent_states,
            "relationships": self.relationships.matrix.tolist()
        }
    
    def get_dominant_layers_distribution(self) -> Dict[str, int]:
        """
        支配層の分布を取得
        
        Returns:
            各層が支配的なエージェントの数
        """
        distribution = {
            "PHYSICAL": 0,
            "BASE": 0,
            "CORE": 0,
            "UPPER": 0
        }
        
        for agent in self.agents:
            dominant = agent.get_dominant_layer()
            distribution[dominant.name] += 1
        
        return distribution
    
    def is_critical_state(self, threshold_ratio: float = 0.8) -> Dict[str, bool]:
        """
        社会が臨界状態にあるかを判定
        
        【理論的意義】
        バタフライエフェクトが発生しうる臨界状態を検出。
        多数のエージェントのEが閾値近傍にある場合、
        小さな揺らぎが社会全体を変化させる可能性がある。
        
        Args:
            threshold_ratio: 閾値に対する比率（0.8 = 80%）
        
        Returns:
            各層ごとの臨界判定 {"BASE": True, "CORE": False, ...}
        """
        critical = {}
        
        for layer_idx, layer_name in enumerate(["PHYSICAL", "BASE", "CORE", "UPPER"]):
            near_threshold_count = 0
            
            for agent in self.agents:
                E = agent.state.E[layer_idx]
                E_threshold = agent.params.E_threshold[layer_idx]
                
                # Eが閾値の80%以上の場合、臨界とみなす
                if E >= threshold_ratio * E_threshold:
                    near_threshold_count += 1
            
            # 50%以上のエージェントが臨界ならTrue
            critical[layer_name] = (near_threshold_count / self.num_agents) >= 0.5
        
        return critical
    
    def get_average_E(self) -> np.ndarray:
        """
        全エージェントの平均E（各層）を取得
        
        【統計力学的意義】
        個別の揺らぎが平均化された後の「マクロ状態」。
        通常時はこの平均値が社会の状態を代表する。
        
        Returns:
            平均E [4] (PHYSICAL, BASE, CORE, UPPER)
        """
        total_E = np.zeros(4)
        for agent in self.agents:
            total_E += agent.state.E
        return total_E / self.num_agents
    
    def get_E_variance(self) -> np.ndarray:
        """
        全エージェントのE分散（各層）を取得
        
        【臨界検出の指標】
        分散が大きい → エージェント間のE差が大きい → 不安定
        分散が小さい → エージェント間で同期 → 安定 or 集団行動
        
        Returns:
            分散 [4]
        """
        avg_E = self.get_average_E()
        variance = np.zeros(4)
        
        for agent in self.agents:
            variance += (agent.state.E - avg_E) ** 2
        
        return variance / self.num_agents
    
    def visualize_network(self):
        """
        社会ネットワークの可視化（簡易版）
        
        Note: 本格的な可視化にはnetworkx/matplotlibが必要
        """
        print("\n=== Social Network State ===")
        print(f"Time: {self.t:.1f}")
        print(f"Agents: {self.num_agents}")
        
        # 臨界状態チェック
        critical = self.is_critical_state()
        print("\n⚠️  Critical State Detection:")
        for layer, is_critical in critical.items():
            status = "🔴 CRITICAL" if is_critical else "🟢 Stable"
            print(f"  {layer}: {status}")
        
        # 統計量
        avg_E = self.get_average_E()
        var_E = self.get_E_variance()
        print("\n📊 Statistical Measures:")
        for i, layer in enumerate(["PHYSICAL", "BASE", "CORE", "UPPER"]):
            print(f"  {layer}: E_avg={avg_E[i]:.2f}, E_var={var_E[i]:.2f}")
        
        print("\nDominant Layers:")
        dist = self.get_dominant_layers_distribution()
        for layer, count in dist.items():
            print(f"  {layer}: {count} agents")
        
        print("\nRelationship Matrix (sample):")
        print(self.relationships.matrix[:5, :5])  # 最初の5x5のみ表示
        
        print("\nAgent States (sample):")
        for i in range(min(3, self.num_agents)):
            print(f"\n{self.agents[i]}")


# ============================================================================
# シナリオヘルパー
# ============================================================================

def create_fear_contagion_scenario(num_agents: int = 5) -> Society:
    """
    恐怖伝染シナリオ
    
    1人のエージェントが高いE_baseを持ち、他に伝播する
    
    使用例:
        # 前デジタル時代（伝播が遅い）
        society = create_fear_contagion_scenario(num_agents=100)
        society.social_params = SocialCouplingParams.create_pre_digital_era()
        
        # デジタル時代（伝播が速い）
        society = create_fear_contagion_scenario(num_agents=100)
        society.social_params = SocialCouplingParams.create_digital_era()
        # → 同じ初期条件でも、パラメータζの違いで結果が激変
    """
    society = Society(
        num_agents=num_agents,
        relationships=RelationshipMatrix.create_cooperative(num_agents)
    )
    
    # Agent_0に高い恐怖（E_base）を設定
    society.agents[0].state.E[HumanLayer.BASE.value] = 150.0
    
    return society


def create_sns_flame_war_scenario(num_agents: int = 50) -> Society:
    """
    SNS炎上シナリオ（デジタル時代専用）
    
    スマホ/SNSによる超高速伝播を再現。
    1つの不適切発言が、数時間で全体を巻き込む炎上へ。
    
    使用例:
        society = create_sns_flame_war_scenario()
        # 1人の不適切発言（Agent_0のE_base上昇）
        # → 数ステップで全員が怒り状態に
    """
    society = Society(
        num_agents=num_agents,
        social_params=SocialCouplingParams.create_digital_era(),  # 高ζ
        relationships=RelationshipMatrix.create_cooperative(num_agents, cooperation_bias=0.3)
    )
    
    # Agent_0が炎上の火種（不適切発言 → 周囲の怒り）
    society.agents[0].state.E[HumanLayer.BASE.value] = 120.0
    society.agents[0].state.E[HumanLayer.CORE.value] = 80.0  # 道徳的怒り
    
    return society


def create_ideology_conflict_scenario(num_agents: int = 6) -> Society:
    """
    イデオロギー対立シナリオ
    
    グループA（3人）とグループB（3人）が競争関係
    """
    society = Society(num_agents=num_agents)
    
    # グループ内は協力、グループ間は競争
    for i in range(3):
        for j in range(3):
            # グループA内部
            society.relationships.set_relation(i, j, 0.8)
            # グループB内部
            society.relationships.set_relation(i+3, j+3, 0.8)
            # A→B競争
            society.relationships.set_relation(i, j+3, -0.9)
            # B→A競争
            society.relationships.set_relation(i+3, j, -0.9)
    
    # 各グループに異なる理念エネルギー
    for i in range(3):
        society.agents[i].state.E[HumanLayer.UPPER.value] = 50.0
        society.agents[i+3].state.E[HumanLayer.UPPER.value] = 45.0
    
    return society


def create_norm_propagation_scenario(num_agents: int = 7) -> Society:
    """
    規範伝播シナリオ
    
    1人の「模範エージェント」が高いκ_coreを持ち、他に伝播
    """
    society = Society(
        num_agents=num_agents,
        relationships=RelationshipMatrix.create_cooperative(num_agents)
    )
    
    # Agent_0を模範として設定（高いκ_core）
    society.agents[0].state.kappa[HumanLayer.CORE.value] = 1.8
    
    return society


# 簡易ラッパークラス（フランス革命シミュレーター用）
class SocialDynamicsEngine:
    """Societyクラスのラッパー（カスタムエージェントリスト対応）"""
    
    def __init__(self, agents: List[HumanAgent], relationships: 'RelationshipMatrix', 
                 params: Optional[SocialCouplingParams] = None):
        self.agents = agents
        self.relationships = relationships
        self.social_params = params or SocialCouplingParams()
        self.num_agents = len(agents)
        self.t = 0.0
        
        # Societyの内部メソッドを使用するため、一時的なSocietyインスタンスを作成
        self._society_template = Society(num_agents=self.num_agents, social_params=self.social_params)
    
    def step(self):
        """1ステップの社会的相互作用"""
        # 各エージェントへの社会的カップリングを計算
        for agent_idx in range(self.num_agents):
            coupling = self._compute_social_coupling_for_agent(agent_idx)
            
            # エネルギーカップリングを適用
            for layer in range(4):
                self.agents[agent_idx].state.E[layer] += coupling['energy_coupling'][layer]
            
            # κカップリングを適用
            for layer in range(4):
                self.agents[agent_idx].state.kappa[layer] += coupling['kappa_coupling'][layer]
                # κは最小値を下回らない
                self.agents[agent_idx].state.kappa[layer] = max(
                    self.agents[agent_idx].state.kappa[layer],
                    self.agents[agent_idx].params.to_core_params().kappa_min_values[layer]
                )
        
        self.t += 1.0
    
    def _compute_social_coupling_for_agent(self, agent_idx: int) -> Dict[str, np.ndarray]:
        """特定エージェントへの社会的カップリングを計算"""
        agent = self.agents[agent_idx]
        
        energy_coupling = np.zeros(4)
        kappa_coupling = np.zeros(4)
        
        for other_idx in range(self.num_agents):
            if other_idx == agent_idx:
                continue
            
            other_agent = self.agents[other_idx]
            relation = self.relationships.matrix[agent_idx, other_idx]
            
            # 協力関係
            if relation > self.social_params.cooperation_threshold:
                # エネルギー伝播
                zetas = [
                    self.social_params.zeta_physical,
                    self.social_params.zeta_base,
                    self.social_params.zeta_core,
                    self.social_params.zeta_upper
                ]
                for layer in range(4):
                    energy_coupling[layer] += zetas[layer] * other_agent.state.E[layer] * relation
                
                # κ伝播
                xis = [
                    self.social_params.xi_physical,
                    self.social_params.xi_base,
                    self.social_params.xi_core,
                    self.social_params.xi_upper
                ]
                for layer in range(4):
                    kappa_coupling[layer] += xis[layer] * (other_agent.state.kappa[layer] - agent.state.kappa[layer]) * relation
            
            # 競争関係
            elif relation < self.social_params.competition_threshold:
                omegas = [
                    self.social_params.omega_physical,
                    self.social_params.omega_base,
                    self.social_params.omega_core,
                    self.social_params.omega_upper
                ]
                for layer in range(4):
                    energy_coupling[layer] += omegas[layer] * other_agent.state.E[layer] * abs(relation)
        
        return {'energy_coupling': energy_coupling, 'kappa_coupling': kappa_coupling}
