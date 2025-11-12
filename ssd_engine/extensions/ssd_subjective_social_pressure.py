"""
主観的社会圧力システム (Subjective Social Pressure)
====================================================

Phase 4の理論的整合を達成するための、主観的な社会的意味圧実装。

核心原理:
- エージェントは他者の内部状態（E, κ）を直接観測できない
- 他者の「行動」や「表情」を「観測」し、「意味圧」として解釈する
- その結果、自己の内部構造（E, κ）が変化する

v5/v6との違い:
- v5: Society が全エージェントの E を直接操作（神の視点）
- v6: エージェントが他者を「観測」→「解釈」→「自己変化」（主観視点）
"""

from dataclasses import dataclass
from typing import Dict, List, Callable, Optional
from enum import Enum
import numpy as np

from ssd_engine.core.ssd_human_module import HumanLayer, HumanAgent


class ObservableSignal(Enum):
    """観測可能なシグナル（他者の外的表現）"""
    FEAR_EXPRESSION = "fear_expression"      # 恐怖の表情
    ANGER_EXPRESSION = "anger_expression"    # 怒りの表情
    COOPERATIVE_ACT = "cooperative_act"      # 協力的行動
    AGGRESSIVE_ACT = "aggressive_act"        # 攻撃的行動
    VERBAL_IDEOLOGY = "verbal_ideology"      # 言語化されたイデオロギー
    NORM_VIOLATION = "norm_violation"        # 規範違反
    NORM_ADHERENCE = "norm_adherence"        # 規範遵守


@dataclass
class ObservationContext:
    """観測コンテキスト
    
    エージェントが他者を観測する際の文脈情報
    """
    observer_id: str
    target_id: str
    signal_type: ObservableSignal
    signal_intensity: float  # 0.0-1.0: シグナルの強度
    relationship: float      # -1.0 (敵対) ~ +1.0 (親密)
    distance: float         # 0.0 (接触) ~ 1.0 (遠方)
    
    # 追加コンテキスト
    context_data: Optional[Dict] = None


class SubjectiveSocialPressureCalculator:
    """主観的社会圧力計算器
    
    他者の観測可能なシグナルから、自己の意味圧を計算する
    """
    
    def __init__(self):
        """初期化"""
        self.signal_interpreters: Dict[ObservableSignal, Callable] = {
            ObservableSignal.FEAR_EXPRESSION: self._interpret_fear_expression,
            ObservableSignal.ANGER_EXPRESSION: self._interpret_anger_expression,
            ObservableSignal.COOPERATIVE_ACT: self._interpret_cooperative_act,
            ObservableSignal.AGGRESSIVE_ACT: self._interpret_aggressive_act,
            ObservableSignal.VERBAL_IDEOLOGY: self._interpret_verbal_ideology,
            ObservableSignal.NORM_VIOLATION: self._interpret_norm_violation,
            ObservableSignal.NORM_ADHERENCE: self._interpret_norm_adherence,
        }
    
    def calculate_pressure(
        self,
        observer: HumanAgent,
        observation: ObservationContext
    ) -> Dict[HumanLayer, float]:
        """観測から意味圧を計算
        
        Args:
            observer: 観測者エージェント
            observation: 観測コンテキスト
            
        Returns:
            各層の意味圧 {PHYSICAL: 0.0, BASE: 0.5, CORE: 0.3, UPPER: 0.1}
        """
        # シグナルに対応する解釈関数を取得
        interpreter = self.signal_interpreters.get(
            observation.signal_type,
            self._interpret_default
        )
        
        # 解釈実行（主観的プロセス）
        layer_pressures = interpreter(observer, observation)
        
        # 距離と関係性で減衰
        attenuation = self._compute_attenuation(observation)
        
        return {
            layer: pressure * attenuation
            for layer, pressure in layer_pressures.items()
        }
    
    def _compute_attenuation(self, obs: ObservationContext) -> float:
        """距離と関係性による減衰係数
        
        - 距離が遠いほど減衰
        - 関係性が強い（絶対値が大きい）ほど影響大
        """
        distance_factor = 1.0 - obs.distance * 0.5  # 遠方でも50%は残る
        relationship_factor = 0.5 + abs(obs.relationship) * 0.5
        
        return distance_factor * relationship_factor
    
    # ========================================
    # シグナル解釈関数群（主観的意味付け）
    # ========================================
    
    def _interpret_fear_expression(
        self,
        observer: HumanAgent,
        obs: ObservationContext
    ) -> Dict[HumanLayer, float]:
        """恐怖表情の解釈
        
        - 親しい相手 → 共感的恐怖（BASE層）
        - 敵対的相手 → 優越感（CORE層の負圧）
        """
        intensity = obs.signal_intensity
        
        if obs.relationship > 0.3:  # 親密な関係
            # 共感的恐怖伝染（BASE層）
            return {
                HumanLayer.PHYSICAL: 0.0,
                HumanLayer.BASE: intensity * 0.8 * obs.relationship,
                HumanLayer.CORE: 0.0,
                HumanLayer.UPPER: 0.0
            }
        elif obs.relationship < -0.3:  # 敵対的関係
            # 敵の恐怖 → 自己の優越感（CORE層の安定化）
            return {
                HumanLayer.PHYSICAL: 0.0,
                HumanLayer.BASE: -intensity * 0.3,  # 負の圧力 = 安心
                HumanLayer.CORE: -intensity * 0.2,
                HumanLayer.UPPER: 0.0
            }
        else:  # 中立的関係
            # わずかな警戒感
            return {
                HumanLayer.PHYSICAL: 0.0,
                HumanLayer.BASE: intensity * 0.2,
                HumanLayer.CORE: intensity * 0.1,
                HumanLayer.UPPER: 0.0
            }
    
    def _interpret_anger_expression(
        self,
        observer: HumanAgent,
        obs: ObservationContext
    ) -> Dict[HumanLayer, float]:
        """怒り表情の解釈
        
        - 自分に向けられた怒り → 脅威（BASE層）
        - 第三者への怒り → 観察（CORE層）
        """
        intensity = obs.signal_intensity
        
        # コンテキストから「怒りの対象」を判定
        is_directed_at_self = obs.context_data and \
            obs.context_data.get('anger_target') == obs.observer_id
        
        if is_directed_at_self:
            # 直接的脅威
            return {
                HumanLayer.PHYSICAL: intensity * 0.3,  # 身体的緊張
                HumanLayer.BASE: intensity * 0.7,      # 本能的恐怖
                HumanLayer.CORE: intensity * 0.4,      # 社会的危機
                HumanLayer.UPPER: 0.0
            }
        else:
            # 第三者への怒り → 社会的緊張の観察
            return {
                HumanLayer.PHYSICAL: 0.0,
                HumanLayer.BASE: intensity * 0.2,
                HumanLayer.CORE: intensity * 0.3,
                HumanLayer.UPPER: 0.0
            }
    
    def _interpret_cooperative_act(
        self,
        observer: HumanAgent,
        obs: ObservationContext
    ) -> Dict[HumanLayer, float]:
        """協力的行動の解釈
        
        - 親しい相手 → 信頼感（負の圧力 = 安心）
        - 敵対的相手 → 疑念（UPPER層での解釈負荷）
        """
        intensity = obs.signal_intensity
        
        if obs.relationship > 0.3:
            # 信頼できる協力 → 圧力の軽減
            return {
                HumanLayer.PHYSICAL: -intensity * 0.2,
                HumanLayer.BASE: -intensity * 0.4,
                HumanLayer.CORE: -intensity * 0.3,
                HumanLayer.UPPER: 0.0
            }
        elif obs.relationship < -0.3:
            # 敵の協力 → 疑念・警戒
            return {
                HumanLayer.PHYSICAL: 0.0,
                HumanLayer.BASE: intensity * 0.3,     # 警戒
                HumanLayer.CORE: intensity * 0.5,     # 裏を読む
                HumanLayer.UPPER: intensity * 0.4     # 動機の解釈
            }
        else:
            # 中立的な協力 → わずかな好感
            return {
                HumanLayer.PHYSICAL: 0.0,
                HumanLayer.BASE: -intensity * 0.1,
                HumanLayer.CORE: -intensity * 0.2,
                HumanLayer.UPPER: 0.0
            }
    
    def _interpret_aggressive_act(
        self,
        observer: HumanAgent,
        obs: ObservationContext
    ) -> Dict[HumanLayer, float]:
        """攻撃的行動の解釈"""
        intensity = obs.signal_intensity
        
        # 自分への攻撃か？
        is_attack_on_self = obs.context_data and \
            obs.context_data.get('attack_target') == obs.observer_id
        
        if is_attack_on_self:
            # 直接的脅威
            return {
                HumanLayer.PHYSICAL: intensity * 0.8,
                HumanLayer.BASE: intensity * 0.9,
                HumanLayer.CORE: intensity * 0.6,
                HumanLayer.UPPER: intensity * 0.3
            }
        else:
            # 第三者への攻撃 → 間接的脅威
            return {
                HumanLayer.PHYSICAL: intensity * 0.2,
                HumanLayer.BASE: intensity * 0.4,
                HumanLayer.CORE: intensity * 0.5,
                HumanLayer.UPPER: 0.0
            }
    
    def _interpret_verbal_ideology(
        self,
        observer: HumanAgent,
        obs: ObservationContext
    ) -> Dict[HumanLayer, float]:
        """言語化されたイデオロギーの解釈
        
        - 自己の信念と一致 → UPPER層の強化（負圧）
        - 自己の信念と対立 → UPPER層の葛藤（正圧）
        """
        intensity = obs.signal_intensity
        
        # コンテキストから「イデオロギーの一致度」を取得
        alignment = obs.context_data.get('ideology_alignment', 0.0) \
            if obs.context_data else 0.0
        
        if alignment > 0.5:  # 一致
            # 信念の強化
            return {
                HumanLayer.PHYSICAL: 0.0,
                HumanLayer.BASE: 0.0,
                HumanLayer.CORE: -intensity * 0.2 * alignment,
                HumanLayer.UPPER: -intensity * 0.4 * alignment
            }
        elif alignment < -0.5:  # 対立
            # イデオロギー的葛藤
            return {
                HumanLayer.PHYSICAL: 0.0,
                HumanLayer.BASE: 0.0,
                HumanLayer.CORE: intensity * 0.4 * abs(alignment),
                HumanLayer.UPPER: intensity * 0.7 * abs(alignment)
            }
        else:  # 中立
            return {
                HumanLayer.PHYSICAL: 0.0,
                HumanLayer.BASE: 0.0,
                HumanLayer.CORE: intensity * 0.1,
                HumanLayer.UPPER: intensity * 0.2
            }
    
    def _interpret_norm_violation(
        self,
        observer: HumanAgent,
        obs: ObservationContext
    ) -> Dict[HumanLayer, float]:
        """規範違反の観測
        
        - 自己のκ_coreが高い → 強い反発（CORE層）
        - 自己のκ_coreが低い → 弱い反応
        """
        intensity = obs.signal_intensity
        
        # 観測者自身の規範意識（κ_core）に依存
        observer_kappa_core = observer.state.kappa[HumanLayer.CORE.value]
        norm_sensitivity = min(observer_kappa_core / 2.0, 1.0)  # 正規化
        
        return {
            HumanLayer.PHYSICAL: 0.0,
            HumanLayer.BASE: intensity * 0.2 * norm_sensitivity,
            HumanLayer.CORE: intensity * 0.8 * norm_sensitivity,
            HumanLayer.UPPER: intensity * 0.3 * norm_sensitivity
        }
    
    def _interpret_norm_adherence(
        self,
        observer: HumanAgent,
        obs: ObservationContext
    ) -> Dict[HumanLayer, float]:
        """規範遵守の観測
        
        - 自己のκ_coreが高い → 好意的評価（負圧）
        """
        intensity = obs.signal_intensity
        
        observer_kappa_core = observer.state.kappa[HumanLayer.CORE.value]
        norm_sensitivity = min(observer_kappa_core / 2.0, 1.0)
        
        return {
            HumanLayer.PHYSICAL: 0.0,
            HumanLayer.BASE: 0.0,
            HumanLayer.CORE: -intensity * 0.3 * norm_sensitivity,
            HumanLayer.UPPER: -intensity * 0.2 * norm_sensitivity
        }
    
    def _interpret_default(
        self,
        observer: HumanAgent,
        obs: ObservationContext
    ) -> Dict[HumanLayer, float]:
        """デフォルト解釈（未定義シグナル）"""
        return {
            HumanLayer.PHYSICAL: 0.0,
            HumanLayer.BASE: 0.0,
            HumanLayer.CORE: 0.0,
            HumanLayer.UPPER: 0.0
        }


# ========================================
# 便利関数
# ========================================

def create_fear_observation(
    observer_id: str,
    target_id: str,
    fear_level: float,
    relationship: float = 0.5,
    distance: float = 0.0
) -> ObservationContext:
    """恐怖表情の観測コンテキスト生成"""
    return ObservationContext(
        observer_id=observer_id,
        target_id=target_id,
        signal_type=ObservableSignal.FEAR_EXPRESSION,
        signal_intensity=fear_level,
        relationship=relationship,
        distance=distance
    )


def create_ideology_observation(
    observer_id: str,
    target_id: str,
    ideology_strength: float,
    alignment: float,  # -1.0 (対立) ~ +1.0 (一致)
    relationship: float = 0.0,
    distance: float = 0.2
) -> ObservationContext:
    """イデオロギー発言の観測コンテキスト生成"""
    return ObservationContext(
        observer_id=observer_id,
        target_id=target_id,
        signal_type=ObservableSignal.VERBAL_IDEOLOGY,
        signal_intensity=ideology_strength,
        relationship=relationship,
        distance=distance,
        context_data={'ideology_alignment': alignment}
    )
