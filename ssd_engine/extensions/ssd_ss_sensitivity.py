# -*- coding: utf-8 -*-
"""
SSD Sensory Sensitivity (SS) Extension
=====================================

HSP/SS型の個人差を神経変調システムに統合。
感覚過敏・高感度センサー・場依存整合の実装。

SS型の二経路モデル:
- 経路A: 高感度センサー優位（場読み・空気社会促進）
- 経路B: 脅威感受性優位（扁桃体反応・ストレス系）

KPI計測機能:
- CAR: 場依存整合率
- LP: 言語圧力  
- CCL: 空気コスト
- XAL: 異整合変換損失
"""

from dataclasses import dataclass, replace
from typing import Optional, Dict, Tuple
import numpy as np
from .ssd_neuro_modulators import NeuroState, NeuroConfig, modulate_params

# -------- SS型パラメータ --------
@dataclass
class SSProfile:
    """Sensory Sensitivity (感覚過敏) プロファイル"""
    ss_level: float = 0.5           # SS度合い (0..1)
    pathway_balance: float = 0.5    # 経路バランス (0=A優位, 1=B優位)
    context_dependency: float = 0.7 # 文脈依存度 (SS社会適応)
    stress_threshold: float = 0.3   # ストレス転換閾値
    
    # 個別感受性
    sensory_gain: float = 1.5       # 感覚ゲイン倍率
    fatigue_rate: float = 1.3       # 疲労蓄積率
    stabilization_seek: float = 1.2 # 安定化指向
    threat_sensitivity: float = 1.4  # 脅威感受性

# -------- SS特化変調設定 --------
@dataclass
class SSNeuroConfig(NeuroConfig):
    """SS型特化神経変調設定"""
    # SS経路A: 高感度センサー優位
    k_ss_sense_gain: float = 0.40   # 感覚ゲイン強化
    k_ss_fatigue: float = 0.25      # 疲労蓄積↑
    k_ss_stabilize: float = 0.20    # 安定化指向↑
    
    # SS経路B: 脅威感受性優位  
    k_ss_barrier_sharp: float = 0.30 # 発火障壁鋭化
    k_ss_noise_amp: float = 0.15     # 熱ノイズ増幅
    k_ss_leap_prone: float = 0.20    # LEAP促進
    
    # ストレス転換パラメータ
    stress_transition_rate: float = 0.1  # A→B転換速度

# -------- 社会・言語KPI --------
@dataclass
class SocialLanguageKPI:
    """社会・言語統合の計測指標"""
    CAR: float = 0.0  # Context Alignment Rate (場依存整合率)
    LP: float = 0.0   # Linguistic Pressure (言語圧力)
    CCL: float = 0.0  # Contextual Cognitive Load (空気コスト)
    XAL: float = 0.0  # Cross-Alignment Loss (異整合変換損失)
    
    # 計算用内部状態
    explicit_info: float = 0.0      # 明示情報量
    implicit_info: float = 0.0      # 暗黙情報量
    context_resolved: float = 0.0   # 文脈で解決された残差
    total_residual: float = 0.0     # 総残差
    inference_steps: int = 0        # 推論ステップ数

# -------- SS変調関数 --------
def modulate_ss_params(core_params, ss_profile: SSProfile, 
                      current_stress: float = 0.0,
                      cfg: Optional[SSNeuroConfig] = None):
    """
    SS型プロファイルによるパラメータ変調
    
    Args:
        core_params: SSDCoreParams
        ss_profile: SS型プロファイル
        current_stress: 現在ストレス水準 (0..1)
        cfg: SS変調設定
    """
    if cfg is None:
        cfg = SSNeuroConfig()
    
    q = replace(core_params)
    ss = ss_profile.ss_level
    
    # ストレス転換: A→B遷移判定
    stress_trigger = current_stress > ss_profile.stress_threshold
    pathway_weight_A = ss_profile.pathway_balance * (1.0 - stress_trigger * cfg.stress_transition_rate)
    pathway_weight_B = (1.0 - ss_profile.pathway_balance) * (1.0 + stress_trigger * cfg.stress_transition_rate)
    
    # 経路A: 高感度センサー優位
    if pathway_weight_A > 0.1:
        # 1) 感覚ゲイン↑ (微細不整合の強化感知)
        sense_enhancement = 1.0 + cfg.k_ss_sense_gain * ss * pathway_weight_A
        q.alpha0 = max(1e-3, core_params.alpha0 * sense_enhancement)
        
        # 2) 疲労蓄積↑ (常時小残差による負荷)
        # 仮想的に未処理圧の熱化を促進 (実装時にalpha_Et相当を調整)
        fatigue_factor = 1.0 + cfg.k_ss_fatigue * ss * pathway_weight_A
        
        # 3) 安定化指向↑ (放熱強化・跳躍抑制)
        stabilize_factor = 1.0 + cfg.k_ss_stabilize * ss * pathway_weight_A
        q.beta_values = [beta * stabilize_factor for beta in core_params.beta_values]
    
    # 経路B: 脅威感受性優位
    if pathway_weight_B > 0.1:
        # 4) 発火障壁鋭化 (LEAPしやすさ↑)
        barrier_sharp = 1.0 + cfg.k_ss_barrier_sharp * ss * pathway_weight_B
        q.Theta_values = [theta / barrier_sharp for theta in core_params.Theta_values]
        
        # 5) 熱ノイズ増幅 (感情的揺らぎ↑)
        noise_amp = 1.0 + cfg.k_ss_noise_amp * ss * pathway_weight_B
        q.epsilon_noise = max(1e-6, core_params.epsilon_noise * noise_amp)
        
        # 6) LEAP促進 (跳躍活動性↑)
        leap_factor = 1.0 + cfg.k_ss_leap_prone * ss * pathway_weight_B
        q.gamma_values = [gamma * leap_factor for gamma in core_params.gamma_values]
    
    return q

# -------- 神経状態生成 --------
def ss_to_neuro_state(ss_profile: SSProfile, current_stress: float = 0.0) -> NeuroState:
    """SS型プロファイルから神経状態を生成"""
    
    # ベース神経状態
    base_d1 = 0.3
    base_d2 = 0.3  
    base_ne = 0.3
    base_5ht = 0.3
    base_ach = 0.3
    
    ss = ss_profile.ss_level
    stress_factor = min(1.0, current_stress * 2.0)  # ストレス効果
    
    # ストレス転換判定
    if current_stress > ss_profile.stress_threshold:
        # 経路B優位: 脅威感受・跳躍モード
        d1 = base_d1 + 0.4 * ss * stress_factor      # 探索・行動活性↑
        d2 = base_d2 - 0.2 * ss * stress_factor      # 抑制↓
        ne = base_ne + 0.5 * ss * stress_factor      # 覚醒・警戒↑
        _5ht = base_5ht - 0.3 * ss * stress_factor   # 制御力↓
        ach = base_ach + 0.2 * ss                    # 注意集中
    else:
        # 経路A優位: 高感度・場依存モード
        d1 = base_d1 + 0.2 * ss                      # 適度な探索
        d2 = base_d2 + 0.1 * ss                      # バランス抑制
        ne = base_ne + 0.3 * ss                      # 感度向上
        _5ht = base_5ht + 0.4 * ss                   # 制御・安定化↑
        ach = base_ach + 0.5 * ss                    # 高注意・微細感知
    
    # 正規化
    return NeuroState(
        D1=max(0.0, min(1.0, d1)),
        D2=max(0.0, min(1.0, d2)),
        NE=max(0.0, min(1.0, ne)),
        _5HT=max(0.0, min(1.0, _5ht)),
        ACh=max(0.0, min(1.0, ach))
    )

# -------- KPI計算関数 --------
def compute_social_language_kpi(explicit_info: float, implicit_info: float,
                               context_resolved: float, total_residual: float,
                               inference_steps: int) -> SocialLanguageKPI:
    """社会・言語KPIの計算"""
    
    kpi = SocialLanguageKPI()
    
    # 内部状態更新
    kpi.explicit_info = explicit_info
    kpi.implicit_info = implicit_info  
    kpi.context_resolved = context_resolved
    kpi.total_residual = total_residual
    kpi.inference_steps = inference_steps
    
    # CAR: 場依存整合率
    if total_residual > 1e-6:
        kpi.CAR = context_resolved / total_residual
    else:
        kpi.CAR = 1.0
    
    # LP: 言語圧力 (明示度)
    total_info = explicit_info + implicit_info
    if total_info > 1e-6:
        kpi.LP = explicit_info / total_info
    else:
        kpi.LP = 0.5
    
    # CCL: 空気コスト (推論負荷)
    kpi.CCL = float(inference_steps)  # 簡易版：ステップ数そのまま
    
    # XAL: 異整合変換損失 (仮想計算)
    # SS→LL変換での残差増分を模擬
    ss_efficiency = kpi.CAR  # 場依存整合の効率
    ll_efficiency = kpi.LP   # 明示整合の効率
    kpi.XAL = abs(ss_efficiency - ll_efficiency) * total_residual
    
    return kpi

# -------- SS型プリセット --------
def ss_preset(profile_name: str) -> SSProfile:
    """SS型プリセット生成"""
    
    name = profile_name.lower()
    
    if name in ("high_ss", "強感受性"):
        return SSProfile(
            ss_level=0.8,
            pathway_balance=0.3,  # 経路A優位
            context_dependency=0.9,
            stress_threshold=0.2,  # 低ストレス閾値
            sensory_gain=2.0,
            fatigue_rate=1.8
        )
    
    elif name in ("balanced_ss", "バランス感受"):
        return SSProfile(
            ss_level=0.5,
            pathway_balance=0.5,  # 経路バランス
            context_dependency=0.6,
            stress_threshold=0.4,
            sensory_gain=1.3,
            fatigue_rate=1.2
        )
    
    elif name in ("stress_reactive", "ストレス反応型"):
        return SSProfile(
            ss_level=0.7,
            pathway_balance=0.7,  # 経路B寄り
            context_dependency=0.4,
            stress_threshold=0.1,  # 極低ストレス閾値
            threat_sensitivity=2.0
        )
    
    else:  # デフォルト
        return SSProfile()

# -------- 統合変調関数 --------  
def modulate_with_ss(core_params, ss_profile: SSProfile, 
                    current_stress: float = 0.0,
                    neuro_config: Optional[NeuroConfig] = None,
                    ss_config: Optional[SSNeuroConfig] = None):
    """
    SS型 + 通常神経変調の統合適用
    
    Returns:
        Tuple[modulated_params, neuro_state, ss_kpi_placeholder]
    """
    
    # 1) SS型による基本変調
    ss_modulated = modulate_ss_params(core_params, ss_profile, current_stress, ss_config)
    
    # 2) SS型から神経状態生成
    neuro_state = ss_to_neuro_state(ss_profile, current_stress)
    
    # 3) 神経変調レイヤーを追加適用
    if neuro_config is not None:
        final_modulated = modulate_params(ss_modulated, neuro_state, neuro_config)
    else:
        final_modulated = ss_modulated
    
    # 4) KPI計算用プレースホルダー
    kpi = SocialLanguageKPI()  # 実際の計算は実行時
    
    return final_modulated, neuro_state, kpi

# -------- デモ用ヘルパー --------
def demonstrate_ss_effects():
    """SS型効果のデモンストレーション"""
    
    print("🧠✨ SS型 (感覚過敏) 神経変調デモ")
    print("=" * 50)
    
    from core.ssd_core_engine_log import SSDCoreParams
    
    base_params = SSDCoreParams(temperature_T=37.0, alpha0=1.0)
    
    # 各SS型プロファイルでの変調効果
    profiles = {
        "通常": SSProfile(ss_level=0.0),
        "強感受性": ss_preset("high_ss"),
        "バランス": ss_preset("balanced_ss"), 
        "ストレス反応型": ss_preset("stress_reactive")
    }
    
    for name, profile in profiles.items():
        print(f"\n🎯 {name} (SS={profile.ss_level:.1f}):")
        
        # 平常時
        modulated, neuro, kpi = modulate_with_ss(base_params, profile, current_stress=0.2)
        print(f"  平常時: α0={modulated.alpha0:.3f}, Θ[0]={modulated.Theta_values[0]:.1f}")
        print(f"         神経: D1={neuro.D1:.2f}, 5HT={neuro._5HT:.2f}, NE={neuro.NE:.2f}")
        
        # 高ストレス時
        modulated, neuro, kpi = modulate_with_ss(base_params, profile, current_stress=0.8) 
        print(f"  高ストレス: α0={modulated.alpha0:.3f}, Θ[0]={modulated.Theta_values[0]:.1f}")
        print(f"           神経: D1={neuro.D1:.2f}, 5HT={neuro._5HT:.2f}, NE={neuro.NE:.2f}")

if __name__ == "__main__":
    demonstrate_ss_effects()
