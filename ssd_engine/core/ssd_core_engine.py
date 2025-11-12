"""
SSD Core Engine - 汎用計算エンジン（Log-Alignment対応版）
============================================================

構造主観力学（SSD）の基本数理を実装した、ドメイン非依存の計算エンジン。
対数整合層により大信号への適応性と数値安定性を強化。

核心概念:
- 意味圧 (p): 構造に作用する外部/内部エネルギー
- 対数整合 (p̂): 符号保持log変換による適応的入力処理
- 整合慣性 (κ): 経路の使いやすさ（学習痕跡）
- 未処理圧 (E): 処理しきれなかった圧力の蓄積
- 抵抗 (R): 構造の動かしにくさ
- 臨界閾値 (Theta): 跳躍を引き起こす閾値

理論的基盤:
- Log-Alignment: p̂ = sign(p)·log(1+α_t|p|)/log(b)
- Ohm's law analogy: j = (G0 + g·κ)·p̂
- Energy accumulation: E蓄積 = 意味圧 - 処理能力
- Leap trigger: E ≥ Theta → 構造的跳躍

参考: https://github.com/HermannDegner/Structural-Subjectivity-Dynamics
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum, auto


class LeapType(Enum):
    """跳躍タイプ（ドメイン非依存）"""
    NO_LEAP = auto()
    LEAP_LAYER_1 = auto()
    LEAP_LAYER_2 = auto()
    LEAP_LAYER_3 = auto()
    LEAP_LAYER_4 = auto()
    # 必要に応じて追加可能


@dataclass
class SSDCoreParams:
    """
    SSD汎用パラメータ（Log-Alignment対応）
    
    レイヤー数やドメインに依存しない基本パラメータセット
    """
    # レイヤー構成
    num_layers: int = 4
    
    # 各レイヤーのパラメータ（配列として指定）
    R_values: List[float] = field(default_factory=lambda: [1000.0, 100.0, 10.0, 1.0])
    
    # エネルギー生成パラメータ（各レイヤー）
    gamma_values: List[float] = field(default_factory=lambda: [0.15, 0.10, 0.08, 0.05])
    
    # エネルギー減衰パラメータ（各レイヤー）
    beta_values: List[float] = field(default_factory=lambda: [0.001, 0.01, 0.05, 0.1])
    
    # κ学習率（各レイヤー）
    eta_values: List[float] = field(default_factory=lambda: [0.9, 0.5, 0.3, 0.2])
    
    # κ減衰率（各レイヤー）
    lambda_values: List[float] = field(default_factory=lambda: [0.001, 0.01, 0.02, 0.05])
    
    # κ最小値（各レイヤー）
    kappa_min_values: List[float] = field(default_factory=lambda: [0.9, 0.8, 0.5, 0.3])
    
    # Theta閾値（各レイヤー）
    Theta_values: List[float] = field(default_factory=lambda: [200.0, 100.0, 50.0, 30.0])
    
    # Dynamic Theta パラメータ
    enable_dynamic_theta: bool = True
    theta_sensitivity: float = 0.3
    
    # 確率的跳躍パラメータ（温度T）
    enable_stochastic_leap: bool = False  # False=決定論的、True=確率的
    temperature_T: float = 0.0  # 0=完全決定論、>0=確率性増加
    
    # Ohm's law パラメータ
    G0: float = 0.5  # ベース導電率
    g: float = 0.7   # 慣性ゲイン
    
    # ノイズ
    epsilon_noise: float = 0.01
    
    # ===== Log-Alignment パラメータ =====
    log_align: bool = True  # 対数整合の有効化（既定で有効）
    alpha0: float = 1.0  # 基準ゲイン
    log_base: float = np.e  # 対数底（e または 10）
    ema_tau: float = 0.98  # EMA減衰定数（1 - 1/N）
    eps_log: float = 1e-6  # ゼロ除算防止
    
    # スケール係数（物理残差モード用、ログ残差モードでは不要）
    use_log_residual: bool = True  # True=ログ空間残差、False=物理スケール残差
    zeta_auto: bool = True  # ζの自動推定
    zeta_init: float = 1.0  # ζの初期値
    zeta_min: float = 1e-3  # ζの下限
    zeta_max: float = 1e3  # ζの上限
    tau_zeta: float = 0.99  # ζのEMA減衰
    
    # クリッピング範囲
    alpha_min: float = 1e-2  # α_tの下限
    alpha_max: float = 10.0  # α_tの上限
    
    # ウォームアップ
    warmup_steps: int = 50  # ウォームアップ期間（ステップ数）
    
    def __post_init__(self):
        """パラメータ配列の長さを検証"""
        arrays = [
            self.R_values, self.gamma_values, self.beta_values,
            self.eta_values, self.lambda_values, self.kappa_min_values,
            self.Theta_values
        ]
        for arr in arrays:
            if len(arr) != self.num_layers:
                raise ValueError(f"パラメータ配列の長さがnum_layers={self.num_layers}と一致しません")


@dataclass
class SSDCoreState:
    """
    SSD汎用状態ベクトル（Log-Alignment対応）
    
    レイヤー数に応じて動的にサイズが決まる
    """
    # 各レイヤーのエネルギー
    E: np.ndarray = field(default_factory=lambda: np.zeros(4))
    
    # 各レイヤーのκ
    kappa: np.ndarray = field(default_factory=lambda: np.ones(4))
    
    # 時間
    t: float = 0.0
    
    # ステップカウンタ（ウォームアップ判定用）
    step_count: int = 0
    
    # 跳躍履歴
    leap_history: List[Tuple[float, LeapType]] = field(default_factory=list)
    
    # Log-Alignment状態
    logalign_state: Dict = field(default_factory=lambda: {
        'm': 0.0,  # 入力ノルムのEMA
        'alpha_t': 1.0,  # 現在の適応ゲイン
        'zeta': 1.0  # スケール係数（物理残差モード用）
    })
    
    # 診断情報（オプション）
    diagnostics: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        """NumPy配列に変換"""
        if not isinstance(self.E, np.ndarray):
            self.E = np.array(self.E)
        if not isinstance(self.kappa, np.ndarray):
            self.kappa = np.array(self.kappa)


class SSDCoreEngine:
    """
    SSD汎用計算エンジン（Log-Alignment対応）
    
    任意のレイヤー数・パラメータセットで動作する計算エンジン。
    対数整合層により大信号への適応性と数値安定性を強化。
    ドメイン固有の解釈は上位モジュール（HumanModule等）が担当。
    """
    
    def __init__(self, params: SSDCoreParams):
        self.params = params
        self.num_layers = params.num_layers
        
    def apply_log_alignment(
        self,
        state: SSDCoreState,
        pressure: np.ndarray
    ) -> np.ndarray:
        """
        対数整合層の適用
        
        p̂ = sign(p)·log(1+α_t|p|)/log(b)
        α_t = α_0 / (ε + EMA_τ(|p|))
        
        Args:
            state: 現在の状態
            pressure: 入力意味圧
            
        Returns:
            変換後の意味圧 p̂
        """
        if not self.params.log_align:
            # 対数整合無効の場合はそのまま返す
            return pressure
        
        # 入力ノルムの計算
        norm_p = np.linalg.norm(pressure)
        
        # EMA更新
        state.logalign_state['m'] = (
            self.params.ema_tau * state.logalign_state['m'] +
            (1 - self.params.ema_tau) * norm_p
        )
        
        # 適応ゲイン計算
        alpha_t = self.params.alpha0 / (self.params.eps_log + state.logalign_state['m'])
        
        # クリッピング
        alpha_t = np.clip(alpha_t, self.params.alpha_min, self.params.alpha_max)
        state.logalign_state['alpha_t'] = alpha_t
        
        # 符号保持対数変換
        phat = np.sign(pressure) * np.log1p(alpha_t * np.abs(pressure)) / np.log(self.params.log_base)
        
        return phat
    
    def compute_structural_power(
        self,
        state: SSDCoreState,
        pressure_hat: np.ndarray
    ) -> np.ndarray:
        """
        構造的影響力の計算（変換後空間で評価）
        
        Power[i] = p̂[i] × E[i] × κ[i] × R[i]
        
        Args:
            pressure_hat: 対数変換後の意味圧
            
        Returns:
            各レイヤーの構造的影響力
        """
        if len(pressure_hat) != self.num_layers:
            raise ValueError(f"圧力ベクトルの長さが{self.num_layers}ではありません")
        
        R_array = np.array(self.params.R_values)
        power = pressure_hat * state.E * state.kappa * R_array
        return power
    
    def compute_dynamic_theta(
        self,
        state: SSDCoreState,
        pressure_hat: np.ndarray,
        layer_index: int
    ) -> float:
        """
        [Phase 2] 動的閾値の計算
        
        Theta_dynamic = Theta_base × (1 - sensitivity × structural_influence)
        
        structural_influence = (p̂ × E × κ × R) / (κ × R)
        """
        if not self.params.enable_dynamic_theta:
            return self.params.Theta_values[layer_index]
        
        # 構造的影響力
        power = self.compute_structural_power(state, pressure_hat)
        total_power = np.sum(power)
        
        # 正規化された影響
        R_array = np.array(self.params.R_values)
        denominator = np.sum(state.kappa * R_array)
        
        if denominator > 0:
            structural_influence = total_power / denominator
        else:
            structural_influence = 0.0
        
        # 動的Theta
        base_theta = self.params.Theta_values[layer_index]
        dynamic_theta = base_theta * (1.0 - self.params.theta_sensitivity * structural_influence)
        
        return max(1.0, dynamic_theta)  # 最小値1.0
    
    def detect_leap(
        self,
        state: SSDCoreState,
        pressure_hat: np.ndarray
    ) -> Tuple[bool, Optional[int]]:
        """
        跳躍検出（統合版）
        
        ウォームアップ期間中は跳躍を抑制
        
        Returns:
            (跳躍発生フラグ, 跳躍したレイヤーのインデックス)
        """
        # ウォームアップ中は跳躍抑制
        if state.step_count < self.params.warmup_steps:
            return False, None
        
        # 確率的跳躍が無効、または温度が0の場合は決定論的判定
        if not self.params.enable_stochastic_leap or self.params.temperature_T <= 0:
            # 決定論的跳躍（従来の実装）
            for i in range(self.num_layers):
                theta_i = self.compute_dynamic_theta(state, pressure_hat, i)
                
                if state.E[i] >= theta_i:
                    # 確率的跳躍判定（互換性のため残す）
                    leap_prob = min(1.0, (state.E[i] - theta_i) / theta_i)
                    if np.random.random() < leap_prob:
                        return True, i
            return False, None
        
        # 確率的跳躍（温度Tベース）
        for i in range(self.num_layers):
            theta_i = self.compute_dynamic_theta(state, pressure_hat, i)
            delta = state.E[i] - theta_i
            
            # シグモイド確率: P(leap) = 1 / (1 + exp(-delta / T))
            # 低T: ほぼ決定論（delta>0で確実、delta<0でほぼ0）
            # 高T: ランダム性増加（delta=0でも50%）
            prob = 1.0 / (1.0 + np.exp(-delta / self.params.temperature_T))
            
            if np.random.rand() < prob:
                return True, i
        
        return False, None
    
    def execute_leap(
        self,
        state: SSDCoreState,
        layer_index: int
    ) -> SSDCoreState:
        """
        跳躍の実行
        
        - エネルギーをリセット
        - κを微増（跳躍による学習）
        """
        new_state = SSDCoreState(
            E=state.E.copy(),
            kappa=state.kappa.copy(),
            t=state.t,
            step_count=state.step_count,
            leap_history=state.leap_history.copy(),
            logalign_state=state.logalign_state.copy()
        )
        
        # エネルギーリセット
        new_state.E[layer_index] *= 0.1
        
        # κ微増（跳躍による学習）
        new_state.kappa[layer_index] += 0.1
        
        # 跳躍履歴記録
        leap_type = LeapType(layer_index + 2)  # NO_LEAPを除いて2から開始
        new_state.leap_history.append((state.t, leap_type))
        
        return new_state
    
    def compute_energy_residual(
        self,
        state: SSDCoreState,
        pressure: np.ndarray,
        pressure_hat: np.ndarray,
        j: np.ndarray
    ) -> np.ndarray:
        """
        エネルギー残差の計算（モード別）
        
        Args:
            pressure: 元の意味圧
            pressure_hat: 変換後の意味圧
            j: 整合流
            
        Returns:
            各レイヤーの残差
        """
        if self.params.use_log_residual:
            # ログ空間残差
            resid = np.maximum(0.0, np.abs(pressure_hat) - np.abs(j))
        else:
            # 物理スケール残差（ζ自動推定）
            if self.params.zeta_auto:
                # ζのEMA更新
                norm_p = np.linalg.norm(pressure) + self.params.eps_log
                norm_j = np.linalg.norm(j) + self.params.eps_log
                zeta_new = state.logalign_state['zeta'] * self.params.tau_zeta + \
                          (1 - self.params.tau_zeta) * (norm_p / norm_j)
                
                # クリッピング
                state.logalign_state['zeta'] = np.clip(
                    zeta_new, self.params.zeta_min, self.params.zeta_max
                )
            
            zeta = state.logalign_state['zeta']
            resid = np.maximum(0.0, np.abs(pressure) - zeta * np.abs(j))
        
        # 診断用：残差ノルムを記録
        state.diagnostics['resid_norm'] = np.linalg.norm(resid)
        
        return resid
    
    def step(
        self,
        state: SSDCoreState,
        pressure: np.ndarray,
        dt: float = 0.1,
        interlayer_transfer: Optional[np.ndarray] = None
    ) -> SSDCoreState:
        """
        1ステップ実行（Log-Alignment対応）
        
        Args:
            state: 現在の状態
            pressure: 意味圧ベクトル（各レイヤー）
            dt: 時間刻み
            interlayer_transfer: 層間転送行列（オプション、上位モジュールが提供）
        
        Returns:
            更新後の状態
        """
        # 対数整合層の適用
        pressure_hat = self.apply_log_alignment(state, pressure)
        
        # 跳躍検出
        leap_occurred, leap_layer = self.detect_leap(state, pressure_hat)
        
        # 診断情報を記録（オプション）
        theta_dynamic = np.array([
            self.compute_dynamic_theta(state, pressure_hat, i) 
            for i in range(self.num_layers)
        ])
        power = self.compute_structural_power(state, pressure_hat)
        dominant_layer = int(np.argmax(power))
        
        # Theta動的平滑化（オプション、既定では無効）
        theta_smooth = state.diagnostics.get('theta_smooth', theta_dynamic.copy())
        # theta_smooth = 0.9 * theta_smooth + 0.1 * theta_dynamic  # 有効化する場合
        
        if leap_occurred:
            state = self.execute_leap(state, leap_layer)
        
        # 新しい状態
        new_state = SSDCoreState(
            E=state.E.copy(),
            kappa=state.kappa.copy(),
            t=state.t + dt,
            step_count=state.step_count + 1,
            leap_history=state.leap_history.copy(),
            logalign_state=state.logalign_state.copy(),
            diagnostics={
                'theta_dynamic': theta_dynamic.copy(),
                'theta_smooth': theta_smooth.copy(),
                'power': power.copy(),
                'dominant_layer': dominant_layer,
                'dominant_power': power[dominant_layer],
                'leap_occurred': leap_occurred,
                'leap_layer': leap_layer,
                'alpha_t': state.logalign_state['alpha_t'],
                'zeta': state.logalign_state['zeta'],
                'unit_check': 'log-space' if self.params.use_log_residual else f"phys-space ζ={state.logalign_state['zeta']:.3g}",
                'pressure_hat': pressure_hat.copy(),
                'pressure_hat_norm': np.linalg.norm(pressure_hat),
                'resid_norm': 0.0,  # compute_energy_residualで更新
                'eta_align_phys': None,  # 後で計算
                'eta_align_log': None,  # 後で計算
                'warmup_complete': state.step_count >= self.params.warmup_steps
            }
        )
        
        # 各レイヤーの更新
        R_array = np.array(self.params.R_values)
        gamma_array = np.array(self.params.gamma_values)
        beta_array = np.array(self.params.beta_values)
        eta_array = np.array(self.params.eta_values)
        lambda_array = np.array(self.params.lambda_values)
        kappa_min_array = np.array(self.params.kappa_min_values)
        
        # Ohm's law: j = (G0 + g·κ)·p̂
        conductance = self.params.G0 + self.params.g * state.kappa
        j = conductance * pressure_hat
        
        # エネルギー残差計算（モード別）
        resid = self.compute_energy_residual(state, pressure, pressure_hat, j)
        
        # エネルギー生成（残差ベース）
        # 抽象モード: 残差 ∝ 未整合量
        energy_generation = gamma_array * resid / R_array
        
        # 物理アナロジー強化モード（コメントアウト）:
        # energy_generation = gamma_array * resid * (np.abs(pressure_hat) / R_array)
        
        # エネルギー減衰
        energy_decay = beta_array * state.E
        
        # エネルギー更新
        dE = energy_generation - energy_decay
        
        # 層間転送があれば加算
        if interlayer_transfer is not None:
            dE += interlayer_transfer
        
        new_state.E = np.maximum(0.0, state.E + dE * dt)
        
        # κ更新（使用による強化と未使用減衰）
        usage_factor = np.abs(j) / (np.abs(j) + 1.0)  # 正規化された使用度
        dkappa = eta_array * usage_factor - lambda_array * state.kappa
        new_state.kappa = np.maximum(kappa_min_array, state.kappa + dkappa * dt)
        
        # KPI計算（診断用）
        norm_p = np.linalg.norm(pressure) + self.params.eps_log
        norm_phat = np.linalg.norm(pressure_hat) + self.params.eps_log
        norm_j = np.linalg.norm(j) + self.params.eps_log
        
        new_state.diagnostics['eta_align_phys'] = norm_j / norm_p
        new_state.diagnostics['eta_align_log'] = norm_j / norm_phat if self.params.log_align else new_state.diagnostics['eta_align_phys']
        new_state.diagnostics['compression_ratio'] = norm_phat / norm_p if self.params.log_align else 1.0
        
        # ウォームアップ完了イベント（1回だけ記録）
        if new_state.step_count == self.params.warmup_steps:
            new_state.diagnostics['warmup_event'] = True
        
        return new_state
    
    def get_dominant_layer(self, state: SSDCoreState, pressure: np.ndarray) -> int:
        """
        最も影響力の高いレイヤーを返す
        
        Args:
            pressure: 意味圧ベクトル
            
        Returns:
            最大構造的影響力を持つレイヤーのインデックス
        """
        pressure_hat = self.apply_log_alignment(state, pressure)
        power = self.compute_structural_power(state, pressure_hat)
        return int(np.argmax(power))
    
    def get_state_vector(self, state: SSDCoreState) -> Dict[str, np.ndarray]:
        """
        外部モジュール向けAPI: 状態ベクトルの取得
        
        上位層（人間モジュール／社会モジュール）が状態を読み取るためのインターフェース
        
        Returns:
            状態ベクトル辞書 {'E', 'kappa', 'alpha_t', 'zeta', etc.}
        """
        return {
            'E': state.E.copy(),
            'kappa': state.kappa.copy(),
            'alpha_t': state.logalign_state['alpha_t'],
            'zeta': state.logalign_state['zeta'],
            'm_ema': state.logalign_state['m'],
            't': state.t,
            'step_count': state.step_count
        }


# ============================================================================
# ユーティリティ関数
# ============================================================================

def create_default_state(num_layers: int = 4) -> SSDCoreState:
    """デフォルト状態の生成"""
    return SSDCoreState(
        E=np.zeros(num_layers),
        kappa=np.ones(num_layers),
        t=0.0,
        step_count=0,
        logalign_state={
            'm': 0.0,
            'alpha_t': 1.0,
            'zeta': 1.0
        }
    )


def create_custom_params(
    num_layers: int,
    R_values: List[float],
    **kwargs
) -> SSDCoreParams:
    """カスタムパラメータの生成ヘルパー"""
    return SSDCoreParams(
        num_layers=num_layers,
        R_values=R_values,
        **kwargs
    )


def print_diagnostics(state: SSDCoreState, step: int = None, verbose: bool = False):
    """
    診断情報の表示
    
    Args:
        state: 現在の状態
        step: ステップ番号（オプション）
        verbose: 詳細表示モード
    """
    if 'diagnostics' not in state.__dict__ or not state.diagnostics:
        print("診断情報がありません")
        return
    
    diag = state.diagnostics
    
    if step is not None:
        print(f"\n=== Step {step} ===")
    
    # ウォームアップ完了イベント
    if diag.get('warmup_event'):
        print("🔓 ウォームアップ完了：跳躍解禁")
    
    print(f"時刻: t={state.t:.2f}, ステップ数: {state.step_count}")
    print(f"単位系: {diag.get('unit_check', 'N/A')}")
    print(f"α_t: {diag.get('alpha_t', 0):.4f}, ζ: {diag.get('zeta', 0):.4f}")
    print(f"整合効率（物理）: {diag.get('eta_align_phys', 0):.4f}")
    print(f"整合効率（ログ）: {diag.get('eta_align_log', 0):.4f}")
    print(f"圧縮比: {diag.get('compression_ratio', 1.0):.4f}")
    print(f"支配レイヤー: {diag.get('dominant_layer', -1)} (パワー: {diag.get('dominant_power', 0):.2f})")
    
    if verbose:
        print(f"p̂ノルム: {diag.get('pressure_hat_norm', 0):.4f}")
        print(f"残差ノルム: {diag.get('resid_norm', 0):.4f}")
    
    if diag.get('leap_occurred'):
        print(f"*** 跳躍発生: Layer {diag.get('leap_layer')} ***")
    
    print(f"E: {state.E}")
    print(f"κ: {state.kappa}")


# ============================================================================
# 簡易テスト
# ============================================================================

if __name__ == "__main__":
    print("SSD Core Engine (Log-Alignment版) - 簡易テスト")
    print("=" * 60)
    
    # パラメータ設定
    params = SSDCoreParams(
        num_layers=4,
        log_align=True,
        use_log_residual=True,
        warmup_steps=10
    )
    
    # エンジン初期化
    engine = SSDCoreEngine(params)
    state = create_default_state(num_layers=4)
    
    # シミュレーション
    pressure = np.array([10.0, 5.0, 2.0, 1.0])
    
    print(f"\n初期入力: {pressure}")
    print(f"log_align: {params.log_align}")
    print(f"use_log_residual: {params.use_log_residual}")
    
    for step in range(100):
        state = engine.step(state, pressure, dt=0.1)
        
        if step in [0, 9, 10, 20, 50, 99]:
            print_diagnostics(state, step)
    
    print("\n" + "=" * 60)
    print("テスト完了")
