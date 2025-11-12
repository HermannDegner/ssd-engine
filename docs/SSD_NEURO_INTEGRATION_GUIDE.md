# SSD神経変調システム - 使用ガイド

## 🧠⚡ 完成した神経変調アーキテクチャ

### ファイル構成
```
extensions/
├── ssd_neuro_modulators.py     # 🧠 神経変調コア（ドロップイン）
examples/demos/
├── demo_neuro_integration.py   # 🔬 統合デモ・使用例
```

### 核心設計
**完全分離アーキテクチャ**: 物理エンジン ↔ 神経変調層

```python
物理エンジン（ssd_core_engine_log.py）
    ↑ パラメータ非破壊変調
神経変調層（ssd_neuro_modulators.py）
    ↑ 受容体レベル制御
神経状態（D1/D2/NE/5HT/ACh）
```

## 🚀 最小統合方法

### 1. インポート
```python
from extensions.ssd_neuro_modulators import NeuroState, modulate_params, neuro_preset
```

### 2. 基本使用（関数的アプローチ）
```python
# 通常のエンジン初期化
engine = SSDCoreEngine(params)

# 神経状態設定
neuro_state = NeuroState(D1=0.7, D2=0.2, NE=0.8, _5HT=0.3, ACh=0.4)

# ステップ実行時に変調適用
for step in range(100):
    # 神経変調をかけたパラメータで実行
    modulated_params = modulate_params(engine.params, neuro_state)
    
    # 一時的に置き換えて実行
    original_params = engine.params
    engine.params = modulated_params
    state = engine.step(state, pressure, dt=0.1)
    engine.params = original_params  # 復元
```

### 3. クラス統合（推奨）
```python
class SSDNeuroEngine(SSDCoreEngine):
    def __init__(self, params):
        super().__init__(params)
        self.base_params = params
        self.neuro_state = NeuroState()  # デフォルト
    
    def step(self, state, pressure, dt=0.1):
        # 自動的に神経変調を適用
        modulated_params = modulate_params(self.base_params, self.neuro_state)
        self.params = modulated_params
        return super().step(state, pressure, dt)

# 使用
engine = SSDNeuroEngine(params)
engine.neuro_state = neuro_preset("explore")  # 探索モード
```

## 🧠 神経変調効果

### 受容体別制御
```python
# ドーパミン D1（促進系）
D1 ↑ → 感覚ゲイン↑、LEAP閾値↓、活動性↑、学習率↑

# ドーパミン D2（抑制系） 
D2 ↑ → LEAP閾値↑、安定性↑、活動抑制

# ノルアドレナリン（覚醒/探索）
NE ↑ → 感覚ゲイン↑、活動性↑、探索温度↑、導電性↑

# セロトニン（制御/安定化）
5HT ↑ → 感覚抑制、安定性↑、ノイズ↓、導電制御

# アセチルコリン（注意/学習）
ACh ↑ → 学習率↑、注意集中
```

### パラメータ変調マッピング
```python
変調対象:
├── alpha0        # Log-Alignment感覚ゲイン
├── Theta_values  # LEAP閾値（各レイヤー）
├── gamma_values  # エネルギー生成（活動性）
├── beta_values   # エネルギー減衰（安定性）
├── eta_values    # 学習率（可塑性）
├── G0, g         # オーム則導電性
├── temperature_T # 探索温度
└── epsilon_noise # ノイズレベル
```

## 🎯 プリセット神経状態

### すぐ使えるプリセット
```python
# 集中モード（鎮静寄り、LEAP控えめ）
neuro_preset("focus")    # D1=0.4, 5HT=0.5, ACh=0.6

# 探索モード（活発、LEAP促進）  
neuro_preset("explore")  # D1=0.7, NE=0.7, 5HT=0.2

# 鎮静モード（安定、LEAP抑制）
neuro_preset("calm")     # D1=0.2, D2=0.5, 5HT=0.7
```

### カスタム神経状態
```python
# カイジ「最後の賭け」状態
kaiji_desperate = NeuroState(
    D1=0.9,   # 極度の期待・報酬追求
    D2=0.1,   # 抑制機能低下
    NE=0.9,   # 極限覚醒状態
    _5HT=0.1, # 制御機能麻痺
    ACh=0.2   # 注意散漫
)

# 研究者「深い集中」状態
researcher_flow = NeuroState(
    D1=0.6,   # 適度な報酬予期
    D2=0.3,   # バランス抑制
    NE=0.4,   # 落ち着いた覚醒
    _5HT=0.7, # 高い制御力
    ACh=0.8   # 最大注意集中
)
```

## ⚙️ 高度な使用法

### 動的神経状態変化
```python
# 時間進行に応じた神経状態変化
for t in range(1000):
    if t < 300:  # 序盤：冷静
        engine.neuro_state = neuro_preset("calm")
    elif t < 700:  # 中盤：探索
        engine.neuro_state = neuro_preset("explore")  
    else:  # 終盤：集中
        engine.neuro_state = neuro_preset("focus")
    
    state = engine.step(state, pressure[t], dt=0.1)
```

### 外部刺激連動
```python
# ストレスレベルに応じた自動調整
def adaptive_neuro_state(stress_level):
    if stress_level < 0.3:
        return neuro_preset("calm")
    elif stress_level < 0.7:
        return neuro_preset("focus") 
    else:
        return neuro_preset("explore")

# 使用
engine.neuro_state = adaptive_neuro_state(current_stress)
```

### カスタム変調設定
```python
# 変調強度のカスタマイズ
custom_config = NeuroConfig(
    k_sense_D1=0.50,    # D1感覚ゲイン効果を強化
    k_theta_D1=-0.40,   # D1のLEAP促進効果を強化
    k_temp_NE=0.30      # NEの探索温度効果を強化
)

modulated_params = modulate_params(params, neuro_state, custom_config)
```

## 🔬 実験・研究用途

### A/Bテスト
```python
# 神経状態の効果比較
results = {}
for name, neuro_state in [("baseline", NeuroState()), 
                         ("high_d1", NeuroState(D1=0.8)),
                         ("high_5ht", NeuroState(_5HT=0.8))]:
    engine.neuro_state = neuro_state
    leap_count = run_simulation(engine, 1000)
    results[name] = leap_count
```

### パラメータ効果分析
```python
# 変調前後のパラメータ比較
base_params = SSDCoreParams()
neuro_state = NeuroState(D1=0.8, NE=0.7)
modulated = modulate_params(base_params, neuro_state)

print(f"Theta変化: {base_params.Theta_values[0]} → {modulated.Theta_values[0]}")
print(f"感覚ゲイン変化: {base_params.alpha0} → {modulated.alpha0}")
```

## 🚧 拡張ポイント

### 新受容体追加
```python
@dataclass 
class NeuroState:
    # 既存受容体
    D1: float = 0.3
    D2: float = 0.3
    # 新受容体追加
    GABA: float = 0.3  # 抑制性
    Glu: float = 0.3   # 興奮性
```

### 新変調ターゲット
```python
# 新しいパラメータ変調追加
def modulate_params(core_params, neuro, cfg):
    # 既存変調...
    
    # 新変調追加
    q.some_new_param = p.some_new_param * (1.0 + cfg.k_new_D1 * s_curve(neuro.D1))
```

---

**🎉 完成度**: 即座に使用可能なプロダクション品質
**🔗 統合性**: 既存コードへの最小変更
**🧠 拡張性**: 受容体・変調ターゲット容易追加
**⚡ 分離性**: 物理エンジンと完全独立