# SSD v9 提案: 動的解釈構造（Dynamic Interpretation Structure）

**作成日**: 2025年11月7日  
**状態**: Phase 9 理論設計

---

## 1. 問題の所在：v8.5における「静的主観」の矛盾

### 1.1 理論的矛盾

**SSDの核心原理**:
- **主観性**: エージェントは他者を「直接」観測できず、シグナルを「解釈」する
- **κ（整合慣性）**: 経験による学習・記憶によって、構造自体が変化する

**v8.5の実装**:
```python
# werewolf_game_v8_5.py (現状)
def calculate_suspicion(self, player_id, behavior):
    suspicion = 0.0
    suspicion += behavior.get("aggressive_act", 0) * 0.5  # ← 固定係数
    suspicion += behavior.get("defensive", 0) * 0.3      # ← 固定係数
    return suspicion
```

**矛盾点**:
- κは変化するのに、**解釈構造（係数0.5, 0.3）は固定**
- 「学習」が起きても、「解釈の仕方」は変わらない
- これは**主観の硬直化** = 理論的整合性の欠如

### 1.2 nano_core_engineでの同じ問題

```python
# nano_core_engine.py (現状)
def interpret_signals(observer_state, target_signals, relationship, distance):
    # signal_pressure_coeffs[layer, signal] は固定行列
    base_pressure = np.dot(params.signal_pressure_coeffs[0], target_signals)
    # ↑ 常に同じ解釈
```

**問題**:
- `signal_pressure_coeffs`は定数行列
- κが変化しても、シグナル→圧力の変換ロジックは不変
- **経験によって「見方」が変わらない**

---

## 2. 整合不能の具体例：人狼ゲーム

### シナリオ: Player 1の「学習失敗」

**Day 1**:
1. Player 2がPlayer 1に対して`aggressive_act`（強い非難）を行う
2. Player 1は`suspicion = 0.5 * aggressive_act`と計算
3. Player 1はPlayer 2を疑うが、投票で負ける
4. Player 1が処刑される（**強烈な整合不能**）

**Day 2（次のゲーム）**:
- Player 1は何も学んでいない
- また`aggressive_act * 0.5`で計算
- 同じ失敗を繰り返す

### 理論的にあるべき姿

**Day 1の処刑後**:
- Player 1のBASE層で強烈なE跳躍（生存欲求の破綻）
- κ_baseが急上昇（この経験を「刻み込む」）
- **解釈構造の書き換え**:
  ```
  aggressive_act → BASE層への圧力
  旧: 0.5 → 新: 0.9  （学習による変化）
  ```

**Day 2以降**:
- `aggressive_act`を受けた瞬間、BASE層が強く反応
- 「これは生存の脅威だ」という解釈が確立
- 投票行動が変わる（防衛的になる）

---

## 3. 理論的解決策：κ依存的解釈構造

### 3.1 基本アイデア

**解釈係数を動的化**:
```
I_ij^(layer)(t) = f(κ_i^(layer)(t), memory_ij(t))
```

- `I_ij^(layer)`: エージェントiがシグナルjを層に解釈する係数
- `κ_i^(layer)`: 層のκ値（経験の蓄積）
- `memory_ij`: 過去のシグナルjに関する記憶

### 3.2 数理モデル（提案）

#### Step 1: 記憶トレースの導入

各エージェントは、過去の「シグナル→圧力→結果」を記憶:

```
Memory_i = {
    (signal_pattern, layer, pressure, outcome, timestamp),
    ...
}
```

**例**:
- `signal_pattern = [0, 0, 0.8, 0, 0, 0, 0]`（signal 3が強い）
- `layer = BASE`
- `pressure = 0.5`（当時の解釈）
- `outcome = -1.0`（処刑された = 最悪の結果）
- `timestamp = t_event`

#### Step 2: 解釈係数の学習

κが高いほど、記憶に基づいて係数を修正:

```python
def compute_interpretation_coeff(layer, signal_idx, kappa, memory):
    """
    κ依存的解釈係数の計算
    """
    # 基本係数（初期値）
    base_coeff = default_coeffs[layer, signal_idx]
    
    # 記憶からの学習項
    learning_term = 0.0
    for mem in memory:
        if mem.signal_idx == signal_idx and mem.layer == layer:
            # 結果が悪いほど、係数を大きくする（警戒を強化）
            impact = -mem.outcome  # outcome が負 → impact が正
            decay = np.exp(-(t_now - mem.timestamp) / tau_memory)
            learning_term += impact * decay
    
    # κによる学習の定着度
    # κが高い = 経験をよく覚えている = 学習項の影響大
    learned_coeff = base_coeff + kappa * learning_term
    
    return np.clip(learned_coeff, 0.0, 1.0)
```

#### Step 3: 動的解釈行列の構築

```python
class DynamicInterpretationEngine:
    def __init__(self):
        self.memory = []  # [(signal, layer, pressure, outcome, t), ...]
        self.base_coeffs = np.array([...])  # 初期値
    
    def get_interpretation_matrix(self, kappa_vector, t_now):
        """
        現在のκと記憶に基づき、解釈行列を動的に生成
        """
        matrix = np.zeros((4, 7))  # [layer, signal]
        
        for layer in range(4):
            for sig in range(7):
                matrix[layer, sig] = self.compute_interpretation_coeff(
                    layer, sig, kappa_vector[layer], t_now
                )
        
        return matrix
    
    def interpret_signals(self, signals, kappa, t_now):
        """
        動的解釈
        """
        interp_matrix = self.get_interpretation_matrix(kappa, t_now)
        pressure = np.zeros(4)
        
        for layer in range(4):
            pressure[layer] = np.dot(interp_matrix[layer], signals)
        
        return pressure
    
    def record_experience(self, signal, layer, pressure, outcome, t):
        """
        経験を記憶に追加
        """
        self.memory.append({
            'signal': signal,
            'layer': layer,
            'pressure': pressure,
            'outcome': outcome,
            'timestamp': t
        })
```

---

## 4. 人狼ゲームv9への適用

### 4.1 学習可能な疑惑計算

```python
class PlayerV9:
    def __init__(self):
        self.interpretation_engine = DynamicInterpretationEngine()
        self.kappa = np.array([0.5, 0.5, 0.5, 0.5])  # 初期κ
    
    def calculate_suspicion(self, player_id, behavior, t_now):
        """
        動的解釈による疑惑計算
        """
        # 行動をシグナルベクトルに変換
        signal = self.behavior_to_signal(behavior)
        
        # 現在のκと記憶に基づいて解釈
        pressure = self.interpretation_engine.interpret_signals(
            signal, self.kappa, t_now
        )
        
        # BASE層の圧力 = 生存脅威 = 疑惑
        suspicion = pressure[1]  # BASE layer
        
        return suspicion
    
    def learn_from_execution(self, executed_player_id, behavior_history):
        """
        処刑から学ぶ
        """
        # 処刑された = 最悪の結果
        outcome = -1.0
        
        # 処刑直前の行動を記憶
        last_behavior = behavior_history[-1]
        signal = self.behavior_to_signal(last_behavior)
        
        # BASE層での経験として記録
        self.interpretation_engine.record_experience(
            signal=signal,
            layer=1,  # BASE
            pressure=self.calculate_suspicion(...),  # 当時の解釈
            outcome=outcome,
            t=t_now
        )
        
        # κの上昇（経験の定着）
        self.kappa[1] += 0.1  # BASE層で学習
```

### 4.2 学習の効果

**1回目のゲーム**:
- Player 1は`aggressive_act`を`0.5`の重みで解釈
- 処刑される
- 記憶に刻まれる:
  ```python
  memory.append({
      'signal': [0, 0, 0.8, 0, 0, 0, 0],  # aggressive_act強
      'layer': 1,  # BASE
      'pressure': 0.4,  # 0.5 * 0.8
      'outcome': -1.0,  # 処刑
      'timestamp': 100
  })
  ```

**2回目のゲーム**:
- κ_base = 0.6（学習済み）
- `aggressive_act`の解釈係数:
  ```python
  base_coeff = 0.5
  learning_term = -(-1.0) * exp(-Δt/τ) = 1.0 * 0.9 = 0.9
  learned_coeff = 0.5 + 0.6 * 0.9 = 1.04 → clip → 1.0
  ```
- 同じ`aggressive_act`を受けても、今度は**圧力1.0（最大警戒）**
- 防衛行動を取る確率が上がる

---

## 5. nano_core_engine v9への拡張

### 5.1 現在のv8.0

```python
class NanoCoreEngine:
    @staticmethod
    def interpret_signals(observer_state, target_signals, relationship, distance):
        # 固定係数
        pressure = np.zeros(4)
        for layer in range(4):
            pressure[layer] = np.dot(
                params.signal_pressure_coeffs[layer],  # ← 固定
                target_signals
            )
        return pressure
```

### 5.2 提案: v9.0 Dynamic Interpretation

```python
class NanoCoreEngineV9:
    @staticmethod
    def interpret_signals_dynamic(
        observer_state: NanoState,
        target_signals: np.ndarray,
        relationship: float,
        distance: float,
        memory: np.ndarray,  # NEW: [n_memory, signal_dim + 3]
        t_now: float
    ) -> np.ndarray:
        """
        動的解釈: κと記憶に基づいて解釈係数を変化させる
        
        memory shape: [n_memory, signal_dim + 3]
        memory columns: [signal_0, ..., signal_6, layer, outcome, timestamp]
        """
        kappa = observer_state.kappa  # [4]
        
        # 動的係数行列を計算
        dynamic_coeffs = compute_dynamic_coeffs(
            base_coeffs=params.signal_pressure_coeffs,  # [4, 7]
            kappa=kappa,  # [4]
            memory=memory,  # [n_memory, 10]
            t_now=t_now
        )
        
        # 動的解釈
        pressure = np.zeros(4)
        for layer in range(4):
            pressure[layer] = np.dot(dynamic_coeffs[layer], target_signals)
        
        return pressure

@njit
def compute_dynamic_coeffs(base_coeffs, kappa, memory, t_now):
    """
    記憶とκに基づき、解釈係数を動的に計算
    """
    n_layers, n_signals = base_coeffs.shape
    dynamic_coeffs = base_coeffs.copy()
    
    tau_memory = 100.0  # 記憶の減衰時定数
    
    for layer in range(n_layers):
        for sig in range(n_signals):
            learning_term = 0.0
            
            # 記憶から学習項を計算
            for i in range(len(memory)):
                mem_signal = memory[i, :n_signals]
                mem_layer = int(memory[i, n_signals])
                mem_outcome = memory[i, n_signals + 1]
                mem_time = memory[i, n_signals + 2]
                
                if mem_layer == layer:
                    # シグナルの類似度
                    similarity = mem_signal[sig]  # 簡易版: 該当シグナルの強度
                    
                    # 結果の影響（悪い結果ほど学習）
                    impact = -mem_outcome
                    
                    # 時間減衰
                    decay = np.exp(-(t_now - mem_time) / tau_memory)
                    
                    learning_term += similarity * impact * decay
            
            # κによる定着度
            dynamic_coeffs[layer, sig] += kappa[layer] * learning_term
            dynamic_coeffs[layer, sig] = np.clip(dynamic_coeffs[layer, sig], 0.0, 1.0)
    
    return dynamic_coeffs
```

---

## 6. 理論的意義

### 6.1 v8 → v9の飛躍

| 項目 | v8.5 | v9 (提案) |
|------|------|-----------|
| **主観的社会** | ✅ 実装済み | ✅ 継続 |
| **層別E・κ** | ✅ 独立管理 | ✅ 継続 |
| **解釈構造** | ❌ 静的（固定係数） | ✅ **動的（κ・記憶依存）** |
| **学習** | ⚠️ κのみ変化 | ✅ **解釈の仕方も変化** |
| **理論整合性** | 85% | **95%** |

### 6.2 哲学的含意

**現象学的主観性の完成**:
- v8.5: エージェントは「主観的に」世界を見る
- v9: エージェントの「見方」そのものが、経験によって変わる

これは**フッサールの「地平（Horizont）」概念**に対応:
- 地平 = 解釈の前提構造
- 地平は経験によって変容する
- SSD v9は、この「地平の変容」を数理的にモデル化

**κの二重の役割**:
1. **状態の慣性**（v8まで）: E跳躍への抵抗
2. **構造の可塑性**（v9）: 解釈構造の学習速度

---

## 7. 実装ロードマップ

### Phase 9.1: Reference実装（理論検証）

**ファイル**: `ssd_dynamic_interpretation.py`

```python
class DynamicInterpretationModule:
    - MemoryStore: 経験の記録
    - InterpretationMatrix: κ依存的係数
    - LearningEngine: 記憶からの学習
```

**テスト**: 人狼ゲームで「学習」を確認
- 同じ状況で、2回目の方が賢くなるか？

### Phase 9.2: Nano実装（最適化）

**ファイル**: `nano_core_engine_v9.py`

```python
class NanoCoreEngineV9:
    - interpret_signals_dynamic(): Numba最適化
    - compute_dynamic_coeffs(): JIT compiled
    - memory management: 固定長バッファ
```

**ベンチマーク目標**:
- 100 agents × 100 steps < 2.0s（v8: 1.667s）
- 記憶オーバーヘッド < 20%

### Phase 9.3: 人狼ゲームv9

**ファイル**: `werewolf_game_v9.py`

**新機能**:
- 過去ゲームの記憶を持つAI
- 「この攻撃パターンは危険だ」と学習
- プレイヤー個別の「解釈プロファイル」

---

## 8. 未解決課題（v10への布石）

### 8.1 記憶の選択的強化

**問題**: 全経験を平等に記憶すると、記憶が肥大化

**解決策**: 重要度に基づく記憶の選別
- |outcome|が大きい経験を優先的に保存
- 類似経験は統合（prototype化）

### 8.2 記憶の構造化

**問題**: 現在の記憶は「リスト」（非構造的）

**解決策**: 記憶のネットワーク化
- 類似シグナルの記憶をクラスタリング
- 「攻撃的な行動」カテゴリの形成
- → これは**概念形成**のモデル

### 8.3 多層的学習

**問題**: 現在は各層が独立に学習

**解決策**: 層間での学習の転移
- BASE層での学習がCORE層の解釈にも影響
- 「本能的警戒」が「社会的疑惑」を強化

---

## 9. 結論

### v9の核心的貢献

**「主観の動態化」**:
- v8: 主観的に見る（静的な主観）
- v9: 主観が変わる（動的な主観）

これにより、SSDは**真の学習系**になります。

### 次のステップ

1. ✅ この提案書を精査
2. ⏳ Reference実装（`ssd_dynamic_interpretation.py`）
3. ⏳ 人狼ゲームでの実証
4. ⏳ Nano最適化
5. ⏳ 論文執筆: "Dynamic Subjective Structures: Learning to Interpret"

---

**作成者**: GitHub Copilot  
**理論責任者**: User (SSD理論開発者)  
**バージョン**: v9.0-proposal  
**日付**: 2025年11月7日
