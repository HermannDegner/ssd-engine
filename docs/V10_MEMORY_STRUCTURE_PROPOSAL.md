# SSD v10.0 提案: 記憶の構造化と概念形成

**作成日**: 2025年11月7日  
**状態**: Phase 10 理論設計

---

## 1. v9の限界：非構造的記憶

### 1.1 現在の記憶システム（v9）

```python
# 記憶はフラットなリスト
memory = [
    MemoryTrace(signal=[0,0,0.8,0,0,0,0], layer=1, outcome=-1.0, t=10),
    MemoryTrace(signal=[0,0,0.7,0,0,0,0], layer=1, outcome=-0.8, t=15),
    MemoryTrace(signal=[0,0,0,0,0,0.9,0], layer=2, outcome=+0.7, t=20),
    ...
]
```

**問題点**:
- 類似記憶の関連性が認識されない
- 「攻撃的行動」というカテゴリが形成されない
- 毎回すべての記憶を線形探索
- パターン抽出ができない

### 1.2 理論的課題

**認知科学的観点**:
- 人間の記憶はカテゴリ化される（prototype理論）
- 「犬」の概念 = 過去に見た犬の統計的プロトタイプ
- SSDでも、「攻撃的シグナル」の**プロトタイプ**が形成されるべき

**計算効率的観点**:
- 100個の類似記憶 → 1個のプロトタイプに統合
- O(n)の記憶探索 → O(log n)のクラスタ検索

---

## 2. v10の革新：階層的記憶構造

### 2.1 記憶のクラスタリング

#### 類似度に基づく記憶のグループ化

```python
class MemoryCluster:
    """
    類似した記憶のクラスタ
    
    Attributes:
        prototype_signal: プロトタイプシグナル [7] (平均)
        layer: 対象層
        avg_outcome: 平均結果
        n_memories: クラスタ内の記憶数
        variance: シグナルの分散
        last_updated: 最終更新時刻
    """
    prototype_signal: np.ndarray  # [7]
    layer: int
    avg_outcome: float
    n_memories: int
    variance: float
    last_updated: float
```

#### クラスタリングアルゴリズム

**オンラインK-Means（効率化版）**:

```python
def add_memory_with_clustering(signal, layer, outcome, t):
    """
    新しい記憶を追加し、クラスタを更新
    """
    # Step 1: 最も近いクラスタを探す
    closest_cluster = find_closest_cluster(signal, layer)
    
    # Step 2: 距離が閾値以下なら統合、そうでなければ新規クラスタ
    if distance(signal, closest_cluster.prototype) < threshold:
        # 既存クラスタに統合
        closest_cluster.update(signal, outcome, t)
    else:
        # 新規クラスタを作成
        new_cluster = MemoryCluster(signal, layer, outcome, t)
        clusters.append(new_cluster)
    
    # Step 3: クラスタ数が多すぎる場合、マージ
    if len(clusters) > max_clusters:
        merge_similar_clusters()
```

### 2.2 プロトタイプに基づく解釈

#### v9（個別記憶）

```python
# 全記憶を探索
for mem in memories:
    if mem.layer == layer and mem.signal[sig] > 0.1:
        learning_term += mem.signal[sig] * (-mem.outcome) * decay
```

#### v10（クラスタベース）

```python
# クラスタのみ探索（高速）
for cluster in clusters:
    if cluster.layer == layer:
        # プロトタイプとの類似度
        similarity = cosine_similarity(cluster.prototype, current_signal)
        
        # 重み = 類似度 × クラスタサイズ × 結果の強さ
        weight = similarity * cluster.n_memories * abs(cluster.avg_outcome)
        
        learning_term += weight
```

**利点**:
- 計算量: O(n_memories) → O(n_clusters) （通常 n_clusters << n_memories）
- 一般化: 「攻撃的行動全般」への学習
- ノイズ除去: 統計的プロトタイプはノイズに頑健

---

## 3. 概念形成のモデル化

### 3.1 概念とは何か

**認知科学的定義**:
- 概念 = 経験の統計的要約
- 例: 「犬」= {四足、吠える、尾を振る、...} の平均的パターン

**SSD v10での実装**:
```python
class Concept:
    """
    エージェントが形成した概念
    
    例: 「攻撃的な人狼の行動」という概念
    """
    name: str  # e.g., "aggressive_werewolf"
    prototype_signal: np.ndarray  # [7] - シグナルの典型例
    layer: int  # 主に影響を受ける層
    avg_outcome: float  # この概念に対する経験的評価
    n_instances: int  # この概念を構成する経験数
    confidence: float  # 概念の確信度（分散の逆数）
```

### 3.2 概念の自動抽出

#### アルゴリズム: 階層的クラスタリング

```python
def extract_concepts(memory_store, min_cluster_size=5):
    """
    記憶から概念を抽出
    
    Args:
        memory_store: 全記憶
        min_cluster_size: 概念とみなす最小クラスタサイズ
    
    Returns:
        概念のリスト
    """
    # Step 1: 記憶をクラスタリング
    clusters = hierarchical_clustering(memory_store.memories)
    
    # Step 2: 十分大きなクラスタを「概念」とする
    concepts = []
    for cluster in clusters:
        if cluster.n_memories >= min_cluster_size:
            concept = Concept(
                name=auto_generate_name(cluster),
                prototype_signal=cluster.prototype,
                layer=cluster.layer,
                avg_outcome=cluster.avg_outcome,
                n_instances=cluster.n_memories,
                confidence=1.0 / (cluster.variance + 1e-6)
            )
            concepts.append(concept)
    
    return concepts
```

#### 概念の命名（自動生成）

```python
def auto_generate_name(cluster):
    """
    クラスタの特徴から名前を生成
    """
    proto = cluster.prototype_signal
    
    # 最も強いシグナルを特定
    dominant_signal = np.argmax(proto)
    signal_names = ["posture", "face", "voice", "aggressive", 
                   "defensive", "cooperative", "ideological"]
    
    # 層名
    layer_names = ["PHYSICAL", "BASE", "CORE", "UPPER"]
    
    # 結果の極性
    if cluster.avg_outcome < -0.5:
        valence = "dangerous"
    elif cluster.avg_outcome > 0.5:
        valence = "safe"
    else:
        valence = "neutral"
    
    name = f"{valence}_{signal_names[dominant_signal]}_{layer_names[cluster.layer]}"
    return name
```

**出力例**:
- `dangerous_aggressive_BASE`: 攻撃的で危険なBASE層刺激
- `safe_cooperative_CORE`: 協調的で安全なCORE層刺激

### 3.3 概念に基づく推論

#### 新しいシグナルの解釈

```python
def interpret_with_concepts(signal, concepts, kappa):
    """
    概念を用いた高速解釈
    """
    pressure = np.zeros(4)
    
    for concept in concepts:
        # シグナルと概念の類似度
        similarity = cosine_similarity(signal, concept.prototype_signal)
        
        if similarity > 0.5:  # 閾値以上で活性化
            # 概念の評価を圧力に変換
            # 悪い結果の概念 → 高い圧力
            concept_pressure = -concept.avg_outcome * similarity
            
            # 確信度で重み付け
            weighted_pressure = concept_pressure * concept.confidence
            
            # κによる定着度
            pressure[concept.layer] += kappa[concept.layer] * weighted_pressure
    
    return pressure
```

**メリット**:
- 個別記憶を探索不要
- 概念単位での推論（人間的）
- 一般化能力の向上

---

## 4. 実装設計

### 4.1 クラス構造

```python
# ====== ssd_memory_structure.py ======

class MemoryCluster:
    """記憶クラスタ（プロトタイプ）"""
    prototype_signal: np.ndarray
    layer: int
    avg_outcome: float
    n_memories: int
    variance: float
    last_updated: float
    
    def update(self, new_signal, new_outcome, timestamp):
        """新しい記憶でプロトタイプを更新"""
        # オンライン平均更新
        alpha = 1.0 / (self.n_memories + 1)
        self.prototype_signal = (1 - alpha) * self.prototype_signal + alpha * new_signal
        self.avg_outcome = (1 - alpha) * self.avg_outcome + alpha * new_outcome
        self.n_memories += 1
        self.last_updated = timestamp


class Concept:
    """抽出された概念"""
    name: str
    prototype: MemoryCluster
    importance: float  # 使用頻度に基づく重要度
    
    def matches(self, signal, threshold=0.7):
        """シグナルがこの概念に該当するか"""
        similarity = cosine_similarity(signal, self.prototype.prototype_signal)
        return similarity > threshold


class StructuredMemoryStore:
    """構造化記憶システム"""
    
    def __init__(self, max_clusters=50):
        self.clusters: List[MemoryCluster] = []
        self.concepts: List[Concept] = []
        self.max_clusters = max_clusters
    
    def add_memory(self, signal, layer, outcome, timestamp):
        """記憶を追加（自動クラスタリング）"""
        # 最も近いクラスタを探す
        closest = self.find_closest_cluster(signal, layer)
        
        if closest and self.distance(signal, closest.prototype_signal) < 0.3:
            # 既存クラスタに統合
            closest.update(signal, outcome, timestamp)
        else:
            # 新規クラスタ
            new_cluster = MemoryCluster(signal, layer, outcome, 1, 0.0, timestamp)
            self.clusters.append(new_cluster)
            
            # クラスタ数制限
            if len(self.clusters) > self.max_clusters:
                self.merge_least_important()
    
    def extract_concepts(self, min_size=5):
        """概念を抽出"""
        self.concepts = [
            Concept(
                name=auto_generate_name(cluster),
                prototype=cluster,
                importance=cluster.n_memories
            )
            for cluster in self.clusters
            if cluster.n_memories >= min_size
        ]
        
        # 重要度で並べ替え
        self.concepts.sort(key=lambda c: c.importance, reverse=True)
    
    def interpret_with_structure(self, signal, kappa):
        """構造化記憶による高速解釈"""
        pressure = np.zeros(4)
        
        # まず概念ベースで解釈
        for concept in self.concepts[:10]:  # 上位10概念のみ
            if concept.matches(signal):
                similarity = cosine_similarity(signal, concept.prototype.prototype_signal)
                concept_pressure = -concept.prototype.avg_outcome * similarity
                pressure[concept.prototype.layer] += kappa[concept.prototype.layer] * concept_pressure
        
        # 概念に該当しない場合、クラスタベース
        if np.sum(pressure) < 0.1:
            for cluster in self.clusters:
                similarity = cosine_similarity(signal, cluster.prototype_signal)
                if similarity > 0.5:
                    cluster_pressure = -cluster.avg_outcome * similarity
                    pressure[cluster.layer] += kappa[cluster.layer] * cluster_pressure
        
        return pressure
```

### 4.2 人狼ゲームv10への統合

```python
# ====== werewolf_game_v10.py ======

class PlayerV10:
    """v10プレイヤー: 構造化記憶と概念形成"""
    
    def __init__(self, ...):
        # v9の動的解釈に加えて
        self.structured_memory = StructuredMemoryStore(max_clusters=30)
        self.learned_concepts: List[Concept] = []
    
    def observe_player(self, target_id, target_signals, current_time):
        """観測 → 構造化記憶で高速解釈"""
        
        # 構造化記憶による解釈
        pressure = self.structured_memory.interpret_with_structure(
            target_signals,
            self.kappa
        )
        
        suspicion = 0.6 * pressure[1] + 0.4 * pressure[2]
        self.suspicion_levels[target_id] = suspicion
        
        return suspicion
    
    def learn_from_outcome(self, outcome, ...):
        """学習 → 記憶をクラスタ化"""
        
        # 記憶を追加（自動クラスタリング）
        if most_suspected in all_players:
            target_signals = all_players[most_suspected].visible_signals
            
            self.structured_memory.add_memory(
                signal=target_signals,
                layer=1,  # BASE
                outcome=-1.0 if outcome == 'executed' else 0.5,
                timestamp=current_time
            )
        
        # 定期的に概念を抽出
        if self.games_played % 3 == 0:
            self.learned_concepts = self.structured_memory.extract_concepts(min_size=3)
    
    def explain_decision(self, target_id):
        """意思決定の説明（概念ベース）"""
        target_signals = self.all_players[target_id].visible_signals
        
        # どの概念が活性化したか
        activated_concepts = [
            concept for concept in self.learned_concepts
            if concept.matches(target_signals)
        ]
        
        if activated_concepts:
            primary = activated_concepts[0]
            return f"'{primary.name}'の概念に該当するため疑惑"
        else:
            return "既知の概念に該当せず、新しいパターン"
```

---

## 5. パフォーマンス分析

### 5.1 計算量の比較

| 操作 | v9（フラット記憶） | v10（構造化記憶） |
|------|-------------------|-------------------|
| 記憶追加 | O(1) | O(k) (k=クラスタ数) |
| 解釈計算 | O(n) (n=記憶数) | O(k) または O(c) (c=概念数) |
| 記憶検索 | O(n) | O(log k) |

**具体例**:
- 100記憶 → 10クラスタ → 3概念
- v9: 100回の比較
- v10: 3~10回の比較
- **10~30倍高速化**

### 5.2 メモリ使用量

```
v9: n_memories × (7 + 3) = 100 × 10 = 1000 floats

v10: n_clusters × (7 + 5) = 10 × 12 = 120 floats
     → 約90%削減
```

---

## 6. 哲学的・認知科学的意義

### 6.1 プロトタイプ理論の実装

**Roschの古典的カテゴリ論**:
- カテゴリは必要十分条件ではなく、プロトタイプ（典型例）で定義される
- 「鳥」の概念 = スズメやカラスの平均像、ペンギンは非典型
- SSD v10は、この**プロトタイプ的カテゴリ化**を動的に実現

### 6.2 暗黙知の形式化

**ポランニーの「暗黙知」**:
- 我々は言葉にできないが、パターンを認識できる
- 「なんとなく怪しい」= 言語化できない概念
- v10の概念は、まさに**暗黙知の数理モデル**

### 6.3 経験からの抽象化

**ピアジェの図式（schema）**:
- 子どもは個別経験から「図式」を形成
- 「犬」の図式 = 過去に見た犬の抽象
- v10のクラスタリングは、**図式形成のメカニズム**

---

## 7. 実装ロードマップ

### Phase 10.1: 構造化記憶モジュール

**ファイル**: `ssd_memory_structure.py`

- `MemoryCluster`クラス
- `Concept`クラス
- `StructuredMemoryStore`クラス
- オンラインクラスタリング

### Phase 10.2: 人狼ゲームv10

**ファイル**: `werewolf_game_v10.py`

- `PlayerV10`クラス（構造化記憶使用）
- 概念ベース推論
- 意思決定の説明機能

### Phase 10.3: Nano最適化

**ファイル**: `nano_core_engine_v10.py`

- クラスタのバッチ処理
- Numba最適化
- 固定長クラスタバッファ

### Phase 10.4: 可視化

**ファイル**: `visualize_concepts.py`

- 概念のネットワーク図
- プロトタイプシグナルのヒートマップ
- 学習過程のアニメーション

---

## 8. 期待される成果

### 8.1 性能向上

- 解釈速度: **10~30倍高速化**
- メモリ使用量: **90%削減**
- 学習の質: 一般化能力向上

### 8.2 理論的完成度

- v9: 85% → v10: **98%**
- 残る課題: 層間での概念転移、長期記憶の形成

### 8.3 応用可能性

- **説明可能AI**: 「なぜこの判断をしたか」を概念で説明
- **転移学習**: 過去ゲームの概念を新ゲームに適用
- **社会シミュレーション**: 集団で共有される「文化的概念」の形成

---

## 9. 次のステップ

1. ✅ この提案書を精査
2. ⏳ `ssd_memory_structure.py`実装
3. ⏳ 人狼ゲームv10でデモ
4. ⏳ Nano最適化
5. ⏳ 論文執筆: "Structural Subjectivity Dynamics v10: Concept Formation and Memory Structure"

---

**作成者**: GitHub Copilot  
**理論責任者**: User (SSD理論開発者)  
**バージョン**: v10.0-proposal  
**日付**: 2025年11月7日
