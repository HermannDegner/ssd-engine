# SSD Engine - 構造主観力学ライブラリ

Structural Subjectivity Dynamics (SSD) Theory の Python 実装ライブラリです。

## 特徴

- **Core Engine**: Log-Alignment対応SSDコアエンジン（正式版）
- **Human Module**: 人間心理特化モジュール
- **Pressure System**: 多次元意味圧システム
- **Extensions**: 記憶構造、神経変調、社会ダイナミクスなど
- **Experimental**: 実験的機能（Nano Core Engineなど）

## インストール

### 開発モード（編集可能）
```bash
pip install -e .
```

### 通常インストール
```bash
pip install .
```

## 使い方

```python
from ssd_engine.core import SSDCoreEngine, SSDCoreParams, SSDCoreState

# エンジン初期化
params = SSDCoreParams()
state = SSDCoreState(num_layers=4)
engine = SSDCoreEngine(params, state)

# 意味圧を与えて処理
pressure = [1.0, 0.5, 0.2, 0.1]
result = engine.step(pressure)
```

### 人間モジュール

```python
from ssd_engine.core import HumanAgent, HumanPressure

agent = HumanAgent()
pressure = HumanPressure(survival=0.8, social=0.5)
agent.step(pressure)
```

### 拡張機能

```python
from ssd_engine.extensions import SubjectiveSociety, MemoryStructure
```

## モジュール構成

- `ssd_engine.core`: コアエンジンと基本モジュール
  - `SSDCoreEngine`: 汎用計算エンジン
  - `HumanAgent`: 人間心理特化エージェント
  - `MultidimensionalPressureEngine`: 多次元意味圧システム
  - `NonlinearInterlayerTransfer`: 非線形層間転送

- `ssd_engine.extensions`: 拡張機能
  - 記憶構造、神経変調、社会ダイナミクスなど

- `ssd_engine.experimental`: 実験的機能
  - Nano Core Engine など

## ライセンス

MIT License

## バージョン

2.0.0 - Log-Alignment + Neuro + SS integration
