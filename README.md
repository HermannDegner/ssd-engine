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

## 高速化オプション

Numbaを使った高速版が利用可能です：

```bash
# 高速版をインストール
pip install -e ".[fast]"
```

Numbaがインストールされていると、計算が自動的に5-50倍高速化されます。
Numbaがない場合は通常版が使われます（自動フォールバック）。

```python
from ssd_engine.core.ssd_core_engine_fast import NUMBA_AVAILABLE

if NUMBA_AVAILABLE:
    print("✓ Numba加速が有効です")
else:
    print("通常版を使用します")
```

## C++ / Native Implementation

高性能が必要な場合、C++実装も利用可能です：

- [SSD-Universal-Engine](https://github.com/HermannDegner/SSD-Universal-Engine) - C++ DLL実装
  - さらなる高速化（C++ネイティブ）
  - DLL/共有ライブラリとしてビルド
  - Python ctypesから呼び出し可能

Python版との比較：
- **Python + Numba**: 4.38倍高速（470K steps/sec）
- **C++ DLL**: 推定10-100倍高速（数百万 steps/sec）
