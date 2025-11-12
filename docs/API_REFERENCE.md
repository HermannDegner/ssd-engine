# SSD Engine API Reference

## Core Modules

### SSDCoreEngine

汎用計算エンジン - Log-Alignment対応版

```python
from ssd_engine import SSDCoreEngine, SSDCoreParams, SSDCoreState

params = SSDCoreParams(
    num_layers=4,
    alpha_t=10.0,      # Log-Alignment強度
    log_base=10.0,     # 対数底
    G0=1.0,            # 基礎コンダクタンス
    g=0.1,             # 慣性依存コンダクタンス
    gamma=0.5,         # エネルギー蓄積率
    beta=0.1,          # 熱散逸率（人体基準: 310K）
    eta=0.01,          # 慣性学習率
    lambda_=0.05       # 慣性減衰率
)

state = SSDCoreState(num_layers=4)
engine = SSDCoreEngine(params, state)

# ステップ実行
pressure = [1.0, 0.5, 0.2, 0.1]  # 各層への意味圧
result = engine.step(pressure)
```

**主要な数式:**
- Log-Alignment: `p̂ = sign(p)·log(1+α|p|)/log(b)`
- Ohm's law: `j = (G0 + g·κ)·p̂`
- Energy: `dE = γ·residual - β·E`
- Inertia: `dκ = η·usage - λ·κ`

### HumanAgent

人間心理特化エージェント

```python
from ssd_engine import HumanAgent, HumanPressure

agent = HumanAgent()

# 意味圧を与える
pressure = HumanPressure(
    survival=0.8,    # 生存圧
    social=0.5,      # 社会圧
    cognitive=0.3    # 認知負荷
)

agent.step(pressure)

# 状態取得
print(f"Energy: {agent.get_energy()}")
print(f"Inertia: {agent.get_inertia()}")
```

### MultidimensionalPressureEngine

多次元意味圧システム

```python
from ssd_engine.core import MultidimensionalPressureEngine, PressureDimension

dimensions = [
    PressureDimension(name="survival", weight=1.0),
    PressureDimension(name="social", weight=0.8),
    PressureDimension(name="cognitive", weight=0.5)
]

engine = MultidimensionalPressureEngine(dimensions)
```

## Extensions

### SubjectiveSociety

主観的社会システム（Phase 8）

```python
from ssd_engine.extensions import SubjectiveSociety

society = SubjectiveSociety(num_agents=10)
society.step()
```

### StructuredMemoryStore

構造化記憶システム（V10）

```python
from ssd_engine.extensions import StructuredMemoryStore, Concept

memory = StructuredMemoryStore()
concept = Concept(name="危険", activation=0.8)
memory.store(concept)
```

### NeuroModulatorSystem

神経変調システム

```python
from ssd_engine.extensions import NeuroState, NeuroConfig

neuro_state = NeuroState()
config = NeuroConfig(dopamine_base=1.0, serotonin_base=1.0)
```

## Experimental

### NanoCoreEngine

軽量版エンジン（numba最適化）

```python
from ssd_engine.experimental import NanoCoreEngine

# numbaがインストールされている場合のみ利用可能
if NanoCoreEngine:
    engine = NanoCoreEngine()
```

## 理論ドキュメント

- [神経変調統合ガイド](SSD_NEURO_INTEGRATION_GUIDE.md)
- [熱力学研究サマリー](SSD_THERMAL_PHYSICS_RESEARCH_SUMMARY.md)
- [V10記憶構造提案](V10_MEMORY_STRUCTURE_PROPOSAL.md)
- [V9動的解釈提案](V9_DYNAMIC_INTERPRETATION_PROPOSAL.md)