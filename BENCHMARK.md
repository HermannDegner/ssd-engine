# SSD Engine ベンチマーク結果

## 環境
- CPU: 8コア
- Python: 3.11
- NumPy: 2.2.6
- テスト: 4層SSDエンジン × 100,000ステップ

## 結果

### Numbaなし（Pure Python + NumPy）
```
処理時間: 0.9276秒
1ステップあたり: 0.009276ms
スループット: 107,811 steps/sec
```

### Numba最適化版（予測）
```
処理時間: ~0.05秒（予測）
1ステップあたり: ~0.0005ms（予測）
スループット: ~2,000,000 steps/sec（予測）
高速化率: 約18-20倍
```

## 高速化の仕組み

1. **JITコンパイル**: Python → ネイティブコード
2. **型推論**: 動的型 → 静的型最適化
3. **ループ最適化**: SIMD命令活用
4. **インライン化**: 関数呼び出しオーバーヘッド削減

## インストール方法

```bash
# 高速版（Numba込み）
pip install "ssd-engine[fast]"

# または
pip install numba
```

## 使用例

```python
from ssd_engine.core.ssd_core_engine_fast import NUMBA_AVAILABLE, compute_step_fast

if NUMBA_AVAILABLE:
    print("✓ JIT加速が有効です")
else:
    print("通常版を使用中（依然として高速）")
```

## 注意事項

- 初回実行時にJITコンパイルが発生（数秒かかる）
- 2回目以降はキャッシュされた最適化コードを使用
- Numbaなしでも十分高速（10万steps/秒）