# SSD Engine ベンチマーク結果

## 環境
- CPU: 8コア
- Python: 3.11.9
- NumPy: 2.2.6
- Numba: 0.62.1
- テスト: 4層SSDエンジン × 100,000ステップ

## 実測結果

### Numbaなし（Pure Python + NumPy）
```
処理時間: 0.9276秒
1ステップあたり: 0.009276ms
スループット: 107,811 steps/sec
```

### ✨ Numba JIT最適化版
```
JITコンパイル: 1.10秒（初回のみ）
処理時間: 0.2117秒
1ステップあたり: 0.002117ms
スループット: 472,371 steps/sec
```

### 📊 高速化率
```
速度向上: 4.38倍
処理時間: 1/4.38に短縮
```

## 実用的な性能例

### Newton's Cradle物理シミュレーション（60fps）
- 1フレーム: 16.67ms
- Numbaなし: 1フレームで1,797ステップ可能
- **Numba有効: 1フレームで7,873ステップ可能** ⚡

### 大規模人狼ゲーム（100エージェント × 100ターン）
- 総計算: 10,000ステップ
- Numbaなし: 0.093秒
- **Numba有効: 0.021秒** ⚡

## 高速化の仕組み

1. **JITコンパイル**: Python → LLVMネイティブコード
2. **型推論**: 動的型 → 静的型最適化
3. **ループ最適化**: SIMD命令活用
4. **インライン化**: 関数呼び出しオーバーヘッド削減
5. **キャッシング**: 2回目以降はコンパイル不要

## インストール方法

```bash
# Numbaをインストール
pip install numba

# または、ssd-engineと一緒に
pip install "ssd-engine[fast]"
```

## 使用例

```python
from ssd_engine.core.ssd_core_engine_fast import NUMBA_AVAILABLE, compute_step_fast

if NUMBA_AVAILABLE:
    print("✓ JIT加速が有効です（4-5倍高速）")
else:
    print("通常版を使用中（依然として高速）")
```

## 注意事項

- ✅ 初回実行時にJITコンパイルが発生（~1秒）
- ✅ 2回目以降はキャッシュされた最適化コードを使用
- ✅ Numbaなしでも十分高速（10万steps/秒）
- ✅ 自動フォールバック：Numbaがなくても動作
