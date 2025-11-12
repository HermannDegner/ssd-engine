"""
SSD Engine - Structural Subjectivity Dynamics Library
=====================================================

構造主観力学（SSD）理論の統合ライブラリ

このパッケージは、SSD理論の完全実装を提供します：
- Core: コアエンジンと基本モジュール
- Extensions: 拡張機能（記憶、神経変調、社会ダイナミクスなど）
- Experimental: 実験的機能
"""

__version__ = "2.0.0"
__author__ = "SSD Research Team"

# Core modules - よく使う機能を直接インポート可能に
from ssd_engine.core import (
    # Core Engine
    SSDCoreEngine,
    SSDCoreParams,
    SSDCoreState,
    LeapType,
    
    # Human Module
    HumanAgent,
    HumanParams,
    HumanPressure,
    HumanLayer,
    
    # Pressure System
    MultidimensionalPressureEngine,
    PressureDimension,
    StructuralLayer,
    
    # Nonlinear Transfer
    NonlinearInterlayerTransfer,
    NonlinearTransferFunction,
)

__all__ = [
    # Core Engine
    'SSDCoreEngine',
    'SSDCoreParams',
    'SSDCoreState',
    'LeapType',
    
    # Human Module
    'HumanAgent',
    'HumanParams',
    'HumanPressure',
    'HumanLayer',
    
    # Pressure System
    'MultidimensionalPressureEngine',
    'PressureDimension',
    'StructuralLayer',
    
    # Nonlinear Transfer
    'NonlinearInterlayerTransfer',
    'NonlinearTransferFunction',
]
