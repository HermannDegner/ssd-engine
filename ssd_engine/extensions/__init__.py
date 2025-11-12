"""
SSD Engine - Extensions
=======================

拡張機能モジュール
"""

# Memory Structure
from .ssd_memory_structure import StructuredMemoryStore, MemoryCluster, Concept

# Neuro Modulators  
from .ssd_neuro_modulators import NeuroState, NeuroConfig

# Subjective Society
from .ssd_subjective_society import SubjectiveSociety, SignalGenerator, AgentState

# Subjective Social Pressure
from .ssd_subjective_social_pressure import SubjectiveSocialPressureCalculator, ObservableSignal, ObservationContext

# Dynamic Interpretation
try:
    from .ssd_dynamic_interpretation import DynamicInterpretation
except ImportError:
    DynamicInterpretation = None

# External Knowledge
try:
    from .ssd_external_knowledge import KnowledgeType
except ImportError:
    KnowledgeType = None

# SS Sensitivity
from .ssd_ss_sensitivity import SSProfile, SSNeuroConfig, SocialLanguageKPI

# Social Dynamics (Legacy)
from .ssd_social_dynamics import Society, RelationshipMatrix, RelationType, SocialDynamicsEngine

__all__ = [
    'StructuredMemoryStore',
    'MemoryCluster',
    'Concept',
    'NeuroState',
    'NeuroConfig',
    'SubjectiveSociety',
    'SignalGenerator',
    'AgentState',
    'SubjectiveSocialPressureCalculator',
    'ObservableSignal',
    'ObservationContext',
    'SSProfile',
    'SSNeuroConfig',
    'SocialLanguageKPI',
    'Society',
    'RelationshipMatrix',
    'RelationType',
    'SocialDynamicsEngine',
]
