"""
SSD External Knowledge Access - 外部固定情報アクセスモジュール
=================================================================

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
理論的位置づけ: 外部に固定された知識への動的アクセス
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

個人の外部に存在する「固定された情報」（本、論文、データベース、
文化的遺産など）へのアクセスによって、個人のκ（整合慣性）が
形成されるプロセスをモデル化する。

【理論的背景】

1. **外部固定情報とは**
   - 個人の脳の外部に保存された知識
   - 本、図書館、インターネット、師匠の教え
   - 時間的に安定（個人の記憶よりも永続的）

2. **アクセスによる意味圧生成**
   - 情報にアクセスする（読む、学ぶ）→ 意味圧が生成される
   - 内容に応じて、異なる層に異なる意味圧
   - 初めはフラット（単なる文字列）→ 理解が深まるにつれて構造化

3. **個人差の創発**
   - 同じ本を読んでも、個人のκ状態により受け取る意味圧が異なる
   - 準備のできた者（κが整った者）には深い意味が見える
   - 「師が現れる時、弟子は準備ができている」

【SSD基本理論との接続】

- 人間モジュール コア: 四層構造への意味圧入力
- メタ整合慣性システム: 知識の抽象化、圧縮
- 社会力学: 知識へのアクセス可能性の格差

【実装の特徴】

1. **内容依存的な意味圧生成**
   - 初期: フラットな文字列として扱う
   - 理解が深まる: κが形成され、より深い意味が抽出される
   - 再読: 同じ本でも、成長した自分には違う意味が見える

2. **アクセス可能性**
   - 物理的アクセス: 図書館の距離、本の価格
   - 認知的アクセス: 言語、前提知識、教育レベル
   - デジタル時代: アクセス障壁の低下

作成日: 2025年11月9日
バージョン: 1.0
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
from enum import Enum

import sys
import os
# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# coreモジュールのパスを追加
core_path = os.path.join(project_root, 'core')
if core_path not in sys.path:
    sys.path.insert(0, core_path)

from ssd_engine.core.ssd_human_module import HumanAgent, HumanPressure, HumanLayer


# ================================================================================
# 知識の種類
# ================================================================================

class KnowledgeType(Enum):
    """知識の種類"""
    PHYSICAL_SKILL = 0     # 身体技能（スポーツ、武術の型）
    PRACTICAL_GUIDE = 1    # 実用的ガイド（料理本、マニュアル）
    CULTURAL_NORM = 2      # 文化規範（法律、道徳書）
    THEORETICAL = 3        # 理論的知識（哲学書、科学論文）
    NARRATIVE = 4          # 物語（小説、神話）


# ================================================================================
# 外部知識ベース
# ================================================================================

@dataclass
class ExternalKnowledge:
    """
    外部固定情報（本、文書、データベース）
    
    【理論的意義】
    個人の外部に存在する「固定されたκパターン」。
    個人はこれにアクセスすることで、自己のκを形成する。
    
    人類の知的遺産の継承メカニズム。
    
    Attributes:
        title: 知識の名称
        content: 内容（テキスト、構造化データなど）
        knowledge_type: 知識の種類
        target_layer: 主に影響を与える層
        base_pressure: 基本的な意味圧の大きさ
        accessibility: アクセスしやすさ [0.0, 1.0]
        prerequisite_kappa: 理解に必要なκの最小値（層ごと）
    """
    title: str
    content: str
    knowledge_type: KnowledgeType
    target_layer: HumanLayer
    base_pressure: float = 50.0
    accessibility: float = 0.5
    prerequisite_kappa: np.ndarray = field(default_factory=lambda: np.ones(4))
    
    # メタデータ
    complexity: float = 0.5  # 複雑度 [0.0, 1.0]
    depth: float = 0.5       # 深さ（抽象度）
    
    def generate_pressure(
        self, 
        agent: HumanAgent, 
        access_duration: float = 1.0,
        understanding_level: float = 1.0
    ) -> HumanPressure:
        """
        知識へのアクセスから意味圧を生成
        
        【理論的意義】
        「読む」という行為 → 意味圧の生成 → κの更新
        
        初めて読む: フラットな意味圧（表面的理解）
        理解が深まる: κが形成され、より深い意味圧が抽出される
        
        Args:
            agent: アクセスするエージェント
            access_duration: アクセス時間（読書時間など）
            understanding_level: 理解度 [0.0, 1.0]（外部から指定可能）
        
        Returns:
            生成された意味圧
        """
        pressure = HumanPressure()
        
        # エージェントのκ状態を取得
        agent_kappa = agent.state.kappa[self.target_layer.value]
        prerequisite = self.prerequisite_kappa[self.target_layer.value]
        
        # 理解度の自動計算（κが高いほど深く理解できる）
        auto_understanding = min(1.0, agent_kappa / prerequisite) if prerequisite > 0 else 1.0
        
        # 最終的な理解度（自動計算と外部指定の積）
        final_understanding = auto_understanding * understanding_level
        
        # 基本意味圧
        base_p = self.base_pressure * access_duration * final_understanding
        
        # 層ごとの意味圧を設定（初期はフラット）
        if self.knowledge_type == KnowledgeType.PHYSICAL_SKILL:
            pressure.physical = base_p * 0.8
            pressure.base = base_p * 0.2
            
        elif self.knowledge_type == KnowledgeType.PRACTICAL_GUIDE:
            pressure.base = base_p * 0.7
            pressure.core = base_p * 0.3
            
        elif self.knowledge_type == KnowledgeType.CULTURAL_NORM:
            pressure.core = base_p * 0.9
            pressure.upper = base_p * 0.1
            
        elif self.knowledge_type == KnowledgeType.THEORETICAL:
            pressure.upper = base_p * 0.8
            pressure.core = base_p * 0.2
            
        elif self.knowledge_type == KnowledgeType.NARRATIVE:
            # 物語は全層に影響（バランス型）
            pressure.base = base_p * 0.3
            pressure.core = base_p * 0.4
            pressure.upper = base_p * 0.3
        
        # 複雑度による補正（複雑なほど理解が難しい）
        complexity_factor = 1.0 - (self.complexity * (1.0 - final_understanding))
        
        pressure.physical *= complexity_factor
        pressure.base *= complexity_factor
        pressure.core *= complexity_factor
        pressure.upper *= complexity_factor
        
        return pressure
    
    def is_accessible(self, agent: HumanAgent) -> Tuple[bool, str]:
        """
        エージェントがこの知識にアクセス可能かを判定
        
        Args:
            agent: エージェント
        
        Returns:
            (アクセス可能か, 理由)
        """
        # 前提知識チェック
        agent_kappa = agent.state.kappa[self.target_layer.value]
        required_kappa = self.prerequisite_kappa[self.target_layer.value]
        
        if agent_kappa < required_kappa * 0.5:  # 50%以下なら理解不能
            return False, f"前提知識不足（κ={agent_kappa:.2f} < 必要={required_kappa:.2f}）"
        
        # アクセス可能性チェック（確率的）
        if np.random.random() > self.accessibility:
            return False, "アクセス障壁（物理的・経済的・言語的制約）"
        
        return True, "アクセス可能"


# ================================================================================
# 知識ライブラリ
# ================================================================================

class KnowledgeLibrary:
    """
    知識ライブラリ（図書館、データベース、インターネット）
    
    【理論的意義】
    人類の集合的知識の保管庫。
    個人はここから選択的にアクセスし、自己のκを形成する。
    
    【デジタル時代の変化】
    - 物理的図書館: accessibility = 0.3（距離、開館時間の制約）
    - インターネット: accessibility = 0.9（いつでもどこでも）
    
    アクセス障壁の低下 → 知識民主化 + 情報過多
    """
    
    def __init__(self):
        self.knowledge_base: List[ExternalKnowledge] = []
    
    def add_knowledge(self, knowledge: ExternalKnowledge):
        """知識を追加"""
        self.knowledge_base.append(knowledge)
    
    def search(
        self, 
        agent: HumanAgent,
        keywords: Optional[List[str]] = None,
        knowledge_type: Optional[KnowledgeType] = None
    ) -> List[ExternalKnowledge]:
        """
        知識を検索
        
        Args:
            agent: 検索するエージェント
            keywords: キーワード（タイトルに含まれるか）
            knowledge_type: 知識の種類で絞り込み
        
        Returns:
            アクセス可能な知識のリスト
        """
        results = []
        
        for knowledge in self.knowledge_base:
            # アクセス可能性チェック
            accessible, _ = knowledge.is_accessible(agent)
            if not accessible:
                continue
            
            # 種類フィルタ
            if knowledge_type and knowledge.knowledge_type != knowledge_type:
                continue
            
            # キーワードフィルタ
            if keywords:
                if not any(kw.lower() in knowledge.title.lower() for kw in keywords):
                    continue
            
            results.append(knowledge)
        
        return results
    
    def get_random_knowledge(
        self, 
        agent: HumanAgent, 
        knowledge_type: Optional[KnowledgeType] = None
    ) -> Optional[ExternalKnowledge]:
        """
        ランダムに知識を1つ取得（セレンディピティ）
        
        Args:
            agent: エージェント
            knowledge_type: 種類で絞り込み（オプション）
        
        Returns:
            知識（見つからなければNone）
        """
        candidates = self.search(agent, knowledge_type=knowledge_type)
        
        if not candidates:
            return None
        
        return np.random.choice(candidates)


# ================================================================================
# 学習セッション
# ================================================================================

class LearningSession:
    """
    学習セッション（読書、受講、修行など）
    
    【理論的意義】
    外部知識へのアクセス → 意味圧の継続的投入 → κの形成
    
    教育の本質: 外部に固定された知識を、個人のκに転写すること
    """
    
    def __init__(self, agent: HumanAgent, knowledge: ExternalKnowledge):
        self.agent = agent
        self.knowledge = knowledge
        self.total_time = 0.0
        self.cumulative_understanding = 0.0
        self.session_count = 0
    
    def study(self, duration: float = 1.0, focus: float = 1.0) -> Dict:
        """
        学習を実施
        
        Args:
            duration: 学習時間
            focus: 集中度 [0.0, 1.0]
        
        Returns:
            学習結果の辞書
        """
        self.session_count += 1
        self.total_time += duration
        
        # 意味圧生成
        pressure = self.knowledge.generate_pressure(
            self.agent, 
            access_duration=duration,
            understanding_level=focus
        )
        
        # エージェントに適用
        self.agent.step(pressure=pressure, dt=duration)
        
        # 理解度の更新（κの成長を反映）
        current_kappa = self.agent.state.kappa[self.knowledge.target_layer.value]
        prerequisite = self.knowledge.prerequisite_kappa[self.knowledge.target_layer.value]
        understanding = min(1.0, current_kappa / prerequisite) if prerequisite > 0 else 1.0
        
        self.cumulative_understanding += understanding * duration
        
        return {
            "session": self.session_count,
            "total_time": self.total_time,
            "current_understanding": understanding,
            "average_understanding": self.cumulative_understanding / self.total_time,
            "kappa_growth": current_kappa,
            "pressure_applied": {
                "physical": pressure.physical,
                "base": pressure.base,
                "core": pressure.core,
                "upper": pressure.upper
            }
        }
    
    def get_summary(self) -> str:
        """学習セッションのサマリー"""
        avg_understanding = self.cumulative_understanding / self.total_time if self.total_time > 0 else 0
        
        return f"""
Learning Session Summary
========================
Knowledge: {self.knowledge.title}
Agent: {self.agent.agent_id}
Total Sessions: {self.session_count}
Total Time: {self.total_time:.1f}
Average Understanding: {avg_understanding:.2%}
Target Layer κ: {self.agent.state.kappa[self.knowledge.target_layer.value]:.3f}
"""


# ================================================================================
# プリセット知識の生成
# ================================================================================

def create_sample_library() -> KnowledgeLibrary:
    """
    サンプル知識ライブラリの生成
    
    Returns:
        知識が登録されたライブラリ
    """
    library = KnowledgeLibrary()
    
    # 1. 武術の型（身体技能）
    library.add_knowledge(ExternalKnowledge(
        title="太極拳の基本型",
        content="...",
        knowledge_type=KnowledgeType.PHYSICAL_SKILL,
        target_layer=HumanLayer.PHYSICAL,
        base_pressure=30.0,
        accessibility=0.6,
        prerequisite_kappa=np.array([1.0, 1.0, 1.0, 1.0]),
        complexity=0.4
    ))
    
    # 2. 料理本（実用ガイド）
    library.add_knowledge(ExternalKnowledge(
        title="基礎から学ぶフランス料理",
        content="...",
        knowledge_type=KnowledgeType.PRACTICAL_GUIDE,
        target_layer=HumanLayer.BASE,
        base_pressure=40.0,
        accessibility=0.8,
        prerequisite_kappa=np.array([1.0, 1.0, 1.0, 1.0]),
        complexity=0.3
    ))
    
    # 3. 法律書（文化規範）
    library.add_knowledge(ExternalKnowledge(
        title="民法入門",
        content="...",
        knowledge_type=KnowledgeType.CULTURAL_NORM,
        target_layer=HumanLayer.CORE,
        base_pressure=60.0,
        accessibility=0.7,
        prerequisite_kappa=np.array([1.0, 1.0, 1.2, 1.0]),
        complexity=0.6
    ))
    
    # 4. 哲学書（理論的知識）
    library.add_knowledge(ExternalKnowledge(
        title="カント『純粋理性批判』",
        content="...",
        knowledge_type=KnowledgeType.THEORETICAL,
        target_layer=HumanLayer.UPPER,
        base_pressure=80.0,
        accessibility=0.4,  # 難解で入手困難
        prerequisite_kappa=np.array([1.0, 1.0, 1.3, 1.5]),
        complexity=0.9
    ))
    
    # 5. 小説（物語）
    library.add_knowledge(ExternalKnowledge(
        title="ドストエフスキー『罪と罰』",
        content="...",
        knowledge_type=KnowledgeType.NARRATIVE,
        target_layer=HumanLayer.CORE,
        base_pressure=50.0,
        accessibility=0.8,
        prerequisite_kappa=np.array([1.0, 1.0, 1.1, 1.0]),
        complexity=0.5
    ))
    
    # 6. オンライン記事（デジタル時代）
    library.add_knowledge(ExternalKnowledge(
        title="Wikipedia: 構造主観力学",
        content="...",
        knowledge_type=KnowledgeType.THEORETICAL,
        target_layer=HumanLayer.UPPER,
        base_pressure=40.0,
        accessibility=0.95,  # 誰でもアクセス可能
        prerequisite_kappa=np.array([1.0, 1.0, 1.1, 1.2]),
        complexity=0.5
    ))
    
    return library


if __name__ == "__main__":
    # デモ: 学習セッション
    print("=== External Knowledge Access Demo ===\n")
    
    # ライブラリ作成
    library = create_sample_library()
    
    # エージェント作成
    from ssd_engine.core.ssd_human_module import HumanParams
    agent = HumanAgent(params=HumanParams(), agent_id="Student_A")
    
    print(f"初期状態: κ = {agent.state.kappa}\n")
    
    # 知識検索
    print("検索: 理論的知識")
    results = library.search(agent, knowledge_type=KnowledgeType.THEORETICAL)
    print(f"見つかった知識: {len(results)}件")
    for knowledge in results:
        print(f"  - {knowledge.title} (accessibility={knowledge.accessibility})")
    print()
    
    # 学習セッション開始
    if results:
        knowledge = results[0]
        print(f"学習開始: {knowledge.title}\n")
        
        session = LearningSession(agent, knowledge)
        
        # 10回学習
        for i in range(10):
            result = session.study(duration=1.0, focus=0.8)
            print(f"Session {result['session']}: "
                  f"理解度={result['current_understanding']:.2%}, "
                  f"κ={result['kappa_growth']:.3f}")
        
        print(session.get_summary())
