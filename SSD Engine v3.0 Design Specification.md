# **SSD Engine v3.0 Design Specification**

## **Bio-Physical Separation Model (物理層と神経制御層の分離)**

作成日: 2025年12月13日  
バージョン: 3.0 Draft  
ステータス: 提案段階 (Request for Comments)

## **1\. 概要と目的**

### **1.1 現状の課題 (v2.x)**

現在の ssd_engine (v2.0) では、物理演算ロジック (SSDCoreEngine) と神経変調ロジック (ssd_neuro_modulators) が密結合している。神経変調は物理パラメータ（gamma, beta, Theta 等）を直接書き換えることで実装されており、以下の課題がある。

* **物理法則と個体特性の混同**: 普遍的であるべき「エネルギー保存則」のパラメータと、可変であるべき「性格」が混在している。  
* **説明性の欠如**: 「なぜ跳躍したか」を説明する際、「パラメータ $\gamma$ が高かったから」という物理的説明と、「ドーパミンが出ていたから」という心理的説明が分離できない。  
* **拡張の困難さ**: 新しい性格モデルや病理モデルを追加する際、コアエンジンのパラメータ調整が複雑になる。

### **1.2 v3.0 の目的**

数理モデルを\*\*「物理層（不変の法則）」**と**「制御層（可変の特性）」\*\*の二層に明確に分離する。これにより、モデルの透明性、拡張性、およびシミュレーションのリアリティ（うつ病やパニックの力学的再現など）を飛躍的に向上させる。

## **2\. アーキテクチャ設計**

### **2.1 レイヤー構造**

| レイヤー | 名称 | 役割 | 性質 | 主要変数 |
| :---- | :---- | :---- | :---- | :---- |
| **Layer 1** | **物理層** (Physics Layer) | **「逃れられない法則」の実行** エネルギー保存、相転移確率の計算 | **普遍・不変・ステートレス** （誰にでも平等に働く物理エンジン） | $E$ (熱) $P_{leap}$ (跳躍確率) $j$ (整合流) |
| **Layer 2** | **制御層** (Control Layer) | **「個体差と状態」の管理** 神経物質による物理パラメータの動的調整 | **固有・可変・ステートフル** （性格や気分、病理を定義） | $T$ (探索温度) $\alpha_{in}$ (入力感度) $\Delta F$ (障壁認識) |

### **2.2 ディレクトリ構造案**

ssd_engine/  
├── core/  
│   ├── physics/              <-- 【新設】物理層: 純粋な計算ロジック  
│   │   ├── __init__.py  
│   │   ├── energy_law.py     # エネルギー保存則 (dE/dt)  
│   │   └── phase_transition.py # 相転移・跳躍確率 (Arrhenius eq)  
│   │  
│   ├── controller/           <-- 【新設】制御層: 生物学的チューニング  
│   │   ├── __init__.py  
│   │   ├── neuro_tuner.py    # 神経物質によるパラメータ変調器  
│   │   ├── personality.py    # 性格特性プロファイル  
│   │   └── pathology.py      # 病理モデル (うつ、躁など)  
│   │  
│   └── agent/  
│       └── biophysical_agent.py # 統合エージェント（Physics + Controller）

## **3\. 数理モデル詳細**

### **3.1 第1層：物理層 (Physics Layer)**

この層には「性格」や「感情」は存在しない。存在するのはエネルギー保存則と確率論だけである。

#### **3.1.1 エネルギー保存則（熱の蓄積）**

処理しきれなかった仕事（残差）は、必ず内部エネルギー（熱）として保存される。

$$ \frac{dE}{dt} = \alpha_{\text{in}} \cdot |p| - |j| - \beta \cdot E $$

* $E$: 未処理圧（熱）。システムの不安定化エネルギー。  
* $\|p\|$: 入力意味圧のノルム。  
* $\|j\|$: 整合流（出力）のノルム。  
* $\alpha_{\text{in}}$: 入力効率（物理的な受容能力）。**制御層から供給**。  
* $\beta$: 放熱係数（物理的な減衰）。**制御層から供給**。

#### **3.1.2 相転移の法則（確率的跳躍）**

構造が変化（跳躍）する確率は、障壁の高さと、その場の「温度」のみによって決定される（アレニウスの式的アプローチ）。

$$ P(\text{Leap}) \propto \exp\left( -\frac{\Delta F}{T} \right) \cdot dt $$

* $\Delta F$: 跳躍に必要なエネルギー障壁（構造の硬さ）。**制御層から供給**。  
* $T$: 探索温度。分子（選択肢）の運動エネルギー。**制御層から供給**。

### **3.2 第2層：制御層 (Neuro Controller)**

「心」「性格」「病理」はすべてこの層に記述される。神経物質は物理層のパラメータを動的に書き換える「変調器（Modulator）」として機能する。

#### **3.2.1 神経物質定義**

* **Dopamine (DA)**: 駆動、報酬予測。$T$ を上昇させる。  
* **Serotonin (5HT)**: 抑制、安定化。$T$ の過剰上昇を抑え、$\beta$（放熱）を支える。  
* **Noradrenaline (NA)**: 覚醒、危機検知。$\alpha_{in}$（入力感度）を増幅させる。  
* **Endorphin (End)**: 鎮痛、緩衝。$\Delta F$（障壁）の見かけの高さを下げる。

#### **3.2.2 パラメータ決定ロジック**

1\. 探索温度 ($T$) の決定  
熱 $E$ を運動エネルギー $T$ に変換する効率は、ドーパミンとセロトニンのバランスで決まる。  
$$ T(t) = \frac{\text{Dopamine}(t)}{\text{Serotonin}(t) + \epsilon} \cdot E(t) $$

* **うつ状態の再現**: $E$（苦しみ）が大きくても、DAが枯渇していると $T \\approx 0$ となり、動けない（跳躍確率ゼロ）。  
* **躁状態の再現**: $E$ が小さくても、DA過剰なら $T$ が暴走し、無秩序な跳躍を起こす。

2\. 入力感度 ($\\alpha\_{\\text{in}}$) の決定  
危機的状況では感度が上がり、些細な刺激も巨大なエネルギーとして取り込む。  
$$ \alpha_{\text{in}}(t) = \text{BaseSensitivity} \times (1.0 + k_{na} \cdot \text{NorAdrenaline}(t)) $$

3\. 障壁認識 ($\\Delta F$) の決定  
エンドルフィンは「痛み（障壁）」を麻痺させ、主観的なハードルを下げる。  
$$ \Delta F(t) = \frac{\text{PhysicalBarrier}}{1.0 + k_{end} \cdot \text{Endorphin}(t)} $$

## **4\. 実装イメージ (Python Draft)**

### **4.1 物理層 (core/physics/energy\_law.py)**

def update_thermal_energy(E: float, p_norm: float, j_norm: float,   
                          alpha_in: float, beta: float, dt: float) -> float:  
    """  
    熱力学第一法則（エネルギー保存）のSSD版  
    dE = (入力仕事) - (出力仕事) - (散逸)  
    """  
    # 入力仕事 (Input Work): 受容した意味圧  
    w_in = alpha_in * p_norm  
      
    # 出力仕事 (Output Work): 整合によって処理された量  
    w_out = j_norm  
      
    # 散逸 (Dissipation): 自然放熱  
    dissipation = beta * E  
      
    dE = w_in - w_out - dissipation  
    return max(0.0, E + dE * dt)

### **4.2 制御層 (core/controller/neuro\_tuner.py)**

@dataclass  
class NeuroState:  
    dopamine: float = 0.5  
    serotonin: float = 0.5  
    noradrenaline: float = 0.5  
    endorphin: float = 0.5

@dataclass  
class PhysicsParams:  
    alpha_in: float  
    beta: float  
    temperature_T: float  
    barrier_F: float

class NeuroTuner:  
    def tune(self, neuro: NeuroState, current_E: float, base_barrier: float) -> PhysicsParams:  
        # 1. 探索温度 T  
        # セロトニンが低いと、わずかなドーパミンでもTが跳ね上がる（衝動性）  
        conversion_efficiency = neuro.dopamine / (neuro.serotonin + 1e-6)  
        T = conversion_efficiency * current_E + 0.01  
          
        # 2. 入力感度 alpha_in  
        # ノルアドレナリンによるゲイン増幅  
        alpha_in = 1.0 + 2.0 * neuro.noradrenaline  
          
        # 3. 障壁認識 F  
        # エンドルフィンによる鎮痛  
        barrier_F = base_barrier / (1.0 + 2.0 * neuro.endorphin)  
          
        # 4. 放熱係数 beta  
        # セロトニンによる安定化  
        beta = 0.1 * (1.0 + neuro.serotonin)  
          
        return PhysicsParams(alpha_in, beta, T, barrier_F)

### **4.3 統合エージェント (core/agent/biophysical\_agent.py)**

class BioPhysicalAgent:  
    def __init__(self):  
        self.E = 0.0  
        self.neuro_state = NeuroState()  
        self.tuner = NeuroTuner()  
          
    def step(self, pressure: float, response: float, dt: float = 0.1):  
        # 1. 制御層：物理パラメータの決定  
        params = self.tuner.tune(self.neuro_state, self.E, base_barrier=100.0)  
          
        # 2. 物理層：エネルギー更新  
        self.E = update_thermal_energy(  
            self.E, abs(pressure), abs(response),  
            params.alpha_in, params.beta, dt  
        )  
          
        # 3. 物理層：跳躍判定  
        leap_prob = compute_leap_probability(  
            params.barrier_F, params.temperature_T, dt  
        )  
          
        is_leaping = np.random.random() < leap_prob  
        return is_leaping, self.E, params

## **5\. 期待される効果**

1. **病理モデルの自然な表現**:  
   * うつ（DA/5HT枯渇）、パニック（NA過剰）、依存症（DA過剰・報酬系異常）などを、個別のロジックではなく**神経伝達物質のバランス**だけで表現可能になる。  
2. **シミュレーションの安定性**:  
   * 物理層が「エネルギー保存則」を厳密に守るため、パラメータ設定ミスによるエネルギーの発散や消失（永久機関化）を防ぎやすくなる。  
3. **デバッグと説明**:  
   * 「なぜエージェントが暴走したのか？」に対し、「NA過剰で入力を過大評価し、かつ5HT不足で放熱が追いつかなかったため」といった**医学的・力学的説明**が可能になる。