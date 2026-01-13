# Zevihanthosa - Advanced Artificial Intelligence Framework
![ZevihaNut Logo](ZevihaNut.png)

## ZevihaNut/2.8 Model (Latest Release - January 2026)
*A lightweight, pure-Python hybrid AI framework combining parametric neurons, non-parametric memory, safe symbolic regression, massively scaled localized ensembles, and **Deep Multi-Layer Networks** — with complete serialization, enhanced numerical stability, aggressive input compression, and dramatically expanded expert pools.*

### Overview
**Zevihanthosa** is a minimalist, powerful, and highly extensible artificial intelligence framework designed for experimentation, education, and rapid prototyping of transparent, persistent, and hybrid intelligent systems.

The **ZevihaNut/2.8** release (January 2026) introduces a major shift toward **massively parallel localized specialization** with the following key advancements:

- **Aggressive Input Compression** — New `squeezeforclust` mechanism for extremely compact multi-dimensional coordinate encoding  
- **Massive Expert Scaling** — `MultiCellForest` now creates `cellscount ** input_count` cells, `CellNetworkForest` uses `networkcount ** input_layer_size` networks  
- **Anchor-based Ensemble Blending** — The most relevant (best) unit/network now has significantly stronger influence on the final output  
- **Center-aware Localization** — Floating-point indexing with cell/network center offset `(i+0.5)` for smoother relevance gradients  
- **Cluster-aware Routing** — `clustmin`/`clustmax` + `squeezeforclust` for sharper and more stable multi-dimensional target selection  
- Retained all stability & flexibility improvements from 2.7

All cells continue to support fully **online, incremental learning** with momentum and optional perceptron mode. Only minimal dependencies required.

### Key Features
- **Pure Python & Minimal Dependencies** (`random`, `math`, `json`, `ast`)
- **Fully Online Learning** – instant adaptation per sample
- **Deep Hierarchical Learning** – full backpropagation through arbitrary layers
- **Momentum Optimization** (default 0.9)
- **Automatic Weight/Bias Clamping**
- **Flexible Activations** – sigmoid, softsign, tanh, Swish, Positive/Negative Linear, custom zevian
- **Complete Model Persistence** – save/load/copy any cell or network to/from JSON
- **13 Learning Paradigms**:
  1. Classic momentum-optimized neuron (`Cell`)
  2. Additive linear unit (`LinearSumCell`)
  3. Multiplicative linear unit (`LinearMulCell`)
  4. Instance-based memory regressor (`DataCell`)
  5. Multi-input fusion neuron (`MultiInputCell`)
  6. Parallel multi-output neuron (`MultiOutputCell`)
  7. Full dense layer (multi-in → multi-out) (`MultiCell`)
  8. Symbolic regression with safe AST evaluation (`FuncCell`)
  9. Minimal perceptron without momentum (`NanoCell`)
  10. Localized single-input ensemble (`CellForest`)
  11. **Massively scaled localized full dense ensemble** (`MultiCellForest`)
  12. Deep Multi-Layer Backpropagation Network (`CellNetwork`)
  13. **Massively scaled localized ensemble of deep multi-layer networks** (`CellNetworkForest`)

- **High Interpretability** – readable symbolic formulas and persistent states
- **Robust Scaling & Aggressive Quantization/Compression Tools**

### Core Components
#### 1. `Cell` — Classic Momentum-Optimized Neuron
#### 2. `LinearSumCell` — `output = activation(input + bias)`
#### 3. `LinearMulCell` — `output = activation(input * weight)`
#### 4. `DataCell` — Distance-weighted memory-based prediction
#### 5. `MultiInputCell` — Multiple inputs → single output
#### 6. `MultiOutputCell` — Single input → multiple outputs
#### 7. `MultiCell` — Full arbitrary dense layer
#### 8. `FuncCell` — Safe symbolic regression (bounded arithmetic expressions)
#### 9. `NanoCell` — Lightweight perceptron (no momentum)
#### 10. `CellForest` — Distance-weighted ensemble of `Cell`s with center-aware localization & top_k selection
#### 11. `MultiCellForest` — **Massive** ensemble of `MultiCell`s (cellscount**icount experts), cluster-compressed routing, anchor blending
#### 12. `CellNetwork` — Multi-layer dense network with automated backpropagation and per-layer activations
#### 13. `CellNetworkForest` — **Massive** ensemble of `CellNetwork`s (networkcount**input_size experts), cluster-compressed multi-dimensional routing, anchor blending

### New in ZevihaNut/2.8
- **squeezeforclust** — New aggressive multi-dimensional input compression & quantization method for forest routing
- **Massive Expert Pool Scaling**  
  • `MultiCellForest`: now spawns `cellscount ** input_count` independent `MultiCell` instances  
  • `CellNetworkForest`: now spawns `networkcount ** first_layer_size` complete deep networks
- **Anchor Blending** — Most relevant expert/network (highest relevance score) now acts as a strong "anchor" and receives extra weight in final output mixing
- **Center-aware Index Calculation** — `(i + 0.5)` centering in relevance computation → smoother transitions between experts
- **clustmin / clustmax** — Specialized min/max-aware clamping for compression-aware quantization
- **Improved Forest Routing** — All multi-dimensional forests now use `squeezeforclust` instead of simple averaging → better handling of structured multi-feature inputs
- **Top-K + Anchor Synergy** — Combination of limited top-k selection + strong anchor influence improves both efficiency and output quality

> [!NOTE]  
> **2.8 Philosophy**  
> ZevihaNut/2.8 moves decisively toward the "thousands of highly specialized experts" paradigm.  
> Instead of trying to make one forest better at everything, we massively increase the number of experts and use very aggressive compression + anchor blending to route inputs to the most relevant specialists.  
> Early tests show significantly sharper local approximation behavior, especially on functions with many different regimes.

> [!WARNING]  
> **Current Trade-offs & Limitations in 2.8**  
> 
> - Memory consumption grows **exponentially** with input dimensionality  
>   → `MultiCellForest` with 64 cells and 4 inputs = **16 million** cells (!)
> - Training time increases dramatically with expert count  
> - Still sensitive to proper tuning of `distance_sharpness`, `top_k`, and `clusters` parameters  
> - Multi-dimensional routing quality, while improved, remains heuristic-based  
> 
> **Recommended usage patterns (2.8):**
> - Single-input or very low-dimensional problems → full power of massive scaling  
> - Medium dimensionality (2–4 inputs) → carefully tune `cellscount`/`networkcount` (16–64 range most practical)  
> - High-dimensional problems → consider dimensionality reduction first or wait for future hierarchical routing improvements
> 
> We continue working toward memory-efficient, adaptive, and truly high-dimensional forest architectures.

Thank you for following Zevihanthosa development — 2.8 represents a bold step toward extreme specialization.

---

### Installation

```bash
git clone https://github.com/aertsimon90/Zevihanthosa.git
cd Zevihanthosa
# Import zevihanthosa.py directly into your project

```

No pip install required.

### Usage Examples

```python
from zevihanthosa import *
import random
import math

random.seed(42) # stability

# 1. Cell – learning a threshold
cell = Cell(learning=0.15)
for _ in range(12000):
    x = random.random()
    target = 1.0 if x > 0.7 else 0.0
    cell.process(x, target)
print("Cell at 0.6 →", cell.process(0.6, train=False))
print("Cell at 0.8 →", cell.process(0.8, train=False))

# 2. DataCell – adaptive averaging / denoising
dc = DataCell(maxdatac=100)
for i in range(300):
    noisy = 0.5 + random.gauss(0, 0.12)
    dc.process(noisy, target=0.5)
print("Current smooth estimate:", dc.process(0.7, train=False))

# 3. MultiInputCell – learning harmonic mean approximation
mi = MultiInputCell(wcount=2, learning=0.08)
for _ in range(30000):
    a = random.randint(1, 10)
    b = random.randint(1, 10)
    mi.process([1/a, 1/b], target=1/(a + b))  # 1/x normalization for 0-1 range
print("MultiInputCell approx 1/(7+2) from [1/7, 1/2]:", 1/mi.process([1/7, 1/2], train=False))

# 4. MultiOutputCell – predicting x² and √x from x
mo = MultiOutputCell(pcount=2, learning=0.1)
for _ in range(20000):
    x = random.uniform(0, 1)
    target1 = x ** 2
    target2 = math.sqrt(x)
    mo.process(x, target=[target1, target2])
print("x = 0.0 → (x², √x):", mo.process(0.0, train=False))
print("x = 0.25 → (x², √x):", mo.process(0.25, train=False))
print("x = 0.5 → (x², √x):", mo.process(0.5, train=False))
print("x = 1.0 → (x², √x):", mo.process(1.0, train=False))

# 5. FuncCell – discovering square root
fc = FuncCell(traindepth=2)
for x in [0,1,4,9,16]:
    y = math.sqrt(x)
    fc.process(x, target=y)
print("Discovered function:", fc.func)
print("Test on x=3 →", fc.process(3, train=False))

# 6. MultiCell – dividing the sum of two numbers by three (equal share to each output) test (new in 2.3)
mc = MultiCell(icount=2, ocount=3, learning=0.15)
for _ in range(30000):
    a = random.uniform(0.0, 1.0)
    b = random.uniform(0.0, 1.0)
    total = a + b
    third = total / 3.0
    mc.process([a, b], target=[third, third, third])
print("Input [0.0, 0.0] →", mc.process([0.0, 0.0], train=False))      # expected: [0.0, 0.0, 0.0]
print("Input [0.3, 0.3] →", mc.process([0.3, 0.3], train=False))      # expected: [0.2, 0.2, 0.2]
print("Input [0.5, 0.6] →", mc.process([0.5, 0.6], train=False))      # expected: ≈[0.3667, 0.3667, 0.3667]
print("Input [1.0, 1.0] →", mc.process([1.0, 1.0], train=False))      # expected: ≈[0.6667, 0.6667, 0.6667]
print("Input [0.2, 0.8] →", mc.process([0.2, 0.8], train=False))      # expected: ≈[0.3333, 0.3333, 0.3333]

# 7. New: CellForest – smooth approximation of complex 1D function
cf = CellForest(cellscount=64, learning=0.12)
for _ in range(25000):
    x = random.random()
    y = math.sin(x * 10) / 2 + 0.5  # fast oscillating wave
    cf.process(x, target=y, distance_sharpness=32)
print("CellForest sin-approx at x=0.1 →", cf.process(0.1, train=False, distance_sharpness=32)) # expected: 0.9207354924039483
print("CellForest sin-approx at x=0.9 →", cf.process(0.9, train=False, distance_sharpness=32)) # expected: 0.7060592426208783

# 8. MultiCellForest – robust approximation of complex 2D function (simplified for better convergence)
mcf = MultiCellForest(cellscount=32, icount=2, ocount=1, learning=0.08)
for i in range(30000):
    x = random.random()
    y = random.random()
    target = (math.sin(x * 5) + math.cos(y * 5)) / 4 + 0.5  # simpler 2D oscillation (lower frequency, easier to learn)
    mcf.process([x, y], target=[target], distance_sharpness=16, top_k=2)
print("MultiCellForest at [0.1, 0.2] →", mcf.process([0.1, 0.2], train=False, distance_sharpness=16, top_k=2)[0]) # expected: 0.7549319611180857
print("MultiCellForest at [0.9, 0.8] →", mcf.process([0.9, 0.8], train=False, distance_sharpness=16, top_k=2)[0]) # expected: 0.09220656536782279
print("MultiCellForest at [0.5, 0.5] →", mcf.process([0.5, 0.5], train=False, distance_sharpness=16, top_k=2)[0]) # expected: 0.4493321321392557
print("MultiCellForest at [0.0, 1.0] →", mcf.process([0.0, 1.0], train=False, distance_sharpness=16, top_k=2)[0]) # expected: 0.5709155463658065

# 9. CellNetwork – Learning a Continuous Sine Wave Mapping
# Architecture: 1 input (x) -> 16 hidden neurons -> 1 output (sin(x))
# We use more hidden neurons to capture the curvature of the wave.
net_sine = CellNetwork(layers=[1, 16, 1], learning=0.05)

print("Training CellNetwork on Sine Wave...")
for i in range(150000):
    # Training range: 0 to 2*PI, normalized to 0-1 for stability
    x_val = random.random() 
    # Target: sin(x) mapped from [-1, 1] to [0, 1] range for neural stability
    target_sin = (math.sin(x_val * math.pi * 2) + 1.0) / 2.0
    net_sine.process([x_val], target=[target_sin])

# --- Comprehensive Sine Wave Inference & Accuracy Report ---
test_points = [
    (0.00, "0°"),      # Expected: 0
    (0.125, "45°"),    # Expected: 0.707
    (0.25, "90°"),     # Expected: 1
    (0.50, "180°"),    # Expected: 0
    (0.75, "270°"),    # Expected: -1
    (1.00, "360°")     # Expected: 0
]

print("--- Zevihanthosa CellNetwork: Sine Wave Accuracy Report ---")
total_acc = 0
for x_norm, label in test_points:
    # Get raw prediction [0, 1]
    raw_out = net_sine.process([x_norm], train=False)[0]
    # Map back to [-1, 1]
    predicted_val = (raw_out * 2.0) - 1.0
    # Calculate real value
    actual_val = math.sin(x_norm * math.pi * 2)
    # Calculate error and accuracy percentage
    error = abs(actual_val - predicted_val)
    # Accuracy: 100% minus the error percentage (relative to range 2.0)
    accuracy = max(0, (1.0 - (error / 2.0)) * 100)
    total_acc += accuracy
    
    print(f"Angle: {label:^5} | Pred: {predicted_val:7.4f} | Real: {actual_val:7.4f} | Error: {error:.6f} | Acc: {accuracy:.2f}%")

print(f"Overall CellNetwork Model Consistency: {(total_acc / len(test_points)):.2f}%")

# 10. CellNetworkForest – Ensemble of deep networks for robust 2D function approximation
# (new in 2.7 – stabilized deep forest architecture, handling multi-dimensional inputs)
# Task: Learn sin(2πx) * cos(2πy) + noise (harder, non-linear 2D interaction like modulated waves)
cnf = CellNetworkForest(
    networkcount=24, 
    layers=[2, 12, 1],  # 2 inputs (x,y) → 12 hidden → 1 output
    learning=0.04,
)

print("Training CellNetworkForest (CNF) on Noisy 2D Modulated Wave...")
# Training data: complex 2D surface with gaussian noise for challenge
for i in range(12000):
    x_val = random.random()
    y_val = random.random()
    # Target: sin(2πx) * cos(2πy) – non-separable, oscillatory 2D function
    raw_target = math.sin(x_val * math.pi * 2) * math.cos(y_val * math.pi * 2)
    # Add noise to simulate real-world data imperfection
    noisy_raw = raw_target + random.gauss(0, 0.1)
    # Normalize to [0,1] for framework stability
    target_val = max(0.0, min(1.0, (noisy_raw + 1.0) / 2.0))
    cnf.process([x_val, y_val], target=[target_val], distance_sharpness=48)

# --- Comprehensive 2D Modulated Wave Inference & Accuracy Report (comparison) ---
# Test points chosen for diverse regions: peaks, troughs, and interactions
test_points = [
    ([0.00, 0.00], "(0,0)"),     # Expected raw: sin(0)*cos(0) = 0*1 = 0 → norm ~0.5
    ([0.25, 0.00], "(90°,0°)"),  # Expected: 1*1 = 1 → norm 1.0
    ([0.50, 0.25], "(180°,90°)"),# Expected: 0*0 = 0 → norm 0.5
    ([0.75, 0.50], "(270°,180°)"),# Expected: (-1)*(-1) = 1 → norm 1.0
    ([0.00, 0.75], "(0°,270°)"), # Expected: 0*0 = 0 → norm 0.5
    ([1.00, 1.00], "(360°,360°)"),# Expected: 0*1 = 0 → norm 0.5
]

print("--- Zevihanthosa CellNetworkForest (CNF – localized ensemble network) Accuracy ---")
total_acc_cnf = 0
for inputs, label in test_points:
    raw_out = cnf.process(inputs, train=False, distance_sharpness=48)[0]
    predicted_raw = (raw_out * 2.0) - 1.0  # Denormalize
    actual_raw = math.sin(inputs[0] * math.pi * 2) * math.cos(inputs[1] * math.pi * 2)
    error = abs(actual_raw - predicted_raw)
    accuracy = max(0, (1.0 - (error / 2.0)) * 100)
    total_acc_cnf += accuracy
    print(f"Point: {label:^10} | Pred: {predicted_raw:7.4f} | Real: {actual_raw:7.4f} | Error: {error:.6f} | Acc: {accuracy:.2f}%")

print(f"Overall CNF (Network Forest) Consistency: {total_acc_cnf / len(test_points):.2f}%")
```

### Example Output

```
Cell at 0.6 → 0.032449743937126636
Cell at 0.8 → 0.964736878960268
Current smooth estimate: 0.5
MultiInputCell approx 1/(7+2) from [1/7, 1/2]: 9.154347961916285
x = 0.0 → (x², √x): [0.012386497402471207, 0.2281075329679371]
x = 0.25 → (x², √x): [0.057940683286017716, 0.4640935052228233]
x = 0.5 → (x², √x): [0.23172214712064113, 0.7173350612656512]
x = 1.0 → (x², √x): [0.8788366685322571, 0.9561269185033562]
Discovered function: x**2**-1
Test on x=3 → 1.7320508075688772
Input [0.0, 0.0] → [0.08764712160414166, 0.08764712160414166, 0.08764712160414166]
Input [0.3, 0.3] → [0.19918553864873334, 0.19918553864873334, 0.19918553864873334]
Input [0.5, 0.6] → [0.35441220242318905, 0.35441220242318905, 0.35441220242318905]
Input [1.0, 1.0] → [0.6960045996154952, 0.6960045996154952, 0.6960045996154952]
Input [0.2, 0.8] → [0.31786344456514853, 0.31786344456514853, 0.31786344456514853]
CellForest sin-approx at x=0.1 → 0.9072199230444465
CellForest sin-approx at x=0.9 → 0.6906707460269892
MultiCellForest at [0.1, 0.2] → 0.7939261980651235
MultiCellForest at [0.9, 0.8] → 0.08501925302360366
MultiCellForest at [0.5, 0.5] → 0.38642050241327847
MultiCellForest at [0.0, 1.0] → 0.55784065747824
Training CellNetwork on Sine Wave...
--- Zevihanthosa CellNetwork: Sine Wave Accuracy Report ---
Angle:  0°   | Pred:  0.0919 | Real:  0.0000 | Error: 0.091869 | Acc: 95.41%
Angle:  45°  | Pred:  0.7403 | Real:  0.7071 | Error: 0.033185 | Acc: 98.34%
Angle:  90°  | Pred:  0.9430 | Real:  1.0000 | Error: 0.057000 | Acc: 97.15%
Angle: 180°  | Pred: -0.0057 | Real:  0.0000 | Error: 0.005684 | Acc: 99.72%
Angle: 270°  | Pred: -0.9686 | Real: -1.0000 | Error: 0.031372 | Acc: 98.43%
Angle: 360°  | Pred: -0.1237 | Real: -0.0000 | Error: 0.123668 | Acc: 93.82%
Overall CellNetwork Model Consistency: 97.14%
Training CellNetworkForest (CNF) on Noisy 2D Modulated Wave...
--- Zevihanthosa CellNetworkForest (CNF – localized ensemble network) Accuracy ---
Point:   (0,0)    | Pred:  0.1379 | Real:  0.0000 | Error: 0.137930 | Acc: 93.10%
Point:  (90°,0°)  | Pred:  0.8302 | Real:  1.0000 | Error: 0.169780 | Acc: 91.51%
Point: (180°,90°) | Pred: -0.0183 | Real:  0.0000 | Error: 0.018299 | Acc: 99.09%
Point: (270°,180°) | Pred:  0.8330 | Real:  1.0000 | Error: 0.167007 | Acc: 91.65%
Point: (0°,270°)  | Pred: -0.0025 | Real: -0.0000 | Error: 0.002482 | Acc: 99.88%
Point: (360°,360°) | Pred:  0.1311 | Real: -0.0000 | Error: 0.131085 | Acc: 93.45%
Overall CNF (Network Forest) Consistency: 94.78%
```

## Why Zevihanthosa?
* **True Hybrid Intelligence**: Neural + Linear + Memory + Symbolic + Massively Scaled Localized Ensembles + Deep Networks + Multi-Layer Networks.
* **Maximum Transparency, Persistence & Interpretability**.
* **Ultra Lightweight & Fully Portable** (Zero external binary dependencies).
* **Ideal for Education, Research, Embedded AI, and Extreme Specialization Experiments**.
* **Extremely Modular** – easy to extend with new cell types.

### License
MIT License — free for any use, commercial or personal.

---
**Zevihanthosa — Simplicity meets persistent, interpretable, hybrid intelligence.**

*ZevihaNut/2.8 — January 2026: Massive expert scaling, aggressive input compression, anchor blending, and thousands of hyper-specialized units. Specialization taken to the extreme. Ready.*
