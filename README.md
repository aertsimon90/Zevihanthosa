# Zevihanthosa - Advanced Artificial Intelligence Framework
![ZevihaNut Logo](ZevihaNut.png)

## ZevihaNut/2.6 Model (Latest Release - January 2026)

*A lightweight, pure-Python hybrid AI framework combining parametric neurons, non-parametric memory, safe symbolic regression, localized ensembles, and **Deep Multi-Layer Networks** — with complete serialization and enhanced numerical stability.*

### Overview

**Zevihanthosa** is a minimalist, powerful, and highly extensible artificial intelligence framework designed for experimentation, education, and rapid prototyping of transparent, persistent, and hybrid intelligent systems.

The **ZevihaNut/2.6** release (January 2026) brings significant advancements:

* **New Deep Learning Engine**: `MultiCellForestNetwork` (MCFN) provides a localized full dense ensemble Deep Multi-Layer Backpropagation framework.
* **Modern Activations**: Added **Swish**, **Positive Linear**, and **Negative Linear** methods.
* **Precise Derivatives**: Custom derivation functions added for every activation type to ensure accurate backpropagation gradients.
* **Numerical Safety**: Robust overflow handling in activation functions to prevent `math.exp` errors.
* New **localized ensemble cells**: `CellForest` and `MultiCellForest` for smoother, more robust function approximation.
* Ultra-lightweight **NanoCell** (momentum-free perceptron).
* Secure **FuncCell** using AST-based expression evaluation (no unsafe `eval`).
* Enhanced numerical tools: quantization, softsigmoid, trainrate modulation, safer division.
* Full JSON serialization for **all 13 cell types** (0–12).
* Improved `DataCell` with better averaging and fallback logic.
* Weight initialization centered in `[-1, 1]` for faster convergence.

All cells support fully **online, incremental learning** with momentum and optional perceptron mode. Only minimal dependencies required.

### Key Features

* **Pure Python & Minimal Dependencies** (`random`, `math`, `json`, `ast`)
* **Fully Online Learning** – instant adaptation per sample.
* **Deep Hierarchical Learning** – full backpropagation through arbitrary layers.
* **Momentum Optimization** (default 0.9).
* **Automatic Weight/Bias Clamping**.
* **Flexible Activations** – sigmoid, softsign, tanh, Swish, Positive/Negative Linear, custom zevian.
* **Complete Model Persistence** – save/load/copy any cell or network to/from JSON.
* **13 Learning Paradigms**:

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
11. Localized full dense ensemble (`MultiCellForest`)
12. Deep Multi-Layer Backpropagation Network (`CellNetwork`)
13. **Localized Full Dense Ensemble Deep Multi-Layer Backpropagation Network** (`MultiCellForestNetwork`)

* **High Interpretability** – readable symbolic formulas and persistent states.
* **Robust Scaling & Quantization Tools**.

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

#### 10. `CellForest` — Distance-weighted ensemble of `Cell`s (localized learning)

#### 11. `MultiCellForest` — Ensemble of `MultiCell`s for robust multi-dimensional mapping

#### 12. `CellNetwork` — Multi-layer dense network with automated backpropagation

#### 13. `MultiCellForestNetwork` — MCFN combines the localized intelligence of `MultiCellForest` with the depth of `CellNetwork`.

### New in ZevihaNut/2.6

* **Swish Activation**: Implementation of `x * sigmoid(x)` normalized for the framework.
* **Linear Activation Variants**: `plinear` (Positive Linear) and `nlinear` (Negative Linear) for directional rectifying behavior.
* **Overflow Protection**: Activation functions now gracefully handle extreme values to maintain training stability.
* **Multi-Layer Forest**: `MultiCellForestNetwork` (MCFN) for high-dimensional, localized deep learning. (Currently, a backpropagation error has been discovered within this feature. If it cannot be fixed, this feature may be removed)
* **Enhanced Derivatives**: Analytical derivatives for all activations (`s`, `ss`, `t`, `l`, `pl`, `nl`, `sw`, `z`) to improve backpropagation accuracy.
* **Serialization Fix**: Resolved JSON serialization inconsistencies in MultiCell layers for reliable model persistence.

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
mcf = MultiCellForest(cellscount=64, icount=2, ocount=1, learning=0.08)
for _ in range(100000):
    x = random.random()
    y = random.random()
    target = (math.sin(x * 5) + math.cos(y * 5)) / 4 + 0.5  # simpler 2D oscillation (lower frequency, easier to learn)
    mcf.process([x, y], target=[target], distance_sharpness=32)
print("MultiCellForest at [0.1, 0.2] →", mcf.process([0.1, 0.2], train=False, distance_sharpness=32)[0]) # expected: 0.7549319611180857
print("MultiCellForest at [0.9, 0.8] →", mcf.process([0.9, 0.8], train=False, distance_sharpness=32)[0]) # expected: 0.09220656536782279
print("MultiCellForest at [0.5, 0.5] →", mcf.process([0.5, 0.5], train=False, distance_sharpness=32)[0]) # expected: 0.4493321321392557
print("MultiCellForest at [0.0, 1.0] →", mcf.process([0.0, 1.0], train=False, distance_sharpness=32)[0]) # expected: 0.5709155463658065

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

# Why MultiCellForestNetwork not have a example code? Currently, a backpropagation error has been discovered within this feature. If it cannot be fixed, this feature may be removed.

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
Input [0.0, 0.0] → [0.08736586764593907, 0.08733168916042566, 0.08730464548696337]
Input [0.3, 0.3] → [0.19890884494565178, 0.19886873872368668, 0.19883595155082934]
Input [0.5, 0.6] → [0.3542877079200685, 0.3542554262119176, 0.354226395777651]
Input [1.0, 1.0] → [0.6965192587019245, 0.696553361199306, 0.696575719364518]
Input [0.2, 0.8] → [0.3173584825325539, 0.31728153453270214, 0.3172141567520689]
CellForest sin-approx at x=0.1 → 0.9192891914216803
CellForest sin-approx at x=0.9 → 0.7069943727992968
MultiCellForest at [0.1, 0.2] → 0.7596576216489921
MultiCellForest at [0.9, 0.8] → 0.12025277413180832
MultiCellForest at [0.5, 0.5] → 0.4806804450111819
MultiCellForest at [0.0, 1.0] → 0.569294229076006
Training CellNetwork on Sine Wave...
--- Zevihanthosa CellNetwork: Sine Wave Accuracy Report ---
Angle:  0°   | Pred:  0.1127 | Real:  0.0000 | Error: 0.112727 | Acc: 94.36%
Angle:  45°  | Pred:  0.7041 | Real:  0.7071 | Error: 0.003009 | Acc: 99.85%
Angle:  90°  | Pred:  0.9719 | Real:  1.0000 | Error: 0.028108 | Acc: 98.59%
Angle: 180°  | Pred:  0.0287 | Real:  0.0000 | Error: 0.028675 | Acc: 98.57%
Angle: 270°  | Pred: -0.9440 | Real: -1.0000 | Error: 0.056022 | Acc: 97.20%
Angle: 360°  | Pred: -0.0792 | Real: -0.0000 | Error: 0.079180 | Acc: 96.04%
Overall CellNetwork Model Consistency: 97.44%
```

### Why Zevihanthosa?

* **True Hybrid Intelligence**: Neural + Linear + Memory + Symbolic + Localized Ensembles + Deep Networks.
* **Maximum Transparency, Persistence & Interpretability**.
* **Ultra Lightweight & Fully Portable** (Zero external binary dependencies).
* **Ideal for Education, Research, and Embedded AI**.
* **Extremely Modular** – easy to extend with new cell types.

### License

MIT License — free for any use, commercial or personal.

---

**Zevihanthosa — Simplicity meets persistent, interpretable, hybrid intelligence.**
*ZevihaNut/2.6 — January 2026: Deep networks, localized ensembles, and symbolic maturity. Ready.*
