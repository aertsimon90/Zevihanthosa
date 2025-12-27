# Zevihanthosa - Advanced Artificial Intelligence Framework
![ZevihaNut Logo](ZevihaNut.png)
## ZevihaNut/2.3 Model (Latest Release - December 2025)
*A lightweight, pure-Python, highly extensible neural computation framework combining parametric, non-parametric, and symbolic learning units with enhanced flexibility and serialization support.*

### Overview
**Zevihanthosa** is a minimalist yet powerful artificial intelligence framework designed for experimentation, education, and rapid prototyping of hybrid intelligent systems.

The **ZevihaNut/2.3** release (December 2025) introduces major enhancements:
- Unified **Activation** class with multiple activation functions (sigmoid, tanh, ReLU, linear, and custom "zevian" mapping).
- New cell types: **LinearSumCell**, **LinearMulCell**, **MultiOutputCell**, and fully generalized **MultiCell** (multi-input to multi-output dense layer).
- Full **model serialization** via `save()`, `load()`, and `copy()` functions (JSON-based).
- Improved **FuncCell** for symbolic regression (bounded expression search).
- Perceptron-mode training option for compatible cells.
- Advanced input/output scaling utilities (`to01`, `from01`, `nom`, `denom`, `rmtolerance`).

All core cells remain fully online, incremental, and momentum-optimized. The framework requires only the Python standard library plus `numpy`.

### Key Features
- **Pure Python & Minimal Dependencies** (`random`, `math`, `json`, `numpy`)
- **Fully Online Learning** – adapts instantly per sample
- **Momentum Optimization** (default 0.9)
- **Automatic Weight/Bias Clamping**
- **Flexible Activations** – sigmoid, tanh, ReLU, linear, custom zevian
- **Model Persistence** – save/load entire cell states to/from JSON
- **Seven Learning Paradigms**:
  1. Classic sigmoid neuron (`Cell`)
  2. Linear summation unit (`LinearSumCell`)
  3. Linear multiplication unit (`LinearMulCell`)
  4. Instance-based memory (`DataCell`)
  5. Multi-input fusion (`MultiInputCell`)
  6. Multi-output parallel processing (`MultiOutputCell`)
  7. Full multi-input/multi-output dense layer (`MultiCell`)
  8. Symbolic regression (`FuncCell`)
- **High Interpretability** – human-readable formulas from FuncCell
- **Robust Numerical Handling** – scaling helpers

### Core Components
#### 1. `Cell` — Classic Momentum-Optimized Neuron
Single-input sigmoid (or other activation) unit with optional perceptron mode.

#### 2. `LinearSumCell` — Additive Linear Unit
Performs `output = activation(input + bias)` – useful for offsets and shifts.

#### 3. `LinearMulCell` — Multiplicative Linear Unit
Performs `output = activation(input * weight)` – ideal for scaling and gating.

#### 4. `DataCell` — Non-Parametric Memory-Based Regressor
Fixed-size memory with distance-weighted prediction and smooth adaptation.

#### 5. `MultiInputCell` — Multi-Input Fusion Neuron
Generalizes `Cell` to arbitrary input vectors → single output with shared activation.

#### 6. `MultiOutputCell` — Parallel Multi-Output Neuron
Single input → multiple independent outputs (useful for multi-task learning).

#### 7. `MultiCell` — Full Dense Layer Unit (New generalization in 2.3)
Arbitrary inputs → arbitrary outputs (multi-input to multi-output), fully trainable dense layer equivalent. Replaces limited previous versions for general matrix operations.

#### 8. `FuncCell` — Symbolic Regression Unit
- Configurable search depth and start level
- Bounded brute-force enumeration of expressions
- Falls back to best-found formula

### New Utilities
- `Activation` class – configurable per cell
- `save(obj)`, `load(data)`, `copy(obj)` – full serialization
- Scaling helpers: `to01`, `from01`, `nom`, `denom`, `rmtolerance`

### Installation
```bash
git clone https://github.com/aertsimon90/Zevihanthosa.git
cd Zevihanthosa
# Import zevihanthosa.py directly into your project
```
No pip install required (manual install of `numpy` if needed: `pip install numpy`).

### Usage Examples
```python
from zevihanthosa import *
import random
import math

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
```

### Example Output (approximate, seed-dependent):
```
Cell at 0.6 → 0.03936621863731655
Cell at 0.8 → 0.970928958839111
Current smooth estimate: 0.5
MultiInputCell approx 1/(7+2) from [1/7, 1/2]: 9.04576264032236
x = 0.0 → (x², √x): [0.01231124637701895, 0.22831731209469477]
x = 0.25 → (x², √x): [0.0588352495562964, 0.4679160762452959]
x = 0.5 → (x², √x): [0.23868551855471903, 0.7232844542459053]
x = 1.0 → (x², √x): [0.8874600387615176, 0.958491484693469]
Discovered function: x**2**-1
Test on x=3 → 1.7320508075688772
Input [0.0, 0.0] → [0.09205978587521543, 0.09212060577720313, 0.09217526319405217]
Input [0.3, 0.3] → [0.20459543479530345, 0.20464990031139138, 0.20469908123965996]
Input [0.5, 0.6] → [0.35831451151931704, 0.3583188240447237, 0.3583229471663282]
Input [1.0, 1.0] → [0.6930334838933574, 0.6929096945961872, 0.6927995870267764]
Input [0.2, 0.8] → [0.3227756106664054, 0.3228064972197608, 0.32283250075616854]
```

### Why Zevihanthosa?
- **True Hybrid AI**: Neural + Linear + Memory + Symbolic + Full Dense Layers
- **Maximum Transparency & Persistence**
- **Lightweight & Portable**: Runs anywhere Python runs
- **Educational Gold**: Clear, modular code
- **Highly Extensible**: Easy to add new cell types or activations

### License
MIT License — completely free for any use.

---
**Zevihanthosa — Where simplicity, interpretability, persistence, and hybrid intelligence converge.**
*ZevihaNut/2.3 — December 2025: MultiCell generalization, enhanced hybrid capabilities. Ready.*
