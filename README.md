# Zevihanthosa - Advanced Artificial Intelligence Framework
![ZevihaNut Logo](ZevihaNut.png)
## ZevihaNut/2.2 Model (Latest Release - December 2025)
*A lightweight, pure-Python, highly extensible neural computation framework combining parametric, non-parametric, and symbolic learning units with enhanced flexibility and serialization support.*

### Overview
**Zevihanthosa** is a minimalist yet powerful artificial intelligence framework designed for experimentation, education, and rapid prototyping of hybrid intelligent systems.

The **ZevihaNut/2.2** release (December 2025) introduces major enhancements:
- Unified **Activation** class with multiple activation functions (sigmoid, tanh, ReLU, linear, and custom "zevian" mapping).
- New cell types: **LinearSumCell**, **LinearMulCell**, and **MultiOutputCell**.
- Full **model serialization** via `save()`, `load()`, and `copy()` functions (JSON-based).
- Improved **FuncCell** using SymPy for safer and more robust symbolic evaluation.
- Perceptron-mode training option for compatible cells.
- Advanced input/output scaling utilities (`to01`, `from01`, `nom`, `denom`, `rmtolerance`).

All core cells remain fully online, incremental, and momentum-optimized. The framework requires only the Python standard library plus `numpy` and `sympy`.

### Key Features
- **Pure Python & Minimal Dependencies** (`random`, `math`, `json`, `numpy`, `sympy`)
- **Fully Online Learning** – adapts instantly per sample
- **Momentum Optimization** (default 0.9)
- **Automatic Weight/Bias Clamping**
- **Flexible Activations** – sigmoid, tanh, ReLU, linear, custom zevian
- **Model Persistence** – save/load entire cell states to/from JSON
- **Six Learning Paradigms**:
  1. Classic sigmoid neuron (`Cell`)
  2. Linear summation unit (`LinearSumCell`)
  3. Linear multiplication unit (`LinearMulCell`)
  4. Instance-based memory (`DataCell`)
  5. Multi-input fusion (`MultiInputCell`)
  6. Multi-output parallel processing (`MultiOutputCell`)
  7. Symbolic regression (`FuncCell`)
- **High Interpretability** – human-readable formulas from FuncCell
- **Robust Numerical Handling** – scaling helpers and SymPy evaluation

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
Generalizes `Cell` to arbitrary input vectors with shared activation.

#### 6. `MultiOutputCell` — Parallel Multi-Output Neuron
Single input → multiple independent outputs (useful for multi-task learning).

#### 7. `FuncCell` — Symbolic Regression Unit (Improved in 2.2)
- Safer evaluation via **SymPy**
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
No pip install required (manual install of `sympy` if needed: `pip install sympy`).

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

# 3. MultiInputCell – learning to approximate sum (formerly MultiCell)
mi = MultiInputCell(wcount=2, learning=0.08)
for _ in range(30000):
    a = random.randint(1, 10)
    b = random.randint(1, 10)
    mi.process([1/a, 1/b], target=1/(a + b)) # 1/x normalization for 0-1 range
print("MultiInputCell [7, 2] ≈", 1/mi.process([1/7, 1/2], train=False))

# 4. MultiOutputCell – simple example: predicting x² and √x from x
mo = MultiOutputCell(pcount=2, learning=0.1)
for _ in range(20000):
    x = random.uniform(0, 1)  # input in [0,1]
    target1 = x ** 2        # square
    target2 = math.sqrt(x)  # square root
    mo.process(x, target=[target1, target2])
print("x = 0.0  → prediction (x², √x):", mo.process(0.0, train=False), "  true: (0.0, 0.0)")
print("x = 0.25 → prediction (x², √x):", mo.process(0.25, train=False), "  true: (0.0625, 0.5)")
print("x = 0.5  → prediction (x², √x):", mo.process(0.5, train=False), "  true: (0.25, ~0.707)")
print("x = 1.0  → prediction (x², √x):", mo.process(1.0, train=False), "  true: (1.0, 1.0)")

# 5. FuncCell – discovering square root
fc = FuncCell(traindepth=2)
for x in range(5):
    y = math.sqrt(x)
    fc.process(x, target=y)
print("Discovered function:", fc.func)
print("Test on x=3 →", fc.process(3, train=False))
```
### Example Output:
```
Cell at 0.6 → 0.041278469501507695
Cell at 0.8 → 0.9721519183272612
Current smooth estimate: 0.5
MultiInputCell [7, 2] ≈ 9.589436088382946
x = 0.0  → prediction (x², √x): [0.011683315904739942, 0.2229527836890146]   true: (0.0, 0.0)
x = 0.25 → prediction (x², √x): [0.056061428408423056, 0.46143848469357274]   true: (0.0625, 0.5)
x = 0.5  → prediction (x², √x): [0.22980992363646982, 0.7189855465512454]   true: (0.25, ~0.707)
x = 1.0  → prediction (x², √x): [0.882784993707556, 0.9580093837535523]   true: (1.0, 1.0)
Discovered function: x**2**-1
Test on x=3 → 1.7320508075688772
```

### Why Zevihanthosa?
- **True Hybrid AI**: Neural + Linear + Memory + Symbolic + Multi-output
- **Maximum Transparency & Persistence**
- **Lightweight & Portable**: Runs anywhere Python runs
- **Educational Gold**: Clear, modular code
- **Highly Extensible**: Easy to add new cell types or activations

### Future Roadmap
- Network composition helpers
- Visualization tools
- Genetic programming integration for deeper symbolic search
- Optional GPU/back-end acceleration
- RL and evolutionary examples

### License
MIT License — completely free for any use.

---
**Zevihanthosa — Where simplicity, interpretability, persistence, and hybrid intelligence converge.**  
*ZevihaNut/2.2 — December 2025: Serialization, New Cells, Flexible Activations, SymPy Safety. Ready.*
