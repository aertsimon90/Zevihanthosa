# Zevihanthosa - Advanced Artificial Intelligence Framework  
## ZevihaNut/2.1 Model (Latest Release - December 2025)

*A lightweight, pure-Python, highly extensible neural computation framework combining parametric, non-parametric, and symbolic learning units.*

### Overview

**Zevihanthosa** is a minimalist yet powerful artificial intelligence framework designed for experimentation, education, and rapid prototyping of hybrid intelligent systems.

The **ZevihaNut/2.1** release (December 2025) delivers a significantly improved and more practical **FuncCell** — the symbolic regression unit — with smarter expression generation, reduced redundancy, and better performance, while preserving full interpretability.

All core cells remain fully online, incremental, and momentum-optimized. The framework requires only the Python standard library plus `numpy` (used solely in FuncCell for controlled numeric sampling in earlier versions; now simplified).

ZevihaNut/2.1 includes critical bug fixes (e.g., MultiCell limitation), refined symbolic search logic, and hardened numerical stability.

### Key Features

- **Pure Python & Minimal Dependencies** (`random`, `math`, `numpy` only in legacy FuncCell logic)
- **Fully Online Learning** – every cell adapts instantly per sample
- **Momentum Optimization** (default 0.9) for smooth convergence
- **Automatic Weight/Bias Clamping** to prevent explosion
- **Four Distinct Learning Paradigms**:
  1. Gradient-based sigmoid neurons (`Cell`, `MultiCell`)
  2. Instance-based adaptive memory (`DataCell`)
  3. Automatic symbolic function discovery (`FuncCell`)
- **High Interpretability** – especially via human-readable formulas from FuncCell
- **Robust Edge-Case Handling**

### Core Components

#### 1. `Cell` — Single-Input Momentum-Optimized Sigmoid Neuron
Classic logistic unit with per-cell learning rate, momentum, and bias scaling (`truely`).

- Input automatically mapped from `[0,1]` → `[-1,1]`
- Standard sigmoid activation + gradient descent with momentum
- Built-in `limitation()` for numerical safety

Ideal for simple regression, gating, or building larger structures.

#### 2. `DataCell` — Non-Parametric Memory-Based Regressor
Fixed-size short-term memory buffer with distance-weighted prediction.

- Stores up to `maxdatac` (default 64) averaged input-output pairs
- Predicts via inverse-distance weighted averaging
- Smooth adaptation: blends prediction with true target before storing
- Graceful fallback (returns 0.5 when memory empty)

Excellent for denoising, time-series smoothing, or as working memory.

####  3. `MultiCell` — Multi-Input Sigmoid Neuron
Generalization of `Cell` to arbitrary input vectors.

- Supports dynamic weight count
- Shared momentum, learning rate, and safety mechanisms
- Default auto-associative target (average of normalized inputs)

Perfect for feature fusion and dense-layer-like behavior.

#### 4. `FuncCell` — Symbolic Regression & Function Discovery Unit **(Major Update in 2.1)**

Now significantly smarter and faster:

- Redesigned expression generation:
  - `comb()` generates operator skeletons (e.g., `x+sumr`, `x**forc`)
  - `comb2()` exhaustively substitutes integer constants in one pass per skeleton
  - `combs(depth)` iteratively wraps new operators around previous expressions
- Controlled search depth via `traindepth` (default 2)
- Brute-force but bounded enumeration — practical for small-to-medium complexity functions
- Selects expression with lowest total absolute error
- Falls back to best-found formula even if margin not reached
- Uses safe `eval()` only on discovered string (after validation)

Great for:
- Scientific equation discovery
- Reverse-engineering black-box functions
- Creating fully transparent, human-readable models
- Hybrid symbolic-numeric systems

### Installation

```bash
git clone https://github.com/aertsimon90/Zevihanthosa.git
cd Zevihanthosa
# Copy or import zevihanthosa.py directly into your project
```

No pip install required.

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

# 3. MultiCell – learning to approximate sum
mc = MultiCell(wcount=2, learning=0.08)
for _ in range(20000):
    a, b = random.random(), random.random()
    mc.process([a, b], target=a + b)
print("MultiCell [0.2, 0.9] ≈", mc.process([0.2, 0.9], train=False))

# 4. FuncCell – discovering square root
fc = FuncCell(traindepth=2)
for x in range(5):
    y = math.sqrt(x)
    fc.process(x, target=y)

print("Discovered function:", fc.func)
print("Test on x=3 →", fc.process(3, train=False))
```

### Why Zevihanthosa?

- **True Hybrid AI**: Neural + Memory + Symbolic in one tiny package
- **Maximum Transparency**: From weights to exact mathematical expressions
- **Lightweight & Portable**: Runs anywhere Python runs
- **Educational Gold**: Crystal-clear code, perfect for teaching or research
- **Stable & Refined**: ZevihaNut/2.1 brings major symbolic search improvements

### Future Roadmap

- Network/layer composition helpers
- Model serialization (save/load states)
- Visualization tools (weight trajectories, memory contents, discovered trees)
- Deeper symbolic search via genetic programming
- Optional acceleration backends
- Example integrations with RL and evolutionary methods

### License

MIT License — completely free for any use.

---

**Zevihanthosa — Where simplicity, interpretability, and hybrid intelligence converge.**  
*ZevihaNut/2.1 — December 2025: Smarter Symbolic Discovery. Stable. Ready.*
