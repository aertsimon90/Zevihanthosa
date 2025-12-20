# Zevihanthosa - Advanced Artificial Intelligence Framework  
## ZevihaNut/2.1 Model (Latest Release)

*A lightweight, pure-Python, highly extensible neural computation framework focused on hybrid parametric, non-parametric, and symbolic learning units.*

### Overview

**Zevihanthosa** is an innovative, minimalist artificial intelligence framework designed for rapid experimentation, prototyping, and deployment of unconventional, human-interpretable intelligent systems.

The **ZevihaNut/2.1** release introduces a powerful fourth core component: **FuncCell** — a symbolic regression cell capable of automatically discovering compact mathematical expressions that best fit observed input-output patterns. This addition elevates the framework from hybrid neural/memory-based learning to true **hybrid symbolic-neural intelligence**, enabling systems that can learn both adaptive weights and exact, interpretable formulas.

Built entirely in pure Python with only minimal standard library dependencies (`random`, `math`, `numpy` for controlled sampling), Zevihanthosa remains extremely lightweight, portable, and ideal for education, research, embedded systems, or creative AI exploration.

ZevihaNut/2.1 has been rigorously refined: all known bugs fixed (including critical MultiCell limitation), edge cases hardened, and the new symbolic discovery mechanism bounded for practical usability.

### Key Features

- **Pure Python Implementation**: No heavy dependencies (only `numpy` used optionally in FuncCell).
- **Online & Incremental Learning**: Every cell adapts immediately per example.
- **Momentum-Optimized Gradients**: Smooth convergence with configurable momentum (default 0.9).
- **Numerical Safety**: Built-in weight/bias clamping to prevent divergence.
- **Four Complementary Learning Paradigms**:
  - Parametric (gradient-based sigmoid neurons)
  - Non-parametric (instance-based memory averaging)
  - Multi-input fusion
  - Symbolic regression (automatic function discovery)
- **Auto-Associative & Unsupervised Modes**: Works seamlessly without explicit targets.
- **Maximum Interpretability**: Cells are transparent, inspectable, and often human-readable (especially FuncCell).

### Core Components

#### 1. `Cell` — Momentum-Optimized Single-Input Sigmoid Neuron
Classic logistic unit with momentum, per-cell learning rate, and bias scaling.

- Input normalized internally from `[0,1]` → `[-1,1]`.
- Sigmoid activation with gradient descent + momentum.
- `truely` parameter modulates bias strength (useful for gating).
- `limitation()` enforces numerical stability.

Perfect for simple regression, classification, or as gated building blocks.

#### 2. `DataCell` — Adaptive Instance-Based Memory Unit
Non-parametric learner with fixed-size short-term memory buffer.

- Stores up to `maxdatac` (default 64) recent averaged input-output pairs.
- Predicts via distance-weighted averaging of stored outputs.
- Robust edge-case handling (empty memory returns neutral 0.5).
- Smooth online adaptation by blending prediction with target.

Ideal for time-series smoothing, denoising, anomaly detection, or cognitive "working memory".

#### 3. `MultiCell` — Multi-Input Sigmoid Neuron
Generalization of `Cell` to arbitrary input vectors.

- Variable number of weights with shared learning and momentum.
- Same safety and optimization features as `Cell`.
- Default unsupervised target: average of normalized inputs.

Great for feature fusion, dense layers, or multimodal integration.

#### 4. `FuncCell` — Symbolic Regression Discovery Unit **(New in 2.1)**
A groundbreaking cell that automatically evolves compact mathematical expressions to explain observed data.

- Maintains a memory buffer of input-target pairs (like DataCell).
- Searches over compositional expressions built from basic operators: `+`, `*`, `/`, `**`, `%`.
- Controlled search depth (`traindepth`, default 2) prevents combinatorial explosion.
- Uses brute-force enumeration over operator combinations and numeric constants.
- Selects the simplest expression with lowest total absolute error.
- Once discovered, evaluates exactly using `eval()` on the learned formula.
- Falls back to memory averaging if no good fit found.

Enables interpretable modeling, equation discovery, and hybrid symbolic-numeric reasoning.

Excellent for scientific modeling, reverse engineering functions, or creating fully transparent predictors.

### Installation

```bash
# Clone the repository or copy the source file
git clone https://github.com/aertsimon90/Zevihanthosa.git
cd Zevihanthosa
# Then simply import in your project
```

No pip install needed — pure Python module.

### Usage Examples

```python
from zevihanthosa import *
import random
import math

# Example 1: Cell learning a step function
cell = Cell(learning=0.2)
for _ in range(10000):
    x = random.random()
    target = 1.0 if x > 0.6 else 0.0
    cell.process(x, target)
print("Cell 0.5 →", cell.process(0.5, train=False))
print("Cell 0.7 →", cell.process(0.7, train=False))

# Example 2: DataCell denoising noisy signal
dc = DataCell(maxdatac=50)
for i in range(200):
    noisy = 0.5 + random.gauss(0, 0.15)
    dc.process(noisy, target=0.5)
print("Current estimate:", dc.process(0.65, train=False))

# Example 3: MultiCell approximating addition
mc = MultiCell(wcount=2, learning=0.1)
for _ in range(15000):
    a, b = random.random(), random.random()
    mc.process([a, b], target=a + b)
print("MultiCell [0.4, 0.6] →", mc.process([0.4, 0.6], train=False))

# Example 4: FuncCell discovering x² pattern (New!)
fc = FuncCell(maxdatac=64, range=10, rangc=1, traindepth=2)
for x in [i/10 for i in range(-20, 21)]:
    y = x ** 2 / 10  # scaled parabola
    fc.process(x*0.5 + 0.5, target=y)  # normalize x to [0,1]

print("Discovered function:", fc.func)
print("Test x=0.8 (norm) →", fc.process(0.8, train=False))
```

### Why Zevihanthosa?

- **True Hybrid Intelligence**: Combines neural, memory-based, and symbolic learning in one lightweight package.
- **Unparalleled Interpretability**: From weights to full mathematical formulas.
- **Educational & Research Friendly**: Clear, readable code with no black-box magic.
- **Lightweight & Portable**: Runs on microcontrollers, servers, or notebooks alike.
- **Stable & Battle-Tested**: ZevihaNut/2.1 fixes all prior bugs and introduces safe symbolic search.

### Future Roadmap

- Full network/layer composition utilities.
- Cell serialization and checkpointing.
- Visualization dashboard for memory, weights, and discovered functions.
- Genetic programming extensions for deeper symbolic search.
- Optional PyTorch backend for acceleration.
- Reinforcement learning and evolutionary integration examples.

### License

MIT License — free for commercial, academic, and personal use.

---

**Zevihanthosa — Simplicity. Interpretability. Hybrid Intelligence.**  
*ZevihaNut/2.1 — Now with Symbolic Function Discovery. Tested. Ready.*
