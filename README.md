# Zevihanthosa - Advanced Artificial Intelligence Framework  
## ZevihaNut/2.0 Model

*A lightweight, pure-Python, highly extensible neural computation framework focused on hybrid parametric and non-parametric learning units.*

### Overview

**Zevihanthosa** is an innovative, minimalist artificial intelligence framework designed for rapid prototyping, experimentation, and deployment of custom neural-inspired architectures. At its core lies the **ZevihaNut/2.0** model release — a fully standardized, rigorously tested, and stabilized version that introduces three foundational "cell" types for building intelligent systems.

Unlike traditional deep learning frameworks that rely on massive layers of homogeneous neurons, Zevihanthosa emphasizes **modularity, interpretability, and hybrid learning paradigms**. It combines classic gradient-based sigmoid units with momentum optimization and a unique instance-based memory cell, enabling the creation of systems that blend symbolic precision with adaptive, memory-augmented reasoning.

The entire framework is implemented in pure Python with no external dependencies beyond the standard library (`random` and `math`), making it extremely lightweight, portable, and suitable for embedded or educational environments.

ZevihaNut/2.0 has undergone extensive internal testing, including edge cases (empty memory, zero differences, extreme values), basic functionality checks, and weight limitation enforcement. All tests have passed successfully, confirming stability and correctness.

### Key Features

- **Pure Python Implementation**: No heavy dependencies — runs anywhere Python does.
- **Online Learning**: All cells update immediately per sample (fully incremental).
- **Built-in Momentum**: Smooth and accelerated gradient updates (default momentum 0.9).
- **Weight/Bias Clamping**: Automatic limitation to prevent explosion/divergence.
- **Hybrid Intelligence**: Combines traditional neural computation with memory-based non-parametric learning.
- **Auto-Associative Defaults**: Cells can operate in unsupervised modes when targets are omitted.
- **Highly Customizable**: Easy to extend with new cell types or compose into complex networks.

### Core Components

#### 1. `Cell` — Momentum-Optimized Sigmoid Neuron
A classic single-input logistic unit enhanced with momentum and advanced features.

- Input range assumption: `[0, 1]` (internally mapped to `[-1, 1]`).
- Sigmoid activation: `σ(z) = 1 / (1 + exp(-z))`.
- Gradient descent with momentum on both weight and bias.
- `truely` parameter: Scales bias influence (useful for gating or modulation).
- Separate learning rate per cell.
- `limitation()` method clamps weight ∈ [1/512, 512] and bias ∈ [-512, 512] by default.

Ideal for binary classification, regression, or as building blocks in larger networks.

#### 2. `DataCell` — Adaptive Memory-Based Regression Unit
A non-parametric, instance-based learner with fixed short-term memory.

- Maintains a buffer of up to `maxdatac` (default 64) past (input → averaged output) pairs.
- For inference: Performs weighted averaging of stored outputs, where weights are inversely proportional to input distance.
- Highly robust handling of edge cases (empty memory → returns 0.5 neutral).
- During training: Blends prediction with target and stores the average (smooth online adaptation).
- Behaves like a dynamic local regressor or temporal smoother.

Excellent for time-series forecasting, anomaly detection, or as a "working memory" module in cognitive architectures.

#### 3. `MultiCell` — Multi-Input Sigmoid Neuron
Generalization of `Cell` to arbitrary input dimensions.

- Supports variable number of weighted inputs.
- Same momentum, learning rate, truely, and clamping mechanics as `Cell`.
- Default unsupervised target: average of (normalized) inputs (autoencoder-like behavior).

Perfect for feature fusion, multi-modal processing, or dense layer equivalents.

### Installation

```bash
# Simply copy the source file into your project
git clone https://github.com/aertsimon90/Zevihanthosa.git
# or directly save the classes into zevihanthosa.py
```

No pip install required — just import the classes.

### Usage Examples

```python
from zevihanthosa import *
import random

# Example 1: Basic Cell learning XOR-like pattern
cell = Cell(learning=0.1)
for _ in range(10000):
    inp = random.choice([0.0, 1.0])
    target = 1.0 if inp > 0.5 else 0.0
    cell.process(inp, target)

print("Trained Cell: 0.0 →", cell.process(0.0, train=False))
print("Trained Cell: 1.0 →", cell.process(1.0, train=False))

# Example 2: DataCell as adaptive filter
dc = DataCell(maxdatac=32)
for i in range(100):
    noisy_input = 0.5 + random.gauss(0, 0.1)
    dc.process(noisy_input, target=0.5)  # trying to learn mean=0.5

print("Denoised estimate:", dc.process(0.6, train=False))

# Example 3: MultiCell for simple addition approximation
mc = MultiCell(wcount=2, learning=0.05)
for _ in range(20000):
    a, b = random.random(), random.random()
    mc.process([a, b], target=(a + b)/2)  # learn average

print("MultiCell [0.3, 0.7] →", mc.process([0.3, 0.7], train=False))
```

### Why Zevihanthosa?

- **Educational Clarity**: Every operation is transparent and readable.
- **Research Flexibility**: Easy to modify activation functions, add new cell types, or hybridize.
- **Efficiency**: Minimal overhead — ideal for micro-controllers or real-time systems.
- **Stability**: ZevihaNut/2.0 fixes known edge cases and enforces numerical safety.

### Future Roadmap

- Layer and Network abstraction utilities.
- Serialization (save/load cell states).
- Visualization tools for weight evolution and memory contents.
- Integration examples with reinforcement learning or evolutionary algorithms.
- GPU acceleration hooks (optional).

### License

MIT License — free for commercial and academic use.

---

**Zevihanthosa — Where simplicity meets advanced hybrid intelligence.**  
*ZevihaNut/2.0 — Tested. Standardized. Ready.*
