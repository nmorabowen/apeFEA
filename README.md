# 🐒 apeFEA — Nonlinear Finite Element Framework

**LARGA VIDA AL LADRUÑO!!!**

`apeFEA` is a research-oriented nonlinear finite element analysis (FEA) framework focused on structural analysis using frame elements with advanced transformation formulations. Developed by earthquake engineers and computational mechanics nerds, this library brings together clarity, extensibility, and monkey energy.

---

## 🚀 Features

- 🌐 Corotational and P-Delta transformations
- 📏 Explicit units (mm, N, kgf, MPa, etc.)
- 🧠 Abstract base classes for extensible design (e.g. materials, solvers, elements)
- 🪢 Newton–Raphson nonlinear solver
- 🦴 Modular: clean separation of `core`, `materials`, `elements`, `transformations`, and `solvers`
- 📊 Integrated plotting using `matplotlib`
- 🧪 Full testing support (with `pytest`)

---

## 📁 Package Structure

```
apeFEA/
├── core/               # Node, DOF mapping, TimeSeries, Model manager
├── elements/           
│   └── one_dimension/  # Beam, Frame elements
│       └── transformations/  # Corotational, Linear, P-Delta
├── materials/          # Material models
├── solver/             # Solvers (e.g., Newton-Raphson)
└── tests/              # Unit tests
```

---

## 🛠 Installation

```bash
git clone https://github.com/yourusername/apeFEA.git
cd apeFEA
pip install -e .
```

---

## 🧪 Running Tests

```bash
pytest tests/
```

---

## 🧬 Example Usage

```python
from apeFEA.core.model import Model
from apeFEA.elements.one_dimension.frame import FrameElement
from apeFEA.solver.newton_raphson import NewtonRaphsonSolver

# Build your FEM model here...
```

More examples coming soon.

---

## 👥 Authors

- Nicolás Mora Bowen — [nmorabowen@gmail.com](mailto:nmorabowen@gmail.com)
- Patricio Palacios — [pxpalacios@gmail.com](mailto:pxpalacios@gmail.com)

---

## 📜 License

MIT License (2025) — use freely, cite respectfully, and never trust linearity.

---

> “Si la estructura no converge, quizás está tratando de decirte algo...”
