# Leakage-Controlled TGNN for Multi-Robot Security

This repository contains the source code for a temporal graph neural network
pilot study on compromised-robot detection in simulated multi-robot
communication networks.

The public repository intentionally contains source code and reproduction
instructions only. It does not contain private datasets, simulator outputs,
checkpoints, predictions, logs, PDFs, notebooks, or owner-only audit files.

## Evidence scope

The validated pilot uses:

- 11 independently generated scenarios
- 4 training scenarios, 1 validation scenario, and 6 test scenarios
- 20 robots and 100 timesteps per scenario
- simulator seeds 101--111
- 7 benchmark node features
- 6 edge features
- a GRU history encoder, two GAT layers, and differentiable trust propagation
- a fixed primary TGNN threshold of 0.50
- a logistic-regression baseline with a validation-selected threshold of 0.25

The primary TGNN test result over 12,000 saved node-time predictions was:

| Metric | TGNN |
|---|---:|
| Accuracy | 0.70575 |
| Precision | 0.26680 |
| Recall | 0.26958 |
| F1 | 0.26819 |
| Macro F1 | 0.54202 |
| Specificity | 0.81479 |
| False-positive rate | 0.18521 |
| MCC | 0.08405 |

These results are a simulator-based pilot, not evidence of deployment
readiness or general superiority. Multiple training seeds, component
ablations, robustness sweeps, held-out attack-family testing, and real-robot
validation remain future work. The message encoder receives no message-level
tensors in the reported benchmark and therefore contributes zero message
context.

## Installation

Use Python 3.9 or newer. The validated run used Python 3.12 with CPU PyTorch.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For a current CPU installation, install PyTorch and PyTorch Geometric using
the official compatibility instructions for the selected Python version
before installing the remaining requirements.

## Reproduction workflow

The benchmark scripts expect generated scenario artifacts under
`outputs/research_validated/`. Those artifacts are intentionally excluded
from this public repository because they contain generated datasets and
owner-only experiment records.

1. Generate independent scenarios:

   ```powershell
   python experiments\generate_validated_scenarios.py
   ```

2. Run the leakage-controlled benchmark:

   ```powershell
   python experiments\run_validated_benchmark.py
   ```

3. Analyze validation-selected operating points:

   ```powershell
   python experiments\analyze_operating_points.py
   ```

The benchmark enforces scenario-level separation and keeps temporal windows
within scenario boundaries. Scenario identifiers are metadata only and are
not model features.

## Feature and leakage controls

The saved-state benchmark node vector is:

```text
[position_x, position_y, velocity_x, velocity_y,
 heading, normalized_energy, network_degree]
```

The six edge attributes are signal strength, distance, message count, mean
latency, mean packet integrity, and an integrity/latency indicator. Simulator
attacker identities, malicious masks, compromised-state fields, attack priors,
and `fake`/`malicious` edge flags are excluded from inference features.

## Citation and status

This code accompanies a leakage-controlled implementation and pilot
evaluation. Please do not describe the current artifact as a state-of-the-art
system or claim that temporal encoding or trust propagation improves
detection without completing the corresponding ablations.
