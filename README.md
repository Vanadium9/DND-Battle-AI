# D&D Tactical Combat RL Simulator

Python 3.11+ project scaffold for an RL simulator of tactical D&D-style combat.

This repository currently contains only the base structure, package imports, and placeholder classes. PPO and other training algorithms are intentionally not implemented yet.

## Project Structure

```text
src/
  combat/
  agents/
  training/
  configs/
  tests/
scripts/
  run_demo.py
requirements.txt
README.md
```

## Setup

```bash
python -m venv .venv
pip install -r requirements.txt
```

## Demo Entrypoint

```bash
python scripts/run_demo.py
```

The demo only checks that the scaffold imports and instantiates the placeholder classes.

## Tests

```bash
pytest
```
