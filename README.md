# FPL AI Agent

An intelligent Fantasy Premier League manager built in two phases: a rule-based decision system grounded in expected points calculations, followed by a Proximal Policy Optimisation reinforcement learning agent.

---

## Project Overview

This project builds an FPL manager that automatically evaluates transfers, selects captains, and triggers chips. The phased approach means the project is useful and demonstrable at every stage.

**Phase 1 — Rule-Based System**
A transfer engine driven by expected points calculations across a five-gameweek horizon, with a Monte Carlo minutes simulator, constraint enforcement, and chip strategy logic.

**Phase 2 — Reinforcement Learning Agent**
A PPO agent trained on an OpenAI Gym-compatible FPL environment, reusing the Phase 1 data infrastructure and benchmarked against the rule-based system.

---

## Repository Structure

```
fpl-ai-agent/
├── data/
│   ├── raw/                  # Downloaded from FPL API and Vaastav repo
│   └── processed/            # Cleaned Parquet files
├── notebooks/                # Exploratory analysis and model validation
├── src/
│   ├── pipeline/             # Data ingestion and validation
│   ├── models/               # Expected points model and Monte Carlo simulator
│   ├── agent/                # Rule-based transfer engine and constraint layer
│   └── rl/                   # Gym environment, reward function, PPO training
├── tests/                    # Unit tests per module
├── requirements.txt
└── README.md
```

---

## Development Roadmap

| Phase | Focus | Duration |
|-------|-------|----------|
| 1 | Python & tooling foundations | Weeks 1–6 |
| 2 | Data pipeline | Weeks 7–12 |
| 3 | Expected points model | Weeks 13–16 |
| 4 | Rule-based agent | Weeks 17–22 |
| 5–7 | RL environment, PPO training, evaluation | Months 5–12 |

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/your-username/fpl-ai-agent.git
cd fpl-ai-agent
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Fetch current FPL data

```bash
python src/pipeline/fetch_fpl.py
```

This pulls the current season from the official FPL API and saves raw JSON to `data/raw/`.

---

## Data Sources

| Source | Description |
|--------|-------------|
| [FPL API](https://fantasy.premierleague.com/api/bootstrap-static/) | Current season player, fixture, and squad data. No authentication required. |
| [Vaastav / Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) | Historical seasons (2016–present). Community-maintained CSV files. |

---

## Tools & Technologies

| Category | Tool |
|----------|------|
| Language | Python 3.11+ |
| Data manipulation | Pandas, Polars |
| Data storage | Parquet |
| Numerical computing | NumPy |
| API access | requests |
| RL framework | Stable Baselines 3 |
| RL environment | Gymnasium |
| Cloud training | Google Colab |
| Version control | Git + GitHub |
| Notebooks | Jupyter |
| Visualisation | Matplotlib, Seaborn |

---

## Current Status

- [ ] Phase 1 — Python & tooling foundations
- [ ] Phase 2 — Data pipeline
- [ ] Phase 3 — Expected points model
- [ ] Phase 4 — Rule-based agent
- [ ] Phase 5 — RL environment
- [ ] Phase 6 — PPO training
- [ ] Phase 7 — Evaluation & writeup

---

*Project initiated April 2026. Built as a portfolio project demonstrating data engineering, probabilistic modelling, and reinforcement learning.*
