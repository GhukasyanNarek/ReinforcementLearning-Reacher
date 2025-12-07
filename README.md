# Reinforcement Learning – Reacher-v5 (PPO, TD3, SAC)

This project implements and compares three Deep Reinforcement Learning algorithms  
(PPO, TD3, SAC) on the Reacher-v5 continuous-control environment from Gymnasium.

The goal is to evaluate and compare these algorithms using:

- Training curves (TensorBoard logs)
- Deterministic evaluation returns
- Statistical comparison of results

This project was completed as part of the RL course (Fall 2025).

---

## Project Structure

```text
project/
│
├── code/                     # Training & evaluation scripts
│   ├── train_ppo.py
│   ├── train_td3.py
│   ├── train_sac.py
│   ├── evaluate_ppo.py
│   ├── evaluate_td3.py
│   ├── evaluate_sac.py
│   └── evaluate_all.py
│
├── models/                   # Saved final models + checkpoints
│   ├── ppo_final.zip
│   ├── td3_final.zip
│   ├── sac_final.zip
│   └── checkpoints/
│       ├── ppo/
│       ├── td3/
│       └── sac/
│
├── logs/                     # TensorBoard logs
│   ├── ppo/
│   ├── td3/
│   └── sac/
│
├── plots/                    # Generated evaluation plots
│   ├── eval_results.csv
│   ├── comparison.png
│   ├── ppo_curve.png
│   ├── td3_curve.png
│   └── sac_curve.png
│
├── report.pdf                # Final 3–5 page report
└── requirements.txt          # Dependencies
```

---

## Installation

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Training the Agents

Run the training scripts from the **project root directory**:

```bash
python code/train_ppo.py
python code/train_td3.py
python code/train_sac.py
```

Each script will:
  - initialize the Reacher-v5 environment
  - train the agent using Stable-Baselines3
  - log training metrics inside logs/<algorithm>/
  - save intermediate checkpoints every 50k steps into models/checkpoints/<algorithm>/

It will also save the final trained model as:
  - models/ppo_final.zip
  - models/td3_final.zip
  - models/sac_final.zip

---

## Evaluating the Agents

### Evaluate all algorithms together:

```bash
python code/evaluate_all.py
```

This script will:
  - load the trained PPO, TD3, and SAC models
  - run 20 deterministic evaluation episodes for each algorithm
  - compute mean, standard deviation, min, and max returns
  - print a performance summary
  - save results to plots/eval_results.csv

### Evaluate single algorithm:

```bash
python code/evaluate_ppo.py
python code/evaluate_td3.py
python code/evaluate_sac.py
```

## Evaluation Summary (20 Deterministic Episodes per Algorithm)
/////

Across 20 deterministic evaluation episodes for each algorithm, SAC achieves the best overall performance, followed by TD3, while PPO performs the worst.

### **Mean Return**
- **SAC:** −3.47  
- **TD3:** −5.30  
- **PPO:** −5.61  

SAC achieves the highest average return, indicating the best ability to reach the target while minimizing control cost.

### **Stability (Standard Deviation)**
- **SAC:** 1.24  *(most stable)*  
- **TD3:** 1.45  
- **PPO:** 2.27  *(least stable)*  

SAC not only performs best but also has the lowest variance, meaning its performance is consistent across episodes.

### **Performance Range**

| Algorithm | Best Episode | Worst Episode |
|----------|--------------|----------------|
| **SAC** | −1.74 | −6.32 |
| **TD3** | −1.81 | −7.93 |
| **PPO** | −2.19 | −9.85 |

PPO has the widest range, meaning it is the least stable.  
SAC rarely collapses below a return of −6, demonstrating reliability.

### **Conclusion**

**Soft Actor-Critic (SAC)** is the strongest algorithm for the Reacher-v5 task:
- highest mean return  
- lowest variance  
- most stable performance  

TD3 performs moderately well.  
PPO performs the worst and is unstable on this task.


