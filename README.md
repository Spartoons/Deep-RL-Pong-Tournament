# Deep Reinforcement Learning: From Pong to Complex Environments

**Authors:**
* Juan Sebastian Mañosas Guerrero Alveolos (NIU: 1671913)
* Mateo Jure (NIU: 1705977)
* Aran Oliveras (NIU: 1708069)

**Course:** Bachelor's Degree in Artificial Intelligence - UAB  
**Date:** December 2025

---

## 📖 Project Overview

This repository contains the implementation for the Deep Reinforcement Learning project. The project addresses three distinct challenges using the Arcade Learning Environment (ALE):

1.  **Solving Pong:** A comparison between DQN and a custom PPO implementation on `PongNoFrameskip-v4`.
2.  **Pong World Tournament:** A Multi-Agent Reinforcement Learning (MARL) system ("The Arena") training competitive agents using PettingZoo and PPG/Impala architectures.
3.  **Complex Environment (Basic Math):** A study on sparse rewards using `ALE/BasicMath-v5`, comparing Asynchronous DQN and PPG.

---

## ⚙️ Installation & Requirements

This project requires Python 3.8+ and the Gymnasium API (v1.0.0+).

### 1. Clone the repository
```bash
git clone [https://github.com/YourUsername/YourRepoName.git](https://github.com/YourUsername/YourRepoName.git)
cd YourRepoName

```

### 2. Install DependenciesWe have provided a `requirements.txt` file. Ensure `AutoROM` is accepted to download Atari ROMs.

```bash
pip install -r requirements.txt
AutoROM --accept-license

```

**Key Libraries used:**

* `torch` (PyTorch)
* `gymnasium[atari, accept-rom-license]`
* `pettingzoo[atari]`
* `supersuit`
* `wandb` (Weights & Biases for logging)
* `stable-baselines3` (Used for some baseline comparisons)

---

## 📂 Structure & Usage###🕹️ Part 1: Solving PongLocated in `Part1_Pong/`. Compares an Off-Policy DQN against an On-Policy custom PPO.

* **Train DQN:**
```bash
python Part1_Pong/DQN_Pong.py

```


* **Train PPO:**
```bash
python Part1_Pong/PPO_Agent.py

```


* **Results:** The PPO agent achieved a score of **20.44** with 100% win rate, converging significantly faster than DQN due to parallel environment collection.

### 🏆 Part 2: Pong World TournamentLocated in `Part2_Tournament/`. Uses a custom "Arena" framework to train agents via self-play.

* **Train the Duel (Alpha vs Beta):**
```bash
python Part2_Tournament/arena.py --mode train

```


* **Evaluate Champions:**
Run the evaluation script to see the "Alpha" (Right) vs "Beta" (Left) match logic described in the report.
```bash
python Part2_Tournament/eval_duel.py

```


* **Video:** A full episode recording is available at `videos/champions_match.mp4`.

### 🧮 Part 3: Complex Environment (Basic Math)Located in `Part3_ComplexEnv/`. Solves `ALE/BasicMath-v5`, a sparse-reward arithmetic game.

* **Run DQN (Best Model):**
```bash
python Part3_ComplexEnv/DQN_Impala.py

```


* **Run PPG (Experimental):**
```bash
python Part3_ComplexEnv/PPG_Impala.py

```


* **Results:** DQN (Impala) successfully solved the environment (Avg Reward: 3.54, Max: 9.0) by leveraging Experience Replay to overcome reward sparsity. PPG failed to converge.

---

## 📹 Video Demonstrations* **Pong Championship:** [See Video](https://www.google.com/search?q=./videos/champions_match.mp4) - Demonstrates the rallying behavior evolved during the "Duel" training phase.
* **Basic Math Solution:** [See Video](https://www.google.com/search?q=./videos/basic_math_demo.mp4) - Shows the DQN agent correctly inputting multi-step arithmetic answers.

---

## 📄 License
This project is submitted for academic evaluation at Universitat Autònoma de Barcelona.
