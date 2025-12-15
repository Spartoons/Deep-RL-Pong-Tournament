# Deep Reinforcement Learning Project**Authors:**

* Juan Sebastian Mañosas Guerrero Alveolos (NIU: 1671913)
* Mateo Jure (NIU: 1705977)
* Aran Oliveras (NIU: 1708069)

**Course:** Bachelor's Degree in Artificial Intelligence - UAB
**Date:** December 2025

---

## 📖 Project OverviewThis repository contains the source code for our Deep Reinforcement Learning project. The solution is divided into three parts, corresponding to the assignment requirements:

1. 
**Part 1: Solving Pong** - Comparison of DQN and PPO on `PongNoFrameskip-v4`.


2. 
**Part 2: Pong World Tournament** - A Multi-Agent League ("The Arena") using PettingZoo and PPG agents.


3. 
**Part 3: Complex Environment** - Solving `ALE/BasicMath-v5` using ImpalaCNN architectures.



---

## ⚙️ Installation & Requirements###1. PrerequisitesEnsure you have **Python 3.8+** installed. This project relies on `gymnasium` (v1.0.0+), `pettingzoo`, and `torch`.

###2. SetupClone the repository and install the dependencies listed in `requirements.txt`:

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Atari ROMs (Required for Pong and BasicMath)
AutoROM --accept-license

```

**Note:** The `AutoROM` command is mandatory to download the game files.

---

## 🚀 Execution Instructions###🕹️ Part 1: Solving the Pong Environment*Located in directory: `Part1_Pong/*`

**1. Training**
To train the baseline DQN agent or the PPO agent:

```bash
# Train the DQN Agent
python Part1_Pong/DQN_Pong.py

# Train the PPO Agent
python Part1_Pong/PPO_Agent.py

```

**2. Testing & Evaluation**
To evaluate the trained models and verify the preprocessing wrappers (preprocessing verification):

```bash
python Part1_Pong/TestPong.py

```

*This script loads the environment with the defined wrappers and runs test episodes to verify observation spaces and rewards*.

---

### 🏆 Part 2: Pong World Tournament*Located in directory: `Part2_Tournament/*`

**1. Training ("The Arena")**
To launch the multi-agent training loop (Self-Play):

```bash
python Part2_Tournament/arena.py

```

*This script initializes the PettingZoo environment and trains the agents (Alpha vs Beta) using the logic described in the report*.

**2. Testing ("The Duel")**
To evaluate the final champions (Alpha vs Beta) or run the "Head-to-Head" evaluation:

```bash
python Part2_Tournament/eval_duel.py

```

*This script performs the 100-episode evaluation and prints the Win Rate and Average Reward as reported in the project documentation*.

---

### 🧮 Part 3: Complex Environment (Basic Math)*Located in directory: `Part3_ComplexEnv/*`

**1. Training**
We provide two implementations for the `ALE/BasicMath-v5` environment:

```bash
# Train the DQN Agent (Best Performing Model)
python Part3_ComplexEnv/DQN_Impala.py

# Train the PPG Agent (Experimental)
python Part3_ComplexEnv/PPG_Impala.py

```

*Note: The DQN agent uses an ImpalaCNN backbone and asynchronous data collection*.

**2. Testing & Video Generation**
To load a saved checkpoint and record a gameplay video:

```bash
python Part3_ComplexEnv/eval_video.py

```

*This script loads the model from the `models/` subdirectory and saves a `.mp4` file to `videos/*`.

---

## 📂 File Structure
```text
/
[cite_start]├── requirements.txt            # Dependencies [cite: 352]
[cite_start]├── README.md                   # Execution instructions [cite: 353]
├── Part1_Pong/                 # Source code for Part 1
│   ├── DQN_Pong.py
│   └── TestPong.py
├── Part2_Tournament/           # Source code for Part 2
│   ├── arena.py
│   ├── eval_duel.py
│   └── models/
└── Part3_ComplexEnv/           # Source code for Part 3
    ├── DQN_Impala.py
    └── eval_video.py

```

---

## 📹 VisualizationsVideo demonstrations of the agents are available in the `videos/` folder:

* 
`champions_match.mp4`: A full episode of the Part 2 Tournament Final.


* `basic_math_demo.mp4`: A demonstration of the Part 3 solution.

---

**University Compliance:**
This project is submitted for the "Deep Reinforcement Learning" course at Universitat Autònoma de Barcelona.
