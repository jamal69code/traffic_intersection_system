# Simplified DQN Traffic Signal Control

## 1. Project Overview

This project is a simplified version of the supplied DQN traffic-control program.

The goal is to train a Deep Q-Network (DQN) to choose between two traffic-signal actions:

- `0` = Keep the current signal phase
- `1` = Switch the signal phase

The environment contains four traffic queues:

- Lane 0
- Lane 1
- Lane 2
- Lane 3

When phase 0 is active, lanes 0 and 1 receive green service.
When phase 1 is active, lanes 2 and 3 receive green service.

The DQN tries to reduce the average traffic queue.

---

## 2. Files

### `simplified_dqn_traffic_control.ipynb`

Main Jupyter Notebook containing:

1. User input
2. Traffic environment
3. DQN neural network
4. DQN agent
5. Training
6. Evaluation
7. Graphs
8. Formulas
9. Output explanation

### `traffic_dqn_model.pth`

Created automatically after training. It contains the trained neural-network parameters.

---

## 3. Requirements

Python 3.x is recommended.

Install:

```bash
pip install numpy matplotlib gymnasium torch
```

Jupyter can be installed with:

```bash
pip install notebook
```

Run:

```bash
jupyter notebook
```

Then open:

```text
simplified_dqn_traffic_control.ipynb
```

---

## 4. User Input

The notebook asks for:

```text
Training episodes [100]:
Steps per episode [100]:
Evaluation episodes [10]:
```

Press Enter to accept the default.

For a quick test:

```text
Training episodes = 50
Steps per episode = 50
Evaluation episodes = 5
```

For a more complete experiment:

```text
Training episodes = 300
Steps per episode = 100
Evaluation episodes = 20
```

The original supplied program used 300 training episodes and 100 steps per episode.

---

## 5. Working of the System

### Step 1: Initialize traffic

Each episode starts with four small random queues.

The state is:

```text
[queue_0, queue_1, queue_2, queue_3, phase]
```

Therefore, the neural network receives 5 input values.

### Step 2: Select an action

The agent chooses:

```text
0 -> Keep phase
1 -> Switch phase
```

During training, epsilon-greedy exploration allows the agent to try random actions.

### Step 3: Add arriving vehicles

New vehicles are generated using a Poisson distribution.

Conceptually:

```text
new queue = old queue + arriving vehicles
```

### Step 4: Serve vehicles

Two lanes are served according to the current green phase.

Up to 3 vehicles are removed from each green lane during one step.

### Step 5: Calculate reward

The reward is:

```text
Reward = -Average Queue - Switch Penalty
```

The switch penalty is:

```text
0.1 if the signal is switched
0 otherwise
```

Therefore, smaller queues produce a better (less negative) reward.

### Step 6: Store experience

Each experience is stored as:

```text
(state, action, reward, next_state, done)
```

The replay memory allows the DQN to learn from randomly sampled past experiences.

### Step 7: Train the DQN

The DQN predicts two Q-values:

```text
Q(state, keep)
Q(state, switch)
```

The action with the larger Q-value is preferred when the agent is not exploring.

---

## 6. Important Formulas

### 6.1 Average Queue

For four queues:

```text
Qavg = (Q0 + Q1 + Q2 + Q3) / 4
```

Example:

```text
Queues = [4, 6, 2, 8]

Qavg = (4 + 6 + 2 + 8) / 4
     = 20 / 4
     = 5
```

### 6.2 Reward

```text
R = -Qavg - Pswitch
```

If:

```text
Qavg = 5
Pswitch = 0.1
```

then:

```text
R = -5 - 0.1
  = -5.1
```

### 6.3 DQN Target

The simplified DQN target is:

```text
Y = R + (1-d) × gamma × max Qtarget(next_state, next_action)
```

where:

```text
gamma = 0.99
d = 1 when the episode ends
d = 0 otherwise
```

If the episode is not finished, the future Q-value contributes to learning.

### 6.4 MSE Loss

```text
Loss = (Current Q - Target Q)^2
```

The optimizer changes the neural-network weights to reduce this error.

### 6.5 Queue Reduction

The final comparison uses:

```text
Queue Reduction (%) =
    ((Random Queue - DQN Queue) / Random Queue) × 100
```

Example only:

```text
Random Queue = 10
DQN Queue = 8

Reduction = ((10 - 8) / 10) × 100
          = 20%
```

Do not use this example as the actual experimental result. Use the values printed by your notebook.

---

## 7. Understanding the Output

A typical final section looks like:

```text
========== FINAL RESULT ==========
Random Policy Average Queue : ...
DQN Average Queue           : ...
Queue Reduction             : ...%
```

### Random Policy Average Queue

This is the average queue produced when actions are selected randomly.

### DQN Average Queue

This is the average queue when the trained DQN selects actions.

### Queue Reduction

This compares the two measured queues.

A positive value means:

```text
DQN queue < Random-policy queue
```

for that evaluation.

---

## 8. Graph Explanation

### Graph 1: Training Reward

X-axis:

```text
Episode
```

Y-axis:

```text
Total Reward
```

Because queue length is penalized, a less negative reward generally indicates lower queues.

Do not expect a perfectly smooth curve.

### Graph 2: Average Traffic Queue

X-axis:

```text
Episode
```

Y-axis:

```text
Average Queue
```

This shows how the simulated queue changes during training.

### Graph 3: DQN Training Loss

X-axis:

```text
Episode
```

Y-axis:

```text
MSE Loss
```

The loss measures the difference between the predicted Q-value and the calculated DQN target.

Some fluctuation is normal in reinforcement learning.

### Graph 4: Epsilon Decay

Epsilon starts at:

```text
1.0
```

and decreases approximately according to:

```text
epsilon = max(0.01, epsilon × 0.995)
```

A high epsilon means more random exploration.

A low epsilon means the learned policy is used more often.

---

## 9. Complete Execution Flow

```text
Start
  |
  v
Take user inputs
  |
  v
Create traffic environment
  |
  v
Create DQN agent
  |
  v
Initialize queues
  |
  v
Observe state
  |
  v
Choose Keep/Switch action
  |
  v
Generate arriving vehicles
  |
  v
Serve vehicles on green lanes
  |
  v
Calculate queue and reward
  |
  v
Store experience
  |
  v
Train DQN
  |
  v
Repeat until episode ends
  |
  v
Repeat for all episodes
  |
  v
Save trained model
  |
  v
Evaluate DQN
  |
  v
Evaluate random policy
  |
  v
Calculate queue reduction
  |
  v
Display graphs and final result
```

---

## 10. What to Say in a Viva/Presentation

**Purpose:**

> The purpose of this project is to demonstrate how Deep Q-Learning can be used to control a simulated traffic signal by selecting whether to keep or switch the signal phase in order to reduce traffic queues.

**Input:**

> The input state contains four lane queue lengths and the current traffic-signal phase.

**Action:**

> The agent has two actions: keep the current phase or switch the phase.

**Reward:**

> The reward is the negative average queue, with a small penalty for switching the signal.

**Algorithm:**

> The DQN uses a neural network, epsilon-greedy exploration, experience replay, a target network, and the Bellman target.

**Final result:**

> The trained DQN is compared with a random policy using average queue length and percentage queue reduction.

---

## 11. Important Note

This is a simulation for learning and demonstration.

It does not represent a production traffic-light controller. A real deployment would need real traffic sensors/data, calibrated traffic-flow models, safety constraints, signal timing regulations, multi-intersection coordination, and extensive validation.

The exact numerical result is not fixed because traffic arrivals and neural-network training are stochastic. Report the numbers generated by your own run.
