# Traffic Signal Optimization Using Deep Reinforcement Learning

## 1. Project Overview

This project uses **Deep Reinforcement Learning (DRL)** and a **Deep Q-Network (DQN)** to optimize traffic signal control at a four-way intersection.

The system learns whether to:

* Keep the current traffic signal phase
* Switch the traffic signal phase

The objective is to reduce the average number of vehicles waiting at the intersection.

---

## 2. Technologies Used

* Python
* Gymnasium
* NumPy
* PyTorch
* Matplotlib
* Deep Q-Network
* Experience Replay
* Epsilon-Greedy Exploration

---

## 3. Traffic Environment

The intersection contains four traffic queues:

```text
        NORTH
          |
          |
WEST -----+----- EAST
          |
          |
        SOUTH
```

The traffic signal has two phases:

```text
Phase 0 → North + South Green
Phase 1 → East + West Green
```

### Actions

| Action | Meaning            |
| ------ | ------------------ |
| 0      | Keep current phase |
| 1      | Switch phase       |

### State

The DQN receives five values:

```text
[North Queue,
 South Queue,
 East Queue,
 West Queue,
 Current Phase]
```

---

## 4. Reward Function

The reward is based on average queue length:

```text
Reward = -Average Queue - Switch Penalty
```

A lower queue produces a better reward.

A small penalty is also applied when the signal is switched to discourage unnecessary switching.

---

## 5. DQN Architecture

The neural network contains:

```text
Input: 5 values
       ↓
Dense Layer: 64
       ↓
ReLU
       ↓
Dense Layer: 64
       ↓
ReLU
       ↓
Output: 2 Q-values
```

The two output values represent:

```text
Q-value(Action 0)
Q-value(Action 1)
```

The action with the larger Q-value is selected when the agent exploits its learned knowledge.

---

## 6. Experience Replay

The agent stores previous experiences:

```text
(state, action, reward, next_state, done)
```

These experiences are stored in a replay memory.

Random batches are selected from the memory during training. This helps reduce correlation between consecutive experiences.

---

## 7. Target Network

The project uses two neural networks:

* Main Q-network
* Target Q-network

The target network is periodically updated from the main network.

This makes DQN training more stable.

---

## 8. Exploration

The project uses epsilon-greedy exploration.

Initially:

```text
Epsilon = 1.0
```

The agent mainly chooses random actions.

During training, epsilon gradually decreases until it reaches:

```text
Epsilon = 0.01
```

The agent therefore gradually moves from exploration toward exploitation.

---

## 9. Training

The default training configuration is:

```text
Episodes       = 300
Episode length = 100 steps
Learning rate  = 0.001
Gamma          = 0.99
Replay memory  = 10,000
Batch size     = 64
```

Run the project using:

```bash
python traffic_dqn.py
```

---

## 10. Output

After training, the program generates:

```text
traffic_dqn_model.pth
reward.png
queue.png
loss.png
epsilon.png
```

### `reward.png`

Shows how the DQN reward changes during training.

### `queue.png`

Shows the average traffic queue length.

### `loss.png`

Shows neural-network training loss.

### `epsilon.png`

Shows the reduction of exploration over time.

### `traffic_dqn_model.pth`

Contains the trained DQN model parameters.

---

## 11. Performance Analysis

The program evaluates the trained DQN against a random traffic-signal policy.

It reports:

```text
Random Policy Queue
DQN Queue
Queue Reduction
```

The queue reduction is calculated as:

```text
Queue Reduction (%) =
(Random Queue - DQN Queue)
-------------------------------- × 100
        Random Queue
```

The actual performance values depend on the stochastic traffic simulation and should be taken from the program output.

---

## 12. Project Workflow

```text
Traffic Environment
        ↓
Observe Traffic Queues
        ↓
DQN Agent
        ↓
Select Signal Action
        ↓
Traffic Environment
        ↓
Receive Reward
        ↓
Store Experience
        ↓
Experience Replay
        ↓
Update DQN
        ↓
Update Target Network
        ↓
Evaluate Performance
        ↓
Generate Graphs
```

---

## 13. Main Objective

The main objective is to demonstrate how reinforcement learning can learn traffic signal control from simulated traffic conditions rather than using a fixed signal timing strategy.

The project can be extended with:

* Real traffic datasets
* Multiple intersections
* Vehicle waiting time
* Emergency vehicle priority
* Pedestrian signals
* Adaptive traffic arrival rates
* Real-time camera-based vehicle detection
* Multi-agent reinforcement learning

---

## 14. Conclusion

The project demonstrates a DQN-based approach for adaptive traffic signal optimization.

The agent observes traffic queues, chooses signal actions, receives rewards based on congestion, and gradually learns a traffic-control policy through experience.

The generated graphs and comparison with a random policy provide quantitative information about the training process and simulated traffic performance.
