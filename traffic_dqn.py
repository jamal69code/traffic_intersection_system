import random
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.optim as optim

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ===================== ENVIRONMENT =====================

class TrafficEnv(gym.Env):
    def __init__(self, max_steps=100):
        super().__init__()
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            0, 100, shape=(5,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.q = self.np_random.integers(0, 10, 4).astype(float)
        self.phase = 0
        self.steps = 0
        self.switches = 0
        return self.obs(), {}

    def obs(self):
        return np.append(self.q, self.phase).astype(np.float32)

    def step(self, action):
        self.steps += 1

        if action == 1:
            self.phase = 1 - self.phase
            self.switches += 1

        self.q += self.np_random.poisson(1.5, 4)

        green = [0, 1] if self.phase == 0 else [2, 3]
        for i in green:
            self.q[i] = max(0, self.q[i] - 3)

        avg_q = np.mean(self.q)
        reward = -avg_q - (0.1 if action == 1 else 0)

        done = self.steps >= self.max_steps

        return self.obs(), reward, False, done, {
            "queue": avg_q,
            "switches": self.switches
        }


# ===================== DQN =====================

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)


class Agent:
    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.q = DQN().to(self.device)
        self.target = DQN().to(self.device)
        self.target.load_state_dict(self.q.state_dict())

        self.optimizer = optim.Adam(self.q.parameters(), lr=0.001)
        self.memory = deque(maxlen=10000)

        self.gamma = 0.99
        self.epsilon = 1.0
        self.eps_min = 0.01
        self.eps_decay = 0.995

    def action(self, state, train=True):
        if train and random.random() < self.epsilon:
            return random.randrange(2)

        x = torch.tensor(
            state, dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            return self.q(x).argmax(1).item()

    def train(self, batch=64):
        if len(self.memory) < batch:
            return 0

        data = random.sample(self.memory, batch)
        s, a, r, ns, d = zip(*data)

        s = torch.tensor(np.array(s), dtype=torch.float32).to(self.device)
        ns = torch.tensor(np.array(ns), dtype=torch.float32).to(self.device)
        a = torch.tensor(a).unsqueeze(1).to(self.device)
        r = torch.tensor(r, dtype=torch.float32).unsqueeze(1).to(self.device)
        d = torch.tensor(d, dtype=torch.float32).unsqueeze(1).to(self.device)

        current = self.q(s).gather(1, a)

        with torch.no_grad():
            future = self.target(ns).max(1, keepdim=True)[0]
            target = r + (1 - d) * self.gamma * future

        loss = nn.MSELoss()(current, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


# ===================== TRAINING =====================

def run():
    env = TrafficEnv()
    agent = Agent()

    episodes = 300
    rewards, queues, losses, eps = [], [], [], []

    print("Starting DQN Traffic Signal Training...\n")

    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        total_reward = total_queue = total_loss = 0
        count = 0
        done = False

        while not done:
            action = agent.action(state)
            ns, reward, _, done, info = env.step(action)

            agent.memory.append(
                (state, action, reward, ns, done)
            )

            loss = agent.train()

            state = ns
            total_reward += reward
            total_queue += info["queue"]

            if loss:
                total_loss += loss
                count += 1

        agent.epsilon = max(
            agent.eps_min,
            agent.epsilon * agent.eps_decay
        )

        if ep % 10 == 0:
            agent.target.load_state_dict(agent.q.state_dict())

        rewards.append(total_reward)
        queues.append(total_queue / 100)
        losses.append(total_loss / max(count, 1))
        eps.append(agent.epsilon)

        if ep % 20 == 0:
            print(
                f"Episode {ep:3d} | "
                f"Reward {total_reward:8.2f} | "
                f"Queue {queues[-1]:5.2f} | "
                f"Epsilon {agent.epsilon:.3f}"
            )

    torch.save(agent.q.state_dict(), "traffic_dqn_model.pth")

    evaluate(agent)
    graphs(rewards, queues, losses, eps)

    print("\nTraining completed.")
    print("Model: traffic_dqn_model.pth")


# ===================== EVALUATION =====================

def evaluate(agent, episodes=20):
    env = TrafficEnv()
    dqn_q, random_q = [], []

    for _ in range(episodes):
        state, _ = env.reset()
        total = 0
        done = False

        while not done:
            ns, _, _, done, info = env.step(
                agent.action(state, train=False)
            )
            total += info["queue"]
            state = ns

        dqn_q.append(total / 100)

    for _ in range(episodes):
        state, _ = env.reset()
        total = 0
        done = False

        while not done:
            ns, _, _, done, info = env.step(
                env.action_space.sample()
            )
            total += info["queue"]
            state = ns

        random_q.append(total / 100)

    dqn = np.mean(dqn_q)
    baseline = np.mean(random_q)
    improvement = (baseline - dqn) / baseline * 100

    print("\n========== FINAL ANALYSIS ==========")
    print(f"Random Policy Queue : {baseline:.2f}")
    print(f"DQN Queue           : {dqn:.2f}")
    print(f"Queue Reduction     : {improvement:.2f}%")

    return dqn, baseline


# ===================== GRAPHS =====================

def graphs(rewards, queues, losses, eps):
    x = range(1, len(rewards) + 1)

    # 1. Reward
    plt.figure(figsize=(8, 5))
    plt.plot(x, rewards, color="blue", linewidth=2)
    plt.title("DQN Training Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("reward.png", dpi=300)
    plt.show()

    # 2. Queue
    plt.figure(figsize=(8, 5))
    plt.plot(x, queues, color="red", linewidth=2)
    plt.title("Average Traffic Queue")
    plt.xlabel("Episode")
    plt.ylabel("Average Queue Length")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("queue.png", dpi=300)
    plt.show()

    # 3. Loss
    plt.figure(figsize=(8, 5))
    plt.plot(x, losses, color="green", linewidth=2)
    plt.title("DQN Training Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("loss.png", dpi=300)
    plt.show()

    # 4. Epsilon
    plt.figure(figsize=(8, 5))
    plt.plot(x, eps, color="purple", linewidth=2)
    plt.title("Epsilon Decay")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("epsilon.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    run()