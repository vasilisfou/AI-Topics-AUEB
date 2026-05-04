#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, math, os
from dataclasses import dataclass
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np

State = Tuple[int, int, int, int, int]


@dataclass
class Params:
    b_max: int = 5
    r_max: int = 3
    delta_max: int = 40
    e_s: int = 1
    e_tx: int = 1
    p0: float = 0.5
    lam: float = 0.5
    def g(self, r: int) -> float:
        return self.p0 * (self.lam ** r)

class EHEnv:
    def __init__(self, rho: float, params: Params, rng: np.random.Generator):
        self.rho = float(rho)
        self.params = params
        self.rng = rng
        self.p_stay = (1.0 + self.rho) / 2.0
        self.state: State | None = None
    def reset(self, state: State | None = None) -> State:
        if state is None:
            e0 = int(self.rng.random() < 0.5)
            self.state = (e0, self.params.b_max, 1, 1, 0)
        else:
            self.state = state
        return self.state
    def step(self, action: str) -> Tuple[State, int]:
        assert self.state is not None
        self.state, cost = simulate_step(self.state, action, self.params, self.rng, self.p_stay)
        return self.state, cost

def feasible_actions(s: State, params: Params) -> List[str]:
    _, b, _, _, r = s
    acts = ["i"]
    if b >= params.e_s + params.e_tx:
        acts.append("n")
    if r > 0 and b >= params.e_tx:
        acts.append("x")
    return acts

def simulate_step(s: State, action: str, params: Params, rng: np.random.Generator, p_stay: float) -> Tuple[State, int]:
    e, b, d_rx, d_tx, r = s
    if action not in feasible_actions(s, params):
        action = "i"
    cost = d_rx
    if action == "i":
        b_after = b
        d_tx_next = min(d_tx + 1, params.delta_max)
        d_rx_next = min(d_rx + 1, params.delta_max)
        r_next = r
    elif action == "n":
        b_after = b - params.e_s - params.e_tx
        success = rng.random() >= params.g(0)
        d_tx_next = 1
        if success:
            d_rx_next = 1
            r_next = 0
        else:
            d_rx_next = min(d_rx + 1, params.delta_max)
            r_next = 1
    else:
        b_after = b - params.e_tx
        success = rng.random() >= params.g(r)
        d_tx_next = min(d_tx + 1, params.delta_max)
        if success:
            d_rx_next = min(d_tx_next, params.delta_max)
            r_next = 0
        else:
            d_rx_next = min(d_rx + 1, params.delta_max)
            r_next = min(r + 1, params.r_max)
    e_next = e if rng.random() < p_stay else 1 - e
    b_next = min(b_after + e_next, params.b_max)
    return (e_next, b_next, d_rx_next, d_tx_next, r_next), cost


def expected_transitions(s: State, action: str, params: Params, rho: float):
    e, b, d_rx, d_tx, r = s
    p_stay = (1.0 + rho) / 2.0
    if action not in feasible_actions(s, params):
        action = "i"
    cost = d_rx
    out = []
    if action == "i":
        branches = [(1.0, False)]
        b_after = b
        d_tx_next = min(d_tx + 1, params.delta_max)
        r_fail = r
    elif action == "n":
        branches = [(1.0 - params.g(0), True), (params.g(0), False)]
        b_after = b - params.e_s - params.e_tx
        d_tx_next = 1
        r_fail = 1
    else:
        branches = [(1.0 - params.g(r), True), (params.g(r), False)]
        b_after = b - params.e_tx
        d_tx_next = min(d_tx + 1, params.delta_max)
        r_fail = min(r + 1, params.r_max)
    for p_succ, success in branches:
        if action == "i":
            d_rx_base, r_base = min(d_rx + 1, params.delta_max), r
        elif action == "n":
            if success:
                d_rx_base, r_base = 1, 0
            else:
                d_rx_base, r_base = min(d_rx + 1, params.delta_max), r_fail
        else:
            if success:
                d_rx_base, r_base = min(d_tx_next, params.delta_max), 0
            else:
                d_rx_base, r_base = min(d_rx + 1, params.delta_max), r_fail
        for e_next, p_e in [(e, p_stay), (1 - e, 1 - p_stay)]:
            b_next = min(b_after + e_next, params.b_max)
            out.append((p_succ * p_e, (e_next, b_next, d_rx_base, d_tx_next, r_base), cost))
    return out

def enumerate_states(params: Params) -> List[State]:
    return [
        (e, b, d_rx, d_tx, r)
        for e in [0, 1]
        for b in range(params.b_max + 1)
        for d_rx in range(1, params.delta_max + 1)
        for d_tx in range(1, params.delta_max + 1)
        for r in range(params.r_max + 1)
    ]

def _rvi(params: Params, rho: float, tol: float = 1e-8, max_iter: int = 500):
    states = enumerate_states(params)
    idx = {s: i for i, s in enumerate(states)}
    n = len(states)
    h = np.zeros(n)
    ref_i = 0
    transitions: Dict[Tuple[int, str], List[Tuple[float, int, int]]] = {}
    feasible_by_i: Dict[int, List[str]] = {}
    for i, s in enumerate(states):
        feasible_by_i[i] = feasible_actions(s, params)
        for a in feasible_by_i[i]:
            transitions[(i, a)] = [(p, idx[s2], c) for p, s2, c in expected_transitions(s, a, params, rho)]
    for _ in range(max_iter):
        v = np.empty(n)
        for i in range(n):
            best = math.inf
            for a in feasible_by_i[i]:
                q = sum(p * (c + h[j]) for p, j, c in transitions[(i, a)])
                best = min(best, q)
            v[i] = best
        v -= v[ref_i]
        if np.max(np.abs(v - h)) < tol:
            h = v
            break
        h = v
    return states, feasible_by_i, transitions, h

def rvi_optimal_average_cost(params: Params, rho: float) -> float:
    states, feasible_by_i, transitions, h = _rvi(params, rho)
    ref_i = 0
    return min(sum(p * (c + h[j] - h[ref_i]) for p, j, c in transitions[(ref_i, a)]) for a in feasible_by_i[ref_i])

def compute_rvi_policy(params: Params, rho: float) -> Dict[State, str]:
    states, feasible_by_i, transitions, h = _rvi(params, rho)
    policy = {}
    for i, s in enumerate(states):
        best_a, best_q = None, math.inf
        for a in feasible_by_i[i]:
            q = sum(p * (c + h[j]) for p, j, c in transitions[(i, a)])
            if q < best_q:
                best_q, best_a = q, a
        policy[s] = best_a
    return policy

class GreedyPolicy:
    def __init__(self, params: Params): self.params = params
    def act(self, s: State) -> str:
        _, b, _, _, r = s
        if b >= self.params.e_s + self.params.e_tx: return "n"
        if r > 0 and b >= self.params.e_tx: return "x"
        return "i"

class GRLearningPolicy:
    def __init__(self, params: Params, rng: np.random.Generator):
        self.params, self.rng = params, rng
        self.Q: Dict[Tuple[State, str], float] = {}
        self.N: Dict[Tuple[State, str], int] = {}
        self.J = 0.0
        self.n = 0
    def q(self, s: State, a: str) -> float:
        return self.Q.get((s, a), float(s[2]))
    def choose_action(self, s: State) -> str:
        tau = max(0.02, 1.5 * (0.99985 ** self.n))
        acts = feasible_actions(s, self.params)
        vals = np.array([-self.q(s, a) / tau for a in acts])
        vals -= vals.max()
        probs = np.exp(vals); probs /= probs.sum()
        return acts[int(self.rng.choice(len(acts), p=probs))]
    def update(self, s: State, a: str, cost: float, s2: State, a2: str) -> None:
        key = (s, a)
        self.N[key] = self.N.get(key, 0) + 1
        m = self.N[key]
        alpha = 0.55 / (m ** 0.55)
        beta = 0.03 / ((self.n + 1) ** 0.8)
        self.Q[key] = self.q(s, a) + alpha * (cost - self.J + self.q(s2, a2) - self.q(s, a))
        self.J += beta * (cost - self.J)
        self.n += 1

class ThresholdPolicyGradient:
    def __init__(self, rho: float, params: Params, seed: int = 0):
        self.rho, self.params = rho, params
        self.rng = np.random.default_rng(seed)
        self.theta = np.full((2, params.b_max + 1, params.delta_max + 1, params.r_max + 1), 4.0)
        self.tau = 0.35
        self.mask = np.ones_like(self.theta, dtype=bool)
        self.mask[:, :, 0, :] = False
        for e in range(2):
            for b in range(params.b_max + 1):
                for dtx in range(1, params.delta_max + 1):
                    for r in range(params.r_max + 1):
                        if (r == 0 and b < params.e_s + params.e_tx) or (r > 0 and b < params.e_tx):
                            self.theta[e, b, dtx, r] = params.delta_max + 1.0
                            self.mask[e, b, dtx, r] = False
                        else:
                            base = 4.5 - 0.5 * e - 0.35 * b + 0.2 * r + 0.02 * dtx
                            self.theta[e, b, dtx, r] = np.clip(base, 1.0, params.delta_max)
    def prob(self, s: State, theta: np.ndarray | None = None) -> float:
        if theta is None: theta = self.theta
        e, b, d_rx, d_tx, r = s
        if (r == 0 and b < self.params.e_s + self.params.e_tx) or (r > 0 and b < self.params.e_tx):
            return 0.0
        z = np.clip((d_rx - theta[e, b, d_tx, r]) / self.tau, -50, 50)
        return 1 / (1 + np.exp(-z))
    def sample_action(self, s: State, theta: np.ndarray | None = None, rng: np.random.Generator | None = None) -> str:
        if rng is None: rng = self.rng
        p = self.prob(s, theta)
        if rng.random() < p:
            return "n" if s[4] == 0 else "x"
        return "i"
    def deterministic_act(self, s: State) -> str:
        if self.prob(s) >= 0.5:
            return "n" if s[4] == 0 else "x"
        return "i"
    def rollout_cost(self, theta: np.ndarray, steps: int, seed: int) -> float:
        rng = np.random.default_rng(seed)
        env = EHEnv(self.rho, self.params, rng)
        s = env.reset()
        total = 0.0
        burn = min(1000, steps // 5)
        for t in range(steps + burn):
            a = self.sample_action(s, theta, rng)
            s, c = env.step(a)
            if t >= burn: total += c
        return total / steps
    def train(self, iters: int = 140, roll_steps: int = 3000, reps: int = 4) -> None:
        for n in range(iters):
            grad = np.zeros_like(self.theta)
            sigma = 0.08
            for rep in range(reps):
                D = self.rng.choice([-1.0, 1.0], size=self.theta.shape)
                D *= self.mask
                tp = np.clip(self.theta + sigma * D, 0.0, self.params.delta_max + 1.0)
                tm = np.clip(self.theta - sigma * D, 0.0, self.params.delta_max + 1.0)
                seed = 200000 + 1000 * n + rep
                jp = self.rollout_cost(tp, roll_steps, seed)
                jm = self.rollout_cost(tm, roll_steps, seed)  # common random numbers
                grad += D * ((jp - jm) / (2.0 * sigma))
            grad /= reps
            step = 0.12 / ((n + 5) ** 0.7)
            self.theta = np.clip(self.theta - step * grad, 0.0, self.params.delta_max + 1.0)
            self.theta[~self.mask] = self.params.delta_max + 1.0

def evaluate_policy(policy, rho: float, params: Params, steps: int, runs: int, seed: int) -> float:
    vals = []
    for k in range(runs):
        env = EHEnv(rho, params, np.random.default_rng(seed + k))
        s = env.reset(); total = 0.0
        for _ in range(steps):
            s, c = env.step(policy.act(s)); total += c
        vals.append(total / steps)
    return float(np.mean(vals))

def evaluate_gr_learning(rho: float, params: Params, steps: int, runs: int, seed: int) -> float:
    vals = []
    for k in range(runs):
        rng = np.random.default_rng(seed + k)
        env = EHEnv(rho, params, rng)
        learner = GRLearningPolicy(params, rng)
        s = env.reset(); a = learner.choose_action(s); total = 0.0
        for _ in range(steps):
            s2, c = env.step(a)
            a2 = learner.choose_action(s2)
            learner.update(s, a, c, s2, a2)
            total += c
            s, a = s2, a2
        vals.append(total / steps)
    return float(np.mean(vals))

def evaluate_policy_gradient(rho: float, params: Params, steps: int, runs: int, seed: int) -> float:
    vals = []
    for k in range(runs):
        pg = ThresholdPolicyGradient(rho, params, seed + k)
        pg.train()
        env = EHEnv(rho, params, np.random.default_rng(50000 + seed + k))
        s = env.reset(); total = 0.0
        for _ in range(steps):
            s, c = env.step(pg.deterministic_act(s)); total += c
        vals.append(total / steps)
    return float(np.mean(vals))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="/mnt/data")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    params = Params(); rhos = np.arange(0.0, 1.0, 0.1)
    greedy_vals, gr_vals, pg_vals, rvi_vals = [], [], [], []
    greedy = GreedyPolicy(params)
    for i, rho in enumerate(rhos):
        print(f"rho={rho:.1f}", flush=True)
        rvi_vals.append(rvi_optimal_average_cost(params, float(rho)))
        greedy_vals.append(evaluate_policy(greedy, float(rho), params, args.steps, args.runs, args.seed + 1000 * i))
        gr_vals.append(evaluate_gr_learning(float(rho), params, args.steps, max(20, args.runs // 2), args.seed + 2000 * i))
        pg_vals.append(evaluate_policy_gradient(float(rho), params, args.steps, max(10, args.runs // 10), args.seed + 3000 * i))
    csv_path = os.path.join(args.outdir, "figure6_reproduction_v2.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rho", "Greedy policy", "RL, GR-learning", "RL, policy gradient", "RVI"])
        for row in zip(rhos, greedy_vals, gr_vals, pg_vals, rvi_vals): w.writerow(row)
    plt.figure(figsize=(7, 5))
    plt.plot(rhos, greedy_vals, marker="o", label="Greedy policy")
    plt.plot(rhos, gr_vals, marker="o", label="RL, GR-learning")
    plt.plot(rhos, pg_vals, marker="o", label="RL, policy gradient")
    plt.plot(rhos, rvi_vals, marker="o", label="RVI")
    plt.xlabel(r"Correlation coefficient ($\rho$)")
    plt.ylabel("Average AoI")
    plt.legend(); plt.tight_layout()
    png_path = os.path.join(args.outdir, "figure6_reproduction_v2.png")
    plt.savefig(png_path, dpi=200)
    print(csv_path); print(png_path)

if __name__ == "__main__":
    main()
