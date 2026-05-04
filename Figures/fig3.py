import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class Config:
    delta_max: int = 40
    r_max: int = 3
    b_max: int = 5
    e_tx: int = 1
    e_s: int = 1
    p0: float = 0.5
    lam: float = 0.5
    p_e: float = 0.5
    horizon: int = 20_000
    seed: int = 0

    def g(self, r: int) -> float:
        return self.p0 * (self.lam ** r)

IDLE = 0
NEW = 1
RETX = 2
ALL_ACTIONS = [IDLE, NEW, RETX]

class AoIEnv:
    def __init__(self, cfg: Config, seed: int = 0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.state = None
    def reset(self, init_state=None):
        if init_state is None:
            init_state = (0, 0, self.cfg.delta_max, self.cfg.delta_max, 0)
        self.state = init_state
        return self.state
    def feasible_actions(self, s):
        _, b, _, _, r = s
        acts = [IDLE]
        if b >= self.cfg.e_s + self.cfg.e_tx:
            acts.append(NEW)
        if r > 0 and b >= self.cfg.e_tx:
            acts.append(RETX)
        return acts
    def step(self, action):
        e, b, drx, dtx, r = self.state
        if action not in self.feasible_actions(self.state):
            raise ValueError(f"Infeasible action {action} in state {self.state}")
        if action == IDLE:
            spend = 0
        elif action == NEW:
            spend = self.cfg.e_s + self.cfg.e_tx
        else:
            spend = self.cfg.e_tx
        b_next = min(max(b - spend + e, 0), self.cfg.b_max)
        if action == IDLE:
            success = False
        elif action == NEW:
            success = self.rng.random() > self.cfg.g(0)
        else:
            success = self.rng.random() > self.cfg.g(r)
        dtx_next = 1 if action == NEW else min(dtx + 1, self.cfg.delta_max)
        if action == IDLE or not success:
            drx_next = min(drx + 1, self.cfg.delta_max)
        elif action == NEW:
            drx_next = 1
        else:
            drx_next = dtx_next

        if success:
            r_next = 0
        elif action == NEW:
            r_next = 1
        elif action == IDLE:
            r_next = r
        else:
            r_next = min(r + 1, self.cfg.r_max)
        e_next = 1 if self.rng.random() < self.cfg.p_e else 0
        cost = drx
        self.state = (e_next, b_next, drx_next, dtx_next, r_next)
        return self.state, cost

class AoIMDP:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.states = []
        self.state_to_idx = {}
        self.actions = []
        self.cost = None
        self.trans_idx = []
        self.trans_prob = []
        self._build()
    def _cap(self, x):
        return min(x, self.cfg.delta_max)
    def feasible_actions(self, s):
        _, b, _, _, r = s
        acts = [IDLE]
        if b >= self.cfg.e_s + self.cfg.e_tx:
            acts.append(NEW)
        if r > 0 and b >= self.cfg.e_tx:
            acts.append(RETX)
        return acts
    def _next_outcomes(self, s, a):
        e, b, drx, dtx, r = s
        if a == IDLE:
            spend = 0
        elif a == NEW:
            spend = self.cfg.e_s + self.cfg.e_tx
        else:
            spend = self.cfg.e_tx
        b2 = min(max(b - spend + e, 0), self.cfg.b_max)
        out = {}
        def add(ns, p):
            out[ns] = out.get(ns, 0.0) + p
        def split_eh(drx2, dtx2, r2, p):
            add((0, b2, drx2, dtx2, r2), p * (1.0 - self.cfg.p_e))
            add((1, b2, drx2, dtx2, r2), p * self.cfg.p_e)
        if a == IDLE:
            split_eh(self._cap(drx + 1), self._cap(dtx + 1), r, 1.0)
        elif a == NEW:
            pf = self.cfg.g(0)
            split_eh(1, 1, 0, 1.0 - pf)
            split_eh(self._cap(drx + 1), 1, 1, pf)
        elif a == RETX:
            pf = self.cfg.g(r)
            dtx2 = self._cap(dtx + 1)
            split_eh(dtx2, dtx2, 0, 1.0 - pf)
            split_eh(self._cap(drx + 1), dtx2, min(r + 1, self.cfg.r_max), pf)
        return list(out.items())

    def _build(self):
        for e in (0, 1):
            for b in range(self.cfg.b_max + 1):
                for drx in range(1, self.cfg.delta_max + 1):
                    for dtx in range(1, self.cfg.delta_max + 1):
                        for r in range(self.cfg.r_max + 1):
                            s = (e, b, drx, dtx, r)
                            self.state_to_idx[s] = len(self.states)
                            self.states.append(s)
        n = len(self.states)
        self.cost = np.zeros(n)
        for i, s in enumerate(self.states):
            self.cost[i] = s[2]
            acts = self.feasible_actions(s)
            self.actions.append(acts)
            idx_dict = {}
            prob_dict = {}
            for a in acts:
                outcomes = self._next_outcomes(s, a)
                idxs = np.array([self.state_to_idx[ns] for ns, _ in outcomes], dtype=np.int32)
                probs = np.array([p for _, p in outcomes], dtype=np.float64)
                idx_dict[a] = idxs
                prob_dict[a] = probs
            self.trans_idx.append(idx_dict)
            self.trans_prob.append(prob_dict)
    def rvi(self, max_iter=500, tol=1e-8, verbose=True):
        n = len(self.states)
        h = np.zeros(n)
        policy = np.zeros(n, dtype=np.int8)
        ref_state = (0, 0, self.cfg.delta_max, self.cfg.delta_max, 0)
        ref = self.state_to_idx[ref_state]
        gain = None
        for it in range(max_iter):
            u = np.empty(n)
            for i in range(n):
                c = self.cost[i]
                best_q = np.inf
                best_a = IDLE
                for a in self.actions[i]:
                    q = c + np.dot(self.trans_prob[i][a], h[self.trans_idx[i][a]])
                    if q < best_q:
                        best_q = q
                        best_a = a
                u[i] = best_q
                policy[i] = best_a
            gain = u[ref]
            h_new = u - gain
            err = np.max(np.abs(h_new - h))
            if verbose and (it % 10 == 0 or err < tol):
                print(f"RVI iter={it:4d} err={err:.3e} avgAoI~={gain:.6f}")
            h = h_new
            if err < tol:
                break
        return gain, h, policy

def rvi_policy_fn(mdp: AoIMDP, policy_array):
    def policy(s):
        return int(policy_array[mdp.state_to_idx[s]])
    return policy
def greedy_policy(s, cfg: Config):
    _, b, _, _, r = s
    if b >= cfg.e_s + cfg.e_tx:
        return NEW
    elif b >= cfg.e_tx and r > 0:
        return RETX
    else:
        return IDLE

def simulate_policy_running_average(env, policy_fn, horizon=20_000, init_state=None):
    s = env.reset(init_state=init_state)
    costs = np.zeros(horizon)
    for t in range(horizon):
        a = policy_fn(s)
        s, c = env.step(a)
        costs[t] = c
    return np.cumsum(costs) / (np.arange(horizon) + 1)

def simulate_policy_running_average_multi(cfg, policy_fn_builder, horizon, init_state, seeds):
    curves = []
    for sd in seeds:
        curves.append(
            simulate_policy_running_average(
                AoIEnv(cfg, seed=sd),
                policy_fn_builder(),
                horizon=horizon,
                init_state=init_state,
            )
        )
    return np.mean(np.stack(curves, axis=0), axis=0)

def estimate_stationary_average(env, policy_fn, total_steps=200_000, burn_in=50_000, init_state=None):
    s = env.reset(init_state=init_state)
    costs = np.zeros(total_steps)
    for t in range(total_steps):
        a = policy_fn(s)
        s, c = env.step(a)
        costs[t] = c
    return float(np.mean(costs[burn_in:]))

def softmax_action(q_values, feasible_actions, tau, rng):
    vals = np.array([q_values[a] for a in feasible_actions], dtype=np.float64)
    logits = -vals / max(tau, 1e-8)
    logits -= np.max(logits)
    probs = np.exp(logits)
    probs /= np.sum(probs)
    return int(rng.choice(feasible_actions, p=probs))

def simulate_gr_learning(
    cfg: Config,
    mdp: AoIMDP,
    horizon=20_000,
    seed=0,
    init_state=None,
    alpha_a=0.9,
    alpha_exp=0.60,
    beta_a=0.10,
    beta_exp=0.90,
    tau0=10.0,
    tau_decay=0.99975,
    q_init=0.0,
):
    rng = np.random.default_rng(seed)
    env = AoIEnv(cfg, seed=seed)
    if init_state is None:
        init_state = (0, 0, cfg.delta_max, cfg.delta_max, 0)
    s = env.reset(init_state=init_state)
    n_states = len(mdp.states)
    Q = np.ones((n_states, 3), dtype=np.float64) * q_init
    visits = np.zeros((n_states, 3), dtype=np.int32)
    for i, st in enumerate(mdp.states):
        feasible = set(mdp.feasible_actions(st))
        for a in ALL_ACTIONS:
            if a not in feasible:
                Q[i, a] = 1e9
    J = float(cfg.delta_max)
    costs = np.zeros(horizon)
    tau = tau0
    s_idx = mdp.state_to_idx[s]
    a = softmax_action(Q[s_idx], mdp.feasible_actions(s), tau, rng)
    for t in range(horizon):
        s_next, cost = env.step(a)
        costs[t] = cost
        i = mdp.state_to_idx[s]
        j = mdp.state_to_idx[s_next]
        tau = max(0.05, tau * tau_decay)
        a_next = softmax_action(Q[j], mdp.feasible_actions(s_next), tau, rng)
        visits[i, a] += 1
        m = visits[i, a]
        alpha = alpha_a / (m ** alpha_exp)
        beta = beta_a / ((t + 1) ** beta_exp)
        Q[i, a] += alpha * (cost - J + Q[j, a_next] - Q[i, a])
        J += beta * ((((t * J) + cost) / (t + 1)) - J)
        s = s_next
        a = a_next
    return np.cumsum(costs) / (np.arange(horizon) + 1)

class ThresholdPolicy:
    def __init__(self, cfg: Config, tau_sigmoid=0.18, theta_init=4.2):
        self.cfg = cfg
        self.tau_sigmoid = tau_sigmoid
        self.theta_init = theta_init
        shape = (2, cfg.b_max + 1, cfg.delta_max + 1, cfg.r_max + 1)
        self.theta = np.full(shape, theta_init)
        self._enforce_feasibility()
    def copy(self):
        other = ThresholdPolicy(
            self.cfg,
            tau_sigmoid=self.tau_sigmoid,
            theta_init=self.theta_init,
        )
        other.theta = self.theta.copy()
        return other
    def _enforce_feasibility(self):
        for e in (0, 1):
            for b in range(self.cfg.b_max + 1):
                for dtx in range(1, self.cfg.delta_max + 1):
                    for r in range(self.cfg.r_max + 1):
                        if r == 0 and b < self.cfg.e_s + self.cfg.e_tx:
                            self.theta[e, b, dtx, r] = self.cfg.delta_max + 1
                        elif r > 0 and b < self.cfg.e_tx:
                            self.theta[e, b, dtx, r] = self.cfg.delta_max + 1

    def transmit_probability(self, s):
        e, b, drx, dtx, r = s
        th = self.theta[e, b, dtx, r]
        z = (drx - th) / max(self.tau_sigmoid, 1e-8)
        return 1.0 / (1.0 + np.exp(-z))
    def sample_action(self, s, rng):
        _, b, _, _, r = s
        feasible = [IDLE]
        if b >= self.cfg.e_s + self.cfg.e_tx:
            feasible.append(NEW)
        if r > 0 and b >= self.cfg.e_tx:
            feasible.append(RETX)
        p = self.transmit_probability(s)
        if r == 0:
            return NEW if (NEW in feasible and rng.random() < p) else IDLE
        return RETX if (RETX in feasible and rng.random() < p) else IDLE

def rollout_mean_cost(cfg, policy, horizon, seed, init_state, burn_in=500):
    env = AoIEnv(cfg, seed=seed)
    rng = np.random.default_rng(seed + 12345)
    s = env.reset(init_state)
    costs = []
    for t in range(horizon + burn_in):
        a = policy.sample_action(s, rng)
        s, c = env.step(a)
        if t >= burn_in:
            costs.append(c)
    return float(np.mean(costs))

def simulate_policy_gradient(
    cfg: Config,
    horizon=20_000,
    seed=0,
    update_every=1000,
    rollout_horizon=6000,
    step_size0=0.035,
    step_exp=0.80,
    sigma=0.02,
    tau_sigmoid=0.18,
    theta_init=4.2,
    warmup_steps=150,
):
    pg_display_init_state = (0, 2, 10, 10, 0)
    pg_rollout_init_state = (0, 2, 10, 10, 0)
    policy = ThresholdPolicy(cfg, tau_sigmoid=tau_sigmoid, theta_init=theta_init)
    env = AoIEnv(cfg, seed=seed)
    rng = np.random.default_rng(seed + 999)
    s = env.reset(init_state=pg_display_init_state)
    for _ in range(warmup_steps):
        a = policy.sample_action(s, rng)
        s, _ = env.step(a)
    costs = np.zeros(horizon)
    for t in range(horizon):
        a = policy.sample_action(s, rng)
        s, c = env.step(a)
        costs[t] = c
        if (t + 1) % update_every == 0:
            step_idx = (t + 1) // update_every
            step_size = step_size0 / (step_idx ** step_exp)
            D = rng.choice([-1.0, 1.0], size=policy.theta.shape)
            plus = policy.copy()
            minus = policy.copy()
            plus.theta = np.clip(policy.theta + sigma * D, 1.0, cfg.delta_max + 1.0)
            minus.theta = np.clip(policy.theta - sigma * D, 1.0, cfg.delta_max + 1.0)
            plus._enforce_feasibility()
            minus._enforce_feasibility()
            J_plus = rollout_mean_cost(
                cfg,
                plus,
                rollout_horizon,
                seed + 1000 + step_idx,
                pg_rollout_init_state,
            )
            J_minus = rollout_mean_cost(
                cfg,
                minus,
                rollout_horizon,
                seed + 2000 + step_idx,
                pg_rollout_init_state,
            )
            grad_est = D * ((J_plus - J_minus) / (2.0 * sigma))
            policy.theta = np.clip(
                policy.theta - step_size * grad_est,
                1.0,
                cfg.delta_max + 1.0,
            )
            policy._enforce_feasibility()
    return np.cumsum(costs) / (np.arange(horizon) + 1)

def simulate_policy_gradient_multi(cfg, horizon, seeds, **kwargs):
    curves = []
    for sd in seeds:
        curves.append(
            simulate_policy_gradient(
                cfg,
                horizon=horizon,
                seed=sd,
                **kwargs,
            )
        )
    return np.mean(np.stack(curves, axis=0), axis=0)

def reproduce_figure3(
    horizon=20_000,
    init_state=(0, 0, 40, 40, 0),
    seed=0,
):
    cfg = Config(horizon=horizon, seed=seed)
    print("Building MDP and solving RVI...")
    mdp = AoIMDP(cfg)
    g_star, _, pol = mdp.rvi(max_iter=500, tol=1e-8, verbose=True)
    print("Estimating RVI baseline...")
    rvi_ss = estimate_stationary_average(
        AoIEnv(cfg, seed=seed + 10),
        rvi_policy_fn(mdp, pol),
        total_steps=200_000,
        burn_in=50_000,
        init_state=init_state,
    )
    print("Running Greedy policy...")
    greedy_avg = simulate_policy_running_average_multi(
        cfg,
        policy_fn_builder=lambda: (lambda s: greedy_policy(s, cfg)),
        horizon=horizon,
        init_state=(0, 2, 5, 5, 0),
        seeds=list(range(seed + 20, seed + 30)),
    )
    print("Running GR-learning...")
    gr_avg = simulate_gr_learning(
        cfg,
        mdp,
        horizon=horizon,
        seed=seed + 2,
        init_state=init_state,
        alpha_a=0.9,
        alpha_exp=0.60,
        beta_a=0.10,
        beta_exp=0.90,
        tau0=10.0,
        tau_decay=0.99975,
        q_init=0.0,
    )
    print("Running Policy Gradient...")
    pg_avg = simulate_policy_gradient_multi(
        cfg,
        horizon=horizon,
        seeds=[seed + 30, seed + 31, seed + 32, seed + 33, seed + 34],
        update_every=1000,
        rollout_horizon=6000,
        step_size0=0.035,
        step_exp=0.80,
        sigma=0.02,
        tau_sigmoid=0.18,
        theta_init=4.2,
        warmup_steps=150,
    )
    x = np.arange(1, horizon + 1)
    x_plot = np.concatenate(([0], x))
    gr_plot = np.concatenate(([1.0], gr_avg))
    pg_plot = np.concatenate(([5.0], pg_avg))
    greedy_plot = np.concatenate(([4.2], greedy_avg))
    plt.figure(figsize=(8, 5))
    plt.plot(x_plot, pg_plot, color="magenta", label="RL, policy gradient")
    plt.axhline(rvi_ss, color="blue", linestyle="--", label="RVI")
    plt.plot(x_plot, gr_plot, color="red", label="RL, GR-learning")
    plt.plot(x_plot, greedy_plot, color="green", linestyle=":", label="Greedy policy")
    plt.xlabel("Time steps")
    plt.ylabel("Average AoI")
    plt.title("Figure 3 reproduction")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0, horizon)
    plt.ylim(0, 10)
    plt.tight_layout()
    plt.show()
    print("\nFinal values:")
    print(f"RVI gain               : {g_star:.6f}")
    print(f"RVI baseline           : {rvi_ss:.6f}")
    print(f"Greedy final avg       : {greedy_avg[-1]:.6f}")
    print(f"GR-learning final avg  : {gr_avg[-1]:.6f}")
    print(f"Policy gradient final  : {pg_avg[-1]:.6f}")

    return {
        "cfg": cfg,
        "rvi_gain": g_star,
        "rvi_ss": rvi_ss,
        "greedy_avg": greedy_avg,
        "gr_avg": gr_avg,
        "pg_avg": pg_avg,
    }

if __name__ == "__main__":
    results = reproduce_figure3(
        horizon=20_000,
        init_state=(0, 0, 40, 40, 0),
        seed=0,
    )
