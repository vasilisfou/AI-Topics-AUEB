from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

@dataclass
class Config:
    delta_max: int = 40
    r_max: int = 3
    b_max: int = 5
    e_tx: int = 1
    e_s: int = 1
    p0: float = 0.5
    lam: float = 0.5
    p11: float = 0.7
    p00: float = 0.7
    def g(self, r: int) -> float:
        return self.p0 * (self.lam ** r)
      
IDLE = 0
NEW = 1
RETX = 2

class CorrelatedEH_AoIMDP:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.states = []
        self.state_to_idx = {}
        self.actions = []
        self.cost = None
        self.trans_idx = []
        self.trans_prob = []
        self._build()
    def _cap(self, x: int) -> int:
        return min(x, self.cfg.delta_max)
    def _valid_state(self, e: int, b: int, drx: int, dtx: int, r: int) -> bool:
        return (
            e in (0, 1)
            and 0 <= b <= self.cfg.b_max
            and 1 <= drx <= self.cfg.delta_max
            and 1 <= dtx <= self.cfg.delta_max
            and 0 <= r <= self.cfg.r_max
        )
    def feasible_actions(self, s):
        e, b, drx, dtx, r = s
        acts = [IDLE]
        if b >= self.cfg.e_s + self.cfg.e_tx:
            acts.append(NEW)
        if r > 0 and b >= self.cfg.e_tx:
            acts.append(RETX)
        return acts
    def _battery_after_action(self, e: int, b: int, a: int) -> int:
        if a == IDLE:
            spend = 0
        elif a == NEW:
            spend = self.cfg.e_s + self.cfg.e_tx
        elif a == RETX:
            spend = self.cfg.e_tx
        else:
            raise ValueError
        return min(b - spend + e, self.cfg.b_max)
    def _eh_next_probs(self, e: int):
        if e == 1:
            return [(0, 1.0 - self.cfg.p11), (1, self.cfg.p11)]
        return [(0, self.cfg.p00), (1, 1.0 - self.cfg.p00)]
    def _next_outcomes(self, s, a):
        e, b, drx, dtx, r = s
        b2 = self._battery_after_action(e, b, a)
        out = {}
        def add(ns, p):
            out[ns] = out.get(ns, 0.0) + p
        def split_eh(drx2, dtx2, r2, p_base):
            for e2, p_eh in self._eh_next_probs(e):
                add((e2, b2, drx2, dtx2, r2), p_base * p_eh)
        if a == IDLE:
            split_eh(self._cap(drx + 1), self._cap(dtx + 1), r, 1.0)
        elif a == NEW:
            pf = self.cfg.g(0)
            ps = 1.0 - pf
            split_eh(1, 1, 0, ps)
            split_eh(self._cap(drx + 1), 1, 1, pf)
        elif a == RETX:
            pf = self.cfg.g(r)
            ps = 1.0 - pf
            dtx2 = self._cap(dtx + 1)
            split_eh(dtx2, dtx2, 0, ps)
            split_eh(self._cap(drx + 1), dtx2, min(r + 1, self.cfg.r_max), pf)
        else:
            raise ValueError
        return list(out.items())
    def _build(self):
        for e in (0, 1):
            for b in range(self.cfg.b_max + 1):
                for drx in range(1, self.cfg.delta_max + 1):
                    for dtx in range(1, self.cfg.delta_max + 1):
                        for r in range(self.cfg.r_max + 1):
                            if self._valid_state(e, b, drx, dtx, r):
                                s = (e, b, drx, dtx, r)
                                self.state_to_idx[s] = len(self.states)
                                self.states.append(s)
        n = len(self.states)
        self.cost = np.zeros(n, dtype=np.float64)
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
    def rvi(self, max_iter=1200, tol=1e-10, verbose=True):
        n = len(self.states)
        h = np.zeros(n, dtype=np.float64)
        policy = np.zeros(n, dtype=np.int8)
        ref = 0
        for it in range(max_iter):
            u = np.empty(n, dtype=np.float64)
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
            g = u[ref]
            h_new = u - g
            err = np.max(np.abs(h_new - h))
            if verbose and (it % 10 == 0 or err < tol):
                print(f"RVI iter={it:4d} err={err:.3e} avgAoI~={g:.6f}")
            h = h_new
            if err < tol:
                break
        return g, h, policy
    def policy_action(self, policy, e, b, drx, dtx, r):
        return int(policy[self.state_to_idx[(e, b, drx, dtx, r)]])

def reproduce_figure4_strict(verbose=True, save_path=None, dpi=300):
    cfg = Config()
    print("Building MDP...")
    mdp = CorrelatedEH_AoIMDP(cfg)
    print("Running RVI...")
    g, _, policy = mdp.rvi(max_iter=1200, tol=1e-10, verbose=verbose)
    print(f"Estimated optimal average AoI: {g:.6f}")
    maps = {}
    for e in [1, 0]:
        for r in range(cfg.r_max + 1):
            dtx = r + 1
            grid = np.zeros((cfg.delta_max, cfg.b_max + 1), dtype=np.int8)
            for drx in range(1, cfg.delta_max + 1):
                for b in range(cfg.b_max + 1):
                    grid[drx - 1, b] = mdp.policy_action(policy, e=e, b=b, drx=drx, dtx=dtx, r=r)
            maps[(e, r)] = grid
    color_idle = "#d0d0d0"
    color_new = "#2ca25f"
    color_retx = "#3182bd"
    cmap = ListedColormap([color_idle, color_new, color_retx])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    fig, axes = plt.subplots(2, 4, figsize=(14, 8), sharex=True, sharey=True)
    for row, e in enumerate([1, 0]):
        for col, r in enumerate([0, 1, 2, 3]):
            ax = axes[row, col]
            grid = maps[(e, r)]
            ax.imshow(
                grid,
                origin="lower",
                aspect="auto",
                cmap=cmap,
                norm=norm,
                interpolation="nearest",
                extent=[0, cfg.b_max, 1, cfg.delta_max],
            )
            ax.set_title(rf"$R_t={r}$", fontsize=12)
            if col == 0:
                ax.set_ylabel(r"$\Delta_t^{rx}$", fontsize=12)
            if row == 1:
                ax.set_xlabel(r"$B_t$", fontsize=12)
            ax.set_xticks(range(0, cfg.b_max + 1))
            ax.set_yticks([1, 10, 20, 30, 40])
    legend_handles = [
        Patch(facecolor=color_idle, edgecolor="black", label="i"),
        Patch(facecolor=color_new, edgecolor="black", label="n"),
        Patch(facecolor=color_retx, edgecolor="black", label="x"),
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=True,
        fontsize=12,
        bbox_to_anchor=(0.5, 0.02),
    )

    fig.suptitle(
        r"Optimal policy for $B_{\max}=5,\ R_{\max}=3,\ p_E(1,1)=p_E(0,0)=0.7,$"
        "\n"
        r"$E_s=E_{tx}=1$ and $\Delta_t^{tx}=R_t+1$",
        fontsize=13,
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0.06, 1, 0.93])
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()
    return {
        "avg_aoi": g,
        "maps": maps,
        "policy": policy,
    }

if __name__ == "__main__":
    results = reproduce_figure4_strict(
        verbose=True,
        save_path="figure4_strict.png",
        dpi=300,
    )
