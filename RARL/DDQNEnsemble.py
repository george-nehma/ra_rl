"""
DDQNEnsemble.py
---------------
Protagonist critic ensemble for reach-avoid RL epistemic uncertainty estimation.

Designed as a **drop-in replacement** for ``DDQNSingle`` in the protagonist role,
matching the exact constructor signature and method interface that ``Trainer``
expects. The adversary (``CriticEnsemble``) is left completely untouched.

Architecture
------------
* N independent ``DDQNSingle`` critics, all sharing the **same replay buffer**
  (``trainer.memory``), diversified by random weight-init seed only
  (seed_i = base_seed + i).
* A ``MeanEnsembleNet`` wraps all critic Q_networks into a single nn.Module
  exposed as ``self.Q_network`` / ``self.Q_target``. This lets ``Trainer``
  use the protagonist exactly as before for rollout, Bellman target
  computation, and evaluation.
* ``update_one_step`` updates **all** critics on the same batch. Each critic
  uses its own internal target network, so they diverge over time despite
  seeing identical data — pure seed diversification as requested.

Uncertainty outputs (``get_uncertainty``)
-----------------------------------------
    var_q                - per-action Q variance across critics
    min_q                - conservative (pessimistic) Q  [safety lower-bound]
    mean_q               - mean Q  (same as Q_network forward)
    epistemic_uncertainty-scalar per state: mean(var_q) over action dim
    safe_disagreement    - critic disagreement on safe/unsafe classification

SAC upgrade path
----------------
The adversary ``CriticEnsemble`` is untouched. When you convert to SAC, the
protagonist's discrete Q_network becomes a continuous actor + value head. At
that point, replace ``DDQNSingle`` inside the loop here with your SAC critic.

Authors: George Nehma (ensemble extension)
         Original DDQN/RARL: Kai-Chieh Hsu, Vicenç Rubies-Royo
"""

from __future__ import annotations

import copy
import os
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from RARL.DDQNSingle import DDQNSingle
from RARL.utils import save_obj


# ---------------------------------------------------------------------------
# MeanEnsembleNet — unified nn.Module for inference
# ---------------------------------------------------------------------------

class MeanEnsembleNet(nn.Module):
    """
    Wraps N networks and returns their mean output via ``forward(x)``.

    Registered as an ``nn.ModuleList`` so ``parameters()`` and ``.is_cuda``
    checks work exactly as they would on a plain ``Q_network``.
    """

    def __init__(self, networks: List[nn.Module]):
        super().__init__()
        self.networks = nn.ModuleList(networks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, A) — mean Q across all critics."""
        return torch.stack([net(x) for net in self.networks], dim=0).mean(dim=0)

    def stack(self, x: torch.Tensor) -> torch.Tensor:
        """(K, B, A) — raw per-critic outputs without averaging."""
        return torch.stack([net(x) for net in self.networks], dim=0)


# ---------------------------------------------------------------------------
# DDQNEnsemble
# ---------------------------------------------------------------------------

class DDQNEnsemble:
    """
    Protagonist critic ensemble — drop-in for ``DDQNSingle``.

    Parameters
    ----------
    config : ceConfig
        Ensemble config. ``config.NUM_CRITICS`` sets the ensemble size.
        Each critic gets a deep copy with ``SEED = config.SEED + i``.
    actionNum : int
        Cardinality of the discrete action space.
    memory : ReplayMemory
        Shared replay buffer from ``Trainer`` (pass ``trainer.memory``).
    dimList : list[int]
        ``[state_dim, *hidden_sizes, action_num]``.
    mode : str
        RL mode — ``'AARA'`` (default) or ``'RA'``.
    terminalType : str
        Terminal value convention — ``'max'`` (default) or ``'g'``.
    """

    def __init__(
        self,
        config,
        actionNum: int,
        memory,
        dimList: List[int],
        mode: str = 'AARA',
        terminalType: str = 'max',
    ):
        self.config       = config
        self.actionNum    = actionNum
        self.memory       = memory
        self.dimList      = dimList
        self.mode         = mode
        self.terminalType = terminalType
        self.num_critics  = config.NUM_CRITICS
        self.base_seed    = config.SEED

        # --- build N independent critics ------------------------------------
        self.critics: List[DDQNSingle] = []
        for i in range(self.num_critics):
            cfg_i      = copy.deepcopy(config)
            cfg_i.SEED = self.base_seed + i
            critic     = DDQNSingle(
                cfg_i, actionNum, memory,
                dimList=dimList, mode=mode, terminalType=terminalType,
            )
            self.critics.append(critic)
            print(
                f"  [Ensemble] Critic {i:02d} | seed={cfg_i.SEED}"
                f" | device={critic.device}"
            )

        self.device = self.critics[0].device

        # --- unified inference networks -------------------------------------
        # Trainer and simulate_one_trajectory see a single nn.Module here.
        self.Q_network = MeanEnsembleNet(
            [c.Q_network for c in self.critics]
        )
        self.Q_target = MeanEnsembleNet(
            [c.Q_target for c in self.critics]
        )

    # ------------------------------------------------------------------
    # Trainer-facing API  (mirrors DDQNSingle public interface)
    # ------------------------------------------------------------------

    def update_one_step(self, batch):
        """
        Update **all** critics on the same batch.

        Each critic's own target network computes its Bellman target
        independently, so critics diverge from their different inits.
        Returns mean loss across critics (compatible with Trainer logging).
        """
        losses = [c.update_one_step(batch) for c in self.critics]
        return float(np.mean(losses))

    def initQ(self, env, warmupIter, outFolder, **kwargs):
        """
        Supervised warmup for every critic. Each writes to its own sub-folder
        so checkpoints don't collide. Returns the last critic's loss array
        so the existing warmup-loss plot in sim_new_point_mass.py works
        without any changes.
        """
        all_losses = []
        for i, critic in enumerate(self.critics):
            sub = os.path.join(outFolder, f"critic_{i:02d}")
            os.makedirs(sub, exist_ok=True)
            all_losses.append(critic.initQ(env, warmupIter, sub, **kwargs))
        return all_losses[-1]

    def restore(self, num_updates: int, outFolder: str, prefix: str = "pro_") -> None:
        """
        Restore checkpoint for every critic.

        Prefixes are made unique per critic — e.g. with ``prefix="pro_"``:
            critic 0 → ``pro_critic_0_``
            critic 1 → ``pro_critic_1_``
        so files never collide with the adversary (``adv_``) or each other.
        """
        for i, critic in enumerate(self.critics):
            critic.restore(
                num_updates,
                outFolder,
                prefix=f"{prefix}critic_{i}_",
            )

    # ------------------------------------------------------------------
    # Attribute forwarding so Trainer can access anything it needs
    # ------------------------------------------------------------------

    @property
    def GAMMA(self):
        return self.critics[0].GAMMA

    @property
    def EPS(self):
        return self.critics[0].EPS

    @EPS.setter
    def EPS(self, value):
        # Keep all critics in sync on epsilon (exploration is shared)
        for c in self.critics:
            c.EPS = value

    def __getattr__(self, name: str):
        """
        Forward any attribute Trainer accesses that isn't explicitly
        defined here to critic 0, keeping backward compatibility with
        any DDQNSingle API surface we haven't wrapped.
        """
        if name == 'critics':
            raise AttributeError(name)
        return getattr(self.critics[0], name)

    # ------------------------------------------------------------------
    # Epistemic uncertainty  (new API — call these explicitly)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_uncertainty(self, state_tensor: torch.Tensor) -> dict:
        """
        Per-state epistemic uncertainty from the ensemble.

        Parameters
        ----------
        state_tensor : (B, state_dim) tensor

        Returns
        -------
        dict with keys:
            q_stack               (K, B, A)  raw per-critic Q values
            mean_q                (B, A)     mean Q
            var_q                 (B, A)     variance across critics
            std_q                 (B, A)     std dev
            min_q                 (B, A)     conservative (min) Q
            epistemic_uncertainty (B,)       mean(var_q) over action dim
            safe_disagreement     (B,)       disagreement on Q<0 sign
        """
        q_stack = self.Q_network.stack(state_tensor)  # (K, B, A)

        mean_q = q_stack.mean(dim=0)
        var_q  = q_stack.var(dim=0, unbiased=True)
        std_q  = q_stack.std(dim=0, unbiased=True)
        min_q  = q_stack.min(dim=0).values

        # Scalar per state: average variance over actions
        epistemic_uncertainty = var_q.mean(dim=-1)  # (B,)

        # Safe/unsafe disagreement on the min-action decision (RA convention)
        min_q_per_critic  = q_stack.min(dim=-1).values     # (K, B)
        safe_votes        = (min_q_per_critic < 0).float() # (K, B): 1 = critic says safe
        majority          = (safe_votes.mean(dim=0) >= 0.5).float()  # (B,)
        safe_disagreement = (
            (safe_votes - majority.unsqueeze(0)).abs().mean(dim=0)
        )  # (B,)

        return dict(
            q_stack               = q_stack,
            mean_q                = mean_q,
            var_q                 = var_q,
            std_q                 = std_q,
            min_q                 = min_q,
            epistemic_uncertainty = epistemic_uncertainty,
            safe_disagreement     = safe_disagreement,
        )

    @torch.no_grad()
    def get_uncertainty_map(self, env, nx: int = 41, ny: int = 41) -> dict:
        """
        Evaluate uncertainty across the 2-D state grid.

        Returns a dict of (nx, ny) numpy arrays:
            xs, ys, mean_v, var_v, std_v, min_v,
            epistemic_uncertainty, safe_disagreement
        """
        xs = np.linspace(
            env.unwrapped.bounds[0, 0], env.unwrapped.bounds[0, 1], nx
        )
        ys = np.linspace(
            env.unwrapped.bounds[1, 0], env.unwrapped.bounds[1, 1], ny
        )

        out = {k: np.empty((nx, ny)) for k in
               ("mean_v", "var_v", "std_v", "min_v",
                "epistemic_uncertainty", "safe_disagreement")}

        for ix, x in enumerate(xs):
            for iy, y in enumerate(ys):
                st = torch.FloatTensor([x, y]).unsqueeze(0).to(self.device)
                m  = self.get_uncertainty(st)
                out["mean_v"]              [ix, iy] = m["mean_q"].min().item()
                out["var_v"]               [ix, iy] = m["var_q"].min(dim=-1).values.item()
                out["std_v"]               [ix, iy] = m["std_q"].min(dim=-1).values.item()
                out["min_v"]               [ix, iy] = m["min_q"].min().item()
                out["epistemic_uncertainty"][ix, iy] = m["epistemic_uncertainty"].item()
                out["safe_disagreement"]   [ix, iy] = m["safe_disagreement"].item()

        out["xs"], out["ys"] = xs, ys
        return out

    def plot_uncertainty_maps(
        self,
        env,
        out_folder: str,
        nx: int = 41,
        ny: int = 41,
        vmin: float = -4.0,
        vmax: float = 4.0,
        store: bool = True,
        show: bool = False,
    ) -> None:
        """
        4-panel figure:
          mean Q | conservative min Q | epistemic variance | safe disagreement
        """
        print("\n  [Ensemble] Computing uncertainty maps ...")
        maps    = self.get_uncertainty_map(env, nx=nx, ny=ny)
        axStyle = env.unwrapped.get_axes()
        xs, ys  = maps["xs"], maps["ys"]

        panels = [
            ("mean_v",               "seismic",  vmin,  vmax, r"Mean $\hat{V}$ (protagonist ensemble)"),
            ("min_v",                "seismic",  vmin,  vmax, r"Conservative $\hat{V}$ (min critic)"),
            ("epistemic_uncertainty","YlOrRd",   None,  None, r"Epistemic Uncertainty  Var$_k[Q]$"),
            ("safe_disagreement",    "PuRd",     0,     1,    "Safe / Unsafe Disagreement"),
        ]

        fig, axes = plt.subplots(1, 4, figsize=(22, 5))
        for ax, (key, cmap, lo, hi, title) in zip(axes, panels):
            data = maps[key]
            im_kw = dict(interpolation='none', extent=axStyle[0],
                         origin='lower', cmap=cmap)
            if lo is not None:
                im_kw.update(vmin=lo, vmax=hi)
            im = ax.imshow(data.T, **im_kw)
            fig.colorbar(im, ax=ax, pad=0.01, fraction=0.05, shrink=0.9)
            ax.set_title(title, fontsize=11)
            env.unwrapped.plot_target_failure_set(ax=ax)
            env.unwrapped.plot_reach_avoid_set(ax=ax)
            env.unwrapped.plot_formatting(ax=ax)
            if key in ("mean_v", "min_v"):
                ax.contour(xs, ys, data.T, levels=[0],
                           colors='k', linewidths=2, linestyles='dashed')

        fig.suptitle(
            f"Protagonist Ensemble  ({self.num_critics} critics | seed diversification"
            f" | mode={self.mode} | terminalType={self.terminalType})",
            fontsize=13,
        )
        fig.tight_layout()
        if store:
            os.makedirs(out_folder, exist_ok=True)
            path = os.path.join(out_folder, "ensemble_uncertainty.png")
            fig.savefig(path, dpi=150)
            print(f"  [Ensemble] Saved → {path}")
        if show:
            plt.show()
        plt.close(fig)

    def save_metrics(self, metrics: dict, out_folder: str, tag: str = '') -> None:
        """Pickle an uncertainty metrics dict for later analysis."""
        os.makedirs(out_folder, exist_ok=True)
        path = os.path.join(out_folder, f"ensemble_metrics{tag}")
        save_obj(metrics, path)
        print(f"  [Ensemble] Metrics saved → {path}.pkl")

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.num_critics

    def __repr__(self) -> str:
        return (
            f"DDQNEnsemble(num_critics={self.num_critics}, "
            f"mode='{self.mode}', terminalType='{self.terminalType}', "
            f"base_seed={self.base_seed})"
        )