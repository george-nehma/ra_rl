"""
DDQNEnsemble.py
---------------
Protagonist critic ensemble — drop-in replacement for DDQNSingle in the
protagonist role. Designed to match the exact API that Trainer.py expects.

Diversification: random weight-init seeds only (seed_i = base_seed + i).
All critics share the same replay buffer (trainer.memory) so they see
identical data and diverge purely from their different initializations
(Deep Ensembles, Lakshminarayanan et al. 2017).

Trainer API surface honoured
-----------------------------
    protagonist.cntUpdate           – synced property across all critics
    protagonist.EPSILON             – forwarded to critics[0]
    protagonist.GAMMA               – forwarded to critics[0]
    protagonist.optimizer           – forwarded to critics[0]  (for lr logging)
    protagonist.select_action(s, env, agent=..., explore=...)
    protagonist.update(addBias=False)      – fans out, returns mean loss
    protagonist.updateHyperParam()         – fans out to all critics
    protagonist.save(cntUpdate, folder)    – saves each critic to folder/critic_i/
    protagonist.restore(n, folder, prefix) – restores each critic
    protagonist.initQ(env, n, folder, ...) – warms up each critic
    protagonist.Q_network                  – MeanEnsembleNet (nn.Module)
    protagonist.target_network             – MeanEnsembleNet (nn.Module)

SAC upgrade note
----------------
When converting to SAC, replace DDQNSingle inside the critic loop with a
SAC critic. The adversary (DDQNSingle) and Trainer are left unchanged.

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

from concurrent.futures import ThreadPoolExecutor, as_completed

from RARL.DDQNSingle import DDQNSingle
from RARL.utils import save_obj
from .DDQN import Transition


# ---------------------------------------------------------------------------
# MeanEnsembleNet
# ---------------------------------------------------------------------------

class MeanEnsembleNet(nn.Module):
    """
    Wraps N networks; forward(x) returns their mean output.

    Registered as nn.ModuleList so next(parameters()).is_cuda and any
    other nn.Module checks work identically to a plain Q_network.
    """

    def __init__(self, networks: List[nn.Module]):
        super().__init__()
        self.networks = nn.ModuleList(networks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, A) — mean Q across all critics."""
        return torch.stack([net(x) for net in self.networks], dim=0).mean(dim=0)

    def stack(self, x: torch.Tensor) -> torch.Tensor:
        """(K, B, A) — raw per-critic outputs, no averaging."""
        return torch.stack([net(x) for net in self.networks], dim=0)


# ---------------------------------------------------------------------------
# DDQNEnsemble
# ---------------------------------------------------------------------------

class DDQNEnsemble:
    """
    Protagonist critic ensemble — drop-in for DDQNSingle.

    Parameters
    ----------
    config : ceConfig
        config.NUM_CRITICS sets ensemble size.
        Each critic gets a deep copy with SEED = config.SEED + i.
    actionNum : int
    memory : ReplayMemory
        Shared buffer from Trainer (pass trainer.memory).
    dimList : list[int]
        [state_dim, *hidden, action_num]
    mode : str
        'AARA' (default) or 'RA'.
    terminalType : str
        'max' (default) or 'g'.
    """

    def __init__(
        self,
        config,
        actionNum: int,
        memory,
        dimList: List[int],
        mode: str = 'AARA',
        terminalType: str = 'max',
        n_workers: int = None,
    ):
        self.config       = config
        self.actionNum    = actionNum
        self.memory       = memory
        self.dimList      = dimList
        self.mode         = mode
        self.terminalType = terminalType
        self.num_critics  = config.NUM_CRITICS
        self.base_seed    = config.SEED
        self.n_workers    = n_workers if n_workers is not None else self.num_critics

        # Build N independent critics
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

        # Thread pool — one worker per critic, reused across all update() calls.
        # PyTorch releases the GIL during tensor ops so threads genuinely
        # run in parallel on separate cores.
        self._executor = ThreadPoolExecutor(max_workers=self.n_workers)

        # Unified inference modules — what Trainer / env see as Q_network
        self.Q_network = MeanEnsembleNet([c.Q_network for c in self.critics])
        self.Q_target  = MeanEnsembleNet([c.target_network  for c in self.critics])

    # ------------------------------------------------------------------
    # Trainer-facing API
    # ------------------------------------------------------------------

    # --- cntUpdate: must stay synced so Trainer's while-loop and
    #     per-critic updateHyperParam() schedules all stay aligned -------

    @property
    def cntUpdate(self):
        return self.critics[0].cntUpdate

    @cntUpdate.setter
    def cntUpdate(self, value):
        for c in self.critics:
            c.cntUpdate = value

    # --- epsilon / gamma: forward reads to critics[0]; keep all in sync
    #     on writes so exploration is consistent --------------------------

    @property
    def EPSILON(self):
        return self.critics[0].EPSILON

    @EPSILON.setter
    def EPSILON(self, value):
        for c in self.critics:
            c.EPSILON = value

    @property
    def GAMMA(self):
        return self.critics[0].GAMMA

    @GAMMA.setter
    def GAMMA(self, value):
        for c in self.critics:
            c.GAMMA = value

    def update(self, addBias: bool = False) -> float:
        """
        Update all critics on the same batch drawn from shared memory.
        Each critic uses its own target network for Bellman targets, so
        they diverge over time from their different random initialisations.
        Returns mean loss (compatible with Trainer's trainingRecords).
        """
        batch_size = self.critics[0].BATCH_SIZE
        if len(self.memory) < batch_size * 20:
            return
        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))
        (_, _, state, _, _, _,_) = self.unpack_batch(batch)
        epistem_uncertainty = self.get_uncertainty(state)["epistemic_uncertainty"].max().item()


        futures = {
            self._executor.submit(c.update, addBias, epistem_uncertainty, batch): c
            for c in self.critics
        }
        losses = []
        for fut in as_completed(futures):
            result = fut.result()
            if result is not None:
                losses.append(result)
        return (float(np.mean(losses)) if losses else 0.0), epistem_uncertainty


    def updateHyperParam(self):
        """Fan out hyper-parameter schedule (lr, eps, gamma) to all critics."""
        for c in self.critics:
            c.updateHyperParam()

    def save(self, cntUpdate: int, folder: str) -> None:
        """
        Save all critics. Trainer calls protagonist.save(cntUpdate, pro_modelFolder).
        Each critic saves to <folder>/critic_<i>/ so files never collide.
        """
        for i, c in enumerate(self.critics):
            sub = os.path.join(folder, f"critic_{i}")
            os.makedirs(sub, exist_ok=True)
            c.save(cntUpdate, sub)

    def restore(self, cntUpdate: int, outFolder: str, prefix: str = "pro_model") -> None:
        """
        Restore all critics. Called from sim script as:
            protagonist.restore(idx * checkPeriod, outFolder, prefix="pro_model")
        Each critic gets a unique per-critic prefix to avoid filename collisions.
        """
        for i, c in enumerate(self.critics):
            c.restore(cntUpdate, outFolder, prefix=f"{prefix}/critic_{i}")

    def initQ(self, env, warmupIter: int, outFolder: str, **kwargs):
        """
        Supervised Q warmup for every critic. Each writes to its own subfolder.
        Returns the last critic's loss array so the warmup-loss plot in the
        sim script continues to work without modification.
        """
        all_losses = []
        for i, c in enumerate(self.critics):
            sub = os.path.join(outFolder, f"critic_{i:02d}")
            os.makedirs(sub, exist_ok=True)
            all_losses.append(c.initQ(env, warmupIter, sub, **kwargs))
        return all_losses[-1]

    # ------------------------------------------------------------------
    # Generic attribute forwarding
    # ------------------------------------------------------------------

    def __getattr__(self, name: str):
        """
        Forward any attribute Trainer or the sim script accesses that
        isn't explicitly defined here (e.g. select_action, optimizer,
        device_type, etc.) to critics[0].
        """
        if name == 'critics':
            raise AttributeError(name)
        return getattr(self.critics[0], name)

    # ------------------------------------------------------------------
    # Epistemic uncertainty API (ensemble-only, call explicitly)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_uncertainty(self, state_tensor: torch.Tensor) -> dict:
        """
        Per-state epistemic uncertainty.

        Parameters
        ----------
        state_tensor : (B, state_dim) tensor on self.device

        Returns
        -------
        dict:
            q_stack               (K, B, A)
            mean_q                (B, A)
            var_q                 (B, A)   ← primary epistemic uncertainty signal
            std_q                 (B, A)
            min_q                 (B, A)   ← conservative (pessimistic) Q
            epistemic_uncertainty (B,)     mean(var_q) over action dim
            safe_disagreement     (B,)     critic disagreement on Q<0 (safe) sign
        """
        q_stack = self.Q_network.stack(state_tensor)   # (K, B, A)

        mean_q = q_stack.mean(dim=0)
        # var_q  = q_stack.var(dim=0, unbiased=True)
        std_q  = q_stack.std(dim=0, unbiased=True)
        min_q  = q_stack.min(dim=0).values

        # Biased variance — 1/M * sum(Q^2) - (1/M * sum(Q))^2
        var_q = q_stack.var(dim=0, unbiased=False)    # (B, A) — per state per action

        # Average variance across action dimension → scalar per state
        epistemic_uncertainty = var_q.mean(dim=-1)     # (B,)

        # Safe/unsafe disagreement on the min-action decision
        # In RA-RL: min_a Q(s,a) < 0 means the state is believed safe
        min_q_per_critic  = q_stack.min(dim=-1).values     # (K, B)
        safe_votes        = (min_q_per_critic < 0).float() # (K, B)
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
        Sweep the 2-D state grid and compute uncertainty at every cell.

        Returns dict of (nx, ny) numpy arrays:
            xs, ys, mean_v, var_v, std_v, min_v,
            epistemic_uncertainty, safe_disagreement
        """
        xs = np.linspace(
            env.unwrapped.bounds[0, 0], env.unwrapped.bounds[0, 1], nx
        )
        ys = np.linspace(
            env.unwrapped.bounds[1, 0], env.unwrapped.bounds[1, 1], ny
        )

        keys = ("mean_v", "var_v", "std_v", "min_v",
                "epistemic_uncertainty", "safe_disagreement")
        out  = {k: np.empty((nx, ny)) for k in keys}

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
        vmax: float =  4.0,
        store: bool = True,
        show:  bool = False,
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
            ("mean_v",               "seismic",  vmin,  vmax, r"Mean $\hat{V}$ (ensemble avg)"),
            ("min_v",                "seismic",  vmin,  vmax, r"Conservative $\hat{V}$ (min critic)"),
            ("epistemic_uncertainty","YlOrRd",   None,  None, r"Epistemic Uncertainty  Var$_k[Q]$"),
            ("safe_disagreement",    "PuRd",     0,     1,    "Safe / Unsafe Disagreement"),
        ]

        fig, axes = plt.subplots(1, 4, figsize=(22, 5))
        for ax, (key, cmap, lo, hi, title) in zip(axes, panels):
            data  = maps[key]
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
