"""
SAC.py  –  Soft Actor-Critic for two-player reach-avoid games.

Fixes applied vs. the original:
  1.  self.BATCH_SIZE / self.GAMMA / self.CONFIG were referenced but never
      set → assigned from config in __init__.
  2.  self.policy / self.policy_optim were referenced in update() but the
      actual attributes are self.protagonist / self.protagonist_optim.
      Added policy/policy_optim as aliases so existing code keeps working.
  3.  Added adversary policy-loss computation and optimisation step (the
      adversary *maximises* the Q-value, so its loss is negated).
  4.  update() now returns early gracefully (returns None) when the buffer
      is too small, matching how SACTrainer.learn() checks for None.
  5.  Corrected save/load_checkpoint to use protagonist (not self.policy).
"""

import os
from typing import List
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.optim import Adam
import torch.optim as optim
from .utils import soft_update, hard_update, save_model
from .model import GaussianPolicy, QNetwork, DeterministicPolicy, StepLRMargin, StepResetLR
from concurrent.futures import ThreadPoolExecutor, as_completed

from collections import namedtuple
Transition = namedtuple("Transition", ["s", "a", "d", "r", "s_", "a_", "info"])

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

    def forward(self, x, u, d: torch.Tensor) -> torch.Tensor:
        """(B, A) — mean Q across all critics."""
        qs = []
        for net in self.networks:
            q1, q2 = net(x, u, d)
            qs.append(torch.max(q1, q2))
        return torch.stack(qs, dim=0).mean(dim=0)

    def stack(self, x, u, d: torch.Tensor) -> torch.Tensor:
        """(K, B, A) — raw per-critic outputs, no averaging."""
        qs = []
        for net in self.networks:
            q1, q2 = net(x, u, d)
            qs.append(torch.max(q1, q2))
        return torch.stack(qs, dim=0)


class SAC(object):
    def __init__(self, config, dimList, action_space, disturbance_space, n_workers: int = None,):

        # ----------------------------------------------------------------
        # Hyper-parameters  (kept as both lower- and UPPER-case so that
        # existing code referencing either form continues to work)
        # ----------------------------------------------------------------
        self.CONFIG     = config         
        self.tau        = config.TAU
        self.alpha_pro  = config.ALPHA
        self.alpha_adv  = config.ALPHA
        self.BATCH_SIZE = config.BATCH_SIZE   

        # Learning rate of updating the Q-network
        self.LR_C = config.LR_C
        self.LR_C_PERIOD = config.LR_C_PERIOD
        self.LR_C_DECAY = config.LR_C_DECAY
        self.LR_C_END = config.LR_C_END  

        # Learning rate of updating the policy networks
        self.LR_A = config.LR_A
        self.LR_A_PERIOD = config.LR_A_PERIOD
        self.LR_A_DECAY = config.LR_A_DECAY
        self.LR_A_END = config.LR_A_END

        self.autoAlphaTuning = config.AUTO_ALPHA_TUNING

        self.num_critics = config.NUM_CRITICS

        self.n_workers    = n_workers if n_workers is not None else self.num_critics

        self.dimList    = dimList

        self.policy_type           = config.POLICY
        self.target_update_interval = config.TARGET_UPDATE_INTERVAL

        self.device = torch.device(config.DEVICE)

        if self.autoAlphaTuning:
            # Target entropy is -|A| (e.g. -2 for Ant-v2) as per SAC paper
            self.pro_target_entropy = -action_space.shape[0]
            self.pro_log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.pro_alpha_optim = optim.Adam([self.pro_log_alpha], lr=0.0003)

            self.adv_target_entropy = -disturbance_space.shape[0]
            self.adv_log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.adv_alpha_optim = optim.Adam([self.adv_log_alpha], lr=0.0003)

        # ----------------------------------------------------------------
        # Critics
        # ----------------------------------------------------------------
        self.critics: List[QNetwork] = []
        self.critic_optimisers: List[optim.AdamW] = []
        self.schedulers: List [optim.lr_scheduler.StepLR] = []
        self.critic_targets: List[QNetwork] = []

        for i in range(self.num_critics):
            cfg_i = copy.deepcopy(config)
            cfg_i.SEED += i
            self.critic = QNetwork(config, dimList, action_space.shape[0], disturbance_space.shape[0]).to(self.device)

            self.critics.append(self.critic)
            print(
                f"  [Ensemble] Critic {i:02d} | seed={cfg_i.SEED}"
                # f" | device={self.critic.device}"
            )

            self.critic_optim = optim.AdamW(self.critic.parameters(), lr=config.LR_C, weight_decay=1e-3)
            self.critic_optimisers.append(self.critic_optim)

            self.scheduler = optim.lr_scheduler.StepLR(
                self.critic_optim, step_size=self.LR_C_PERIOD, gamma=self.LR_C_DECAY
            )
            self.schedulers.append(self.scheduler)

            self.critic_target = QNetwork(config, dimList, action_space.shape[0], disturbance_space.shape[0]).to(self.device)
            hard_update(self.critic_target, self.critic)
            self.critic_targets.append(self.critic_target)

        self.max_grad_norm = 1
        self.cntUpdate = 0

        # Thread pool — one worker per critic, reused across all update() calls.
        # PyTorch releases the GIL during tensor ops so threads genuinely
        # run in parallel on separate cores.
        self._executor = ThreadPoolExecutor(max_workers=self.n_workers)

        # Unified inference modules — what Trainer / env see as Q_network
        self.Q_network = MeanEnsembleNet([c for c in self.critics])
        self.Q_target  = MeanEnsembleNet([c for c in self.critic_targets])

        # Discount factor: anneal to one
        self.GAMMA      = config.GAMMA 
        self.GammaScheduler = StepLRMargin(
            initValue=self.CONFIG.GAMMA,
            period=self.CONFIG.GAMMA_PERIOD,
            decay=self.CONFIG.GAMMA_DECAY,
            endValue=self.CONFIG.GAMMA_END,
            goalValue=1.0,
        )
        self.GAMMA = self.GammaScheduler.get_variable()

        # ----------------------------------------------------------------
        # Policies
        # ----------------------------------------------------------------
        PolicyCls = GaussianPolicy if self.policy_type == "Gaussian" else DeterministicPolicy
        if self.policy_type != "Gaussian":
            self.alpha = 0  # deterministic → no entropy bonus

        self.protagonist       = PolicyCls(config, dimList, action_space.shape[0], action_space, conditioned_sigma=True).to(self.device)
        self.protagonist_optim = optim.AdamW(self.protagonist.parameters(), lr=config.LR_A, weight_decay=1e-3)

        self.adversary       = PolicyCls(config, dimList, disturbance_space.shape[0], disturbance_space, conditioned_sigma=True).to(self.device)
        self.adversary_optim = optim.AdamW(self.adversary.parameters(), lr=config.LR_A, weight_decay=1e-3)

        # self.protagonist_scheduler = optim.lr_scheduler.StepLR(
        #     self.protagonist_optim, step_size=self.LR_A_PERIOD, gamma=self.LR_A_DECAY
        # )

        # self.adversary_scheduler = optim.lr_scheduler.StepLR(
        #     self.adversary_optim, step_size=self.LR_A_PERIOD, gamma=self.LR_A_DECAY
        # )

        self.prev_pro_loss = 0.0
        self.prev_adv_loss = 0.0

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

    @property
    def GAMMA(self):
        return self.critics[0].GAMMA

    @GAMMA.setter
    def GAMMA(self, value):
        for c in self.critics:
            c.GAMMA = value

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state, explore=False):
        """
        Returns (protagonist_action, adversary_action) as numpy arrays.

        Args:
            state (np.ndarray): current observation.
            explore (bool): if True sample stochastically, else use the
                            deterministic mean.
        """
        state_t = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            if explore:
                action,      _, _ = self.protagonist.sample(state_t)
                disturbance, _, _ = self.adversary.sample(state_t)
            else:
                _, _, action      = self.protagonist.sample(state_t)
                _, _, disturbance = self.adversary.sample(state_t)
        return (
            action.cpu().numpy()[0],
            disturbance.cpu().numpy()[0],
        )

    # ------------------------------------------------------------------
    # Epistemic Uncertainty
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_uncertainty(self, state_tensor, control, disturbance: torch.Tensor) -> dict:
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
        q_stack = self.Q_network.stack(state_tensor, control, disturbance)   # (K, B, A)

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

    # @torch.no_grad()
    # def get_uncertainty_map(self, env, nx: int = 41, ny: int = 41) -> dict:
    #     """
    #     Sweep the 2-D state grid and compute uncertainty at every cell.

    #     Returns dict of (nx, ny) numpy arrays:
    #         xs, ys, mean_v, var_v, std_v, min_v,
    #         epistemic_uncertainty, safe_disagreement
    #     """
    #     xs = np.linspace(
    #         env.unwrapped.bounds[0, 0], env.unwrapped.bounds[0, 1], nx
    #     )
    #     ys = np.linspace(
    #         env.unwrapped.bounds[1, 0], env.unwrapped.bounds[1, 1], ny
    #     )

    #     keys = ("mean_v", "var_v", "std_v", "min_v",
    #             "epistemic_uncertainty", "safe_disagreement")
    #     out  = {k: np.empty((nx, ny)) for k in keys}

    #     for ix, x in enumerate(xs):
    #         for iy, y in enumerate(ys):
    #             st = torch.FloatTensor([x, y]).unsqueeze(0).to(self.device)
    #             m  = self.get_uncertainty(st)
    #             out["mean_v"]              [ix, iy] = m["mean_q"].min().item()
    #             out["var_v"]               [ix, iy] = m["var_q"].min(dim=-1).values.item()
    #             out["std_v"]               [ix, iy] = m["std_q"].min(dim=-1).values.item()
    #             out["min_v"]               [ix, iy] = m["min_q"].min().item()
    #             out["epistemic_uncertainty"][ix, iy] = m["epistemic_uncertainty"].item()
    #             out["safe_disagreement"]   [ix, iy] = m["safe_disagreement"].item()

    #     out["xs"], out["ys"] = xs, ys
    #     return out

    # def plot_uncertainty_maps(
    #     self,
    #     env,
    #     out_folder: str,
    #     nx: int = 41,
    #     ny: int = 41,
    #     vmin: float = -4.0,
    #     vmax: float =  4.0,
    #     store: bool = True,
    #     show:  bool = False,
    # ) -> None:
    #     """
    #     4-panel figure:
    #       mean Q | conservative min Q | epistemic variance | safe disagreement
    #     """
    #     print("\n  [Ensemble] Computing uncertainty maps ...")
    #     maps    = self.get_uncertainty_map(env, nx=nx, ny=ny)
    #     axStyle = env.unwrapped.get_axes()
    #     xs, ys  = maps["xs"], maps["ys"]

    #     panels = [
    #         ("mean_v",               "seismic",  vmin,  vmax, r"Mean $\hat{V}$ (ensemble avg)"),
    #         ("min_v",                "seismic",  vmin,  vmax, r"Conservative $\hat{V}$ (min critic)"),
    #         ("epistemic_uncertainty","YlOrRd",   None,  None, r"Epistemic Uncertainty  Var$_k[Q]$"),
    #         ("safe_disagreement",    "PuRd",     0,     1,    "Safe / Unsafe Disagreement"),
    #     ]

    #     fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    #     for ax, (key, cmap, lo, hi, title) in zip(axes, panels):
    #         data  = maps[key]
    #         im_kw = dict(interpolation='none', extent=axStyle[0],
    #                      origin='lower', cmap=cmap)
    #         if lo is not None:
    #             im_kw.update(vmin=lo, vmax=hi)
    #         im = ax.imshow(data.T, **im_kw)
    #         fig.colorbar(im, ax=ax, pad=0.01, fraction=0.05, shrink=0.9)
    #         ax.set_title(title, fontsize=11)
    #         env.unwrapped.plot_target_failure_set(ax=ax)
    #         env.unwrapped.plot_reach_avoid_set(ax=ax)
    #         env.unwrapped.plot_formatting(ax=ax)
    #         if key in ("mean_v", "min_v"):
    #             ax.contour(xs, ys, data.T, levels=[0],
    #                        colors='k', linewidths=2, linestyles='dashed')

    #     fig.suptitle(
    #         f"Protagonist Ensemble  ({self.num_critics} critics | seed diversification"
    #         f" | mode={self.mode} | terminalType={self.terminalType})",
    #         fontsize=13,
    #     )
    #     fig.tight_layout()
    #     if store:
    #         os.makedirs(out_folder, exist_ok=True)
    #         path = os.path.join(out_folder, "ensemble_uncertainty.png")
    #         fig.savefig(path, dpi=150)
    #         print(f"  [Ensemble] Saved → {path}")
    #     if show:
    #         plt.show()
    #     plt.close(fig)

    def critic_update(self, critic, critic_optim, state, action, disturbance, non_final_state_nxt, non_final_mask, l_x, g_x, ep_unc=None):

        # ----------------------------------------------------------------
        # 1.  Critic update (reach-avoid Bellman target)
        # ----------------------------------------------------------------
        critic.train()

        qf1, qf2 = critic(state, action, disturbance)

        max_qf_next_target = torch.zeros(self.BATCH_SIZE).to(self.device)

        with torch.no_grad():
            next_action, next_log_pi_a, _ = self.protagonist.sample(non_final_state_nxt)
            next_disturb, next_log_pi_d, _ = self.adversary.sample(non_final_state_nxt)
            qf1_next, qf2_next = self.critic_target(non_final_state_nxt, next_action, next_disturb)
            # protagonist minimises → take the minimum of the two Q-heads
        max_qf_next_target[non_final_mask] = (torch.max(qf1_next, qf2_next) + (self.alpha_pro * next_log_pi_a + self.alpha_adv * next_log_pi_d)/2).view(-1)

        # Epistemic-uncertainty weight (FIX 3 – use self.CONFIG.TIME_STEP)
        _lambda  = 0.1 / ep_unc if ep_unc is not None else 1.0
        eu_weight = torch.exp(
            torch.tensor(_lambda * self.CONFIG.TIME_STEP, device=self.device)
        )

        # eu_weight = 1
        # Reach-avoid backup
        terminal     = torch.max(l_x, g_x)
        non_terminal = torch.max(g_x[non_final_mask],
            torch.min(l_x[non_final_mask], eu_weight * max_qf_next_target[non_final_mask]))

        next_q_value = torch.zeros(self.BATCH_SIZE).float().to(self.device)
        final_mask = torch.logical_not(non_final_mask)
        next_q_value[non_final_mask] = (
            (1 - self.GAMMA) * terminal[non_final_mask] + self.GAMMA * non_terminal
        )
        next_q_value[final_mask] = terminal[final_mask]

        qf1_loss = 0 * F.l1_loss(qf1, next_q_value.unsqueeze(-1).detach()) + F.mse_loss(qf1, next_q_value.unsqueeze(-1).detach())
        qf2_loss = 0 * F.l1_loss(qf2, next_q_value.unsqueeze(-1).detach()) + F.mse_loss(qf2, next_q_value.unsqueeze(-1).detach())
        qf_loss  = qf1_loss + qf2_loss

        critic_optim.zero_grad()
        qf_loss.backward()
        # nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        critic_optim.step()

        return qf1_loss, qf2_loss

    # ------------------------------------------------------------------
    # Gradient update
    # ------------------------------------------------------------------

    def update(self, memory, batch_size, updates, batch=None):
        """
        One gradient step for the critic, protagonist, and adversary.

        Args:
            memory (ReplayMemory): replay buffer.
            batch_size (int): mini-batch size.
            updates (int): global update counter (used for target sync).
            ep_unc (float | None): epistemic uncertainty weight.
            batch (Transition | None): pre-assembled batch (optional).

        Returns:
            Tuple (qf1_loss, qf2_loss, pro_loss, adv_loss, alpha_tlog)
            or None if the buffer is not yet large enough.
        """
        # if len(memory) < self.BATCH_SIZE * 20:
        #     return None

        # ---- sample from replay buffer ---------------------------------
        if batch is None:
            transitions = memory.sample(self.BATCH_SIZE)
            batch = Transition(*zip(*transitions))

        (
            non_final_mask,
            non_final_state_nxt,
            state,
            action,
            action_next,
            disturbance,
            _,
            g_x,
            l_x,
        ) = self.unpack_batch(batch)

        epistem_uncertainty = self.get_uncertainty(state, action, disturbance)["epistemic_uncertainty"].max().item()


        futures = {
            self._executor.submit(self.critic_update, c, c_opt, state, action, disturbance, non_final_state_nxt, non_final_mask, l_x, g_x, epistem_uncertainty): c
            for c, c_opt in zip(self.critics, self.critic_optimisers)
        }
        losses = []
        for fut in as_completed(futures):
            c = futures[fut]
            result = fut.result()
            if result is not None:
                losses.append(result)
            # if c is self.critics[0]:
            #     qf1_loss, qf2_loss = result
        q_means = [torch.stack(t).mean(0) for t in zip(*losses)]
        qf1_loss, qf2_loss = q_means

        if updates % 4 == 0:
            self.protagonist.train()
            pi_pro, log_pi_pro, _ = self.protagonist.sample(state)

            self.adversary.train()
            pi_adv, log_pi_adv, _ = self.adversary.sample(state)

            # ----------------------------------------------------------------
            # 2.  Protagonist update  (minimise Q)
            # ----------------------------------------------------------------
            # qf1_pi, qf2_pi = self.critic[0](state, pi_pro, pi_adv.detach())
            # max_qf_pi = torch.max(qf1_pi, qf2_pi)
            max_qf_pi = self.Q_network(state, pi_pro, pi_adv.detach())

            # Protagonist wants to *minimise* Q  → minimise  Q - α·H
            smoothness_loss = 0.002 * F.mse_loss(action_next, action)
            pro_loss = (max_qf_pi + self.alpha_pro * log_pi_pro).mean() + smoothness_loss

            self.protagonist_optim.zero_grad()
            pro_loss.backward(retain_graph=True)
            self.protagonist_optim.step()

            self.prev_pro_loss = pro_loss.item()
            # ----------------------------------------------------------------
            # 3.  Adversary update  (maximise Q)   
            # ----------------------------------------------------------------
            # qf1_pi, qf2_pi = self.critic[0](state, pi_pro.detach(), pi_adv)
            # max_qf_pi = torch.max(qf1_pi, qf2_pi)
            max_qf_pi = self.Q_network(state, pi_pro.detach(), pi_adv)

            # The adversary feeds its action into the *same* critic but wants
            # to drive Q *up* → maximise  Q + α·H  (entropy regularised) -loss
            adv_loss = (-max_qf_pi + self.alpha_adv * log_pi_adv).mean()

            self.adversary_optim.zero_grad()
            adv_loss.backward(retain_graph=True)
            self.adversary_optim.step()

            self.prev_adv_loss = adv_loss.item()

        # ----------------------------------------------------------------
        # 4.  Alpha / entropy tuning (disabled – kept for future use)
        # ----------------------------------------------------------------
        if self.autoAlphaTuning and updates % 8 == 0:
            pro_alpha_loss = -(self.pro_log_alpha * (log_pi_pro + self.pro_target_entropy).detach()).mean()
            self.pro_alpha_optim.zero_grad()
            pro_alpha_loss.backward()
            self.pro_alpha_optim.step()
            with torch.no_grad():
                self.pro_log_alpha.data.clamp_(-10, 2)
            self.alpha_pro = self.pro_log_alpha.exp()

            adv_alpha_loss = -(self.adv_log_alpha * (log_pi_adv + self.adv_target_entropy).detach()).mean()
            self.adv_alpha_optim.zero_grad()
            adv_alpha_loss.backward()
            self.adv_alpha_optim.step()
            with torch.no_grad():
                self.adv_log_alpha.data.clamp_(-10, 2)
            self.alpha_adv = self.adv_log_alpha.exp()

        alpha_tlogs = torch.tensor(float(self.alpha_pro))

        # ----------------------------------------------------------------
        # 5.  Soft-update of target critic
        # ----------------------------------------------------------------
        if updates % self.target_update_interval == 0:
            for critic_target, critic in zip(self.critic_targets, self.critics):
                soft_update(critic_target, critic, self.tau)

        return (
            qf1_loss.item(),
            qf2_loss.item(),
            self.prev_pro_loss,
            self.prev_adv_loss,
            alpha_tlogs.item(),
            epistem_uncertainty
        )

    # ------------------------------------------------------------------
    # Checkpointing 
    # ------------------------------------------------------------------

    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        os.makedirs("checkpoints/", exist_ok=True)
        if ckpt_path is None:
            ckpt_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
        print("Saving models to {}".format(ckpt_path))
        torch.save(
            {
                "protagonist_state_dict":       self.protagonist.state_dict(),
                "adversary_state_dict":         self.adversary.state_dict(),
                "critic_state_dict":            self.critic.state_dict(),
                "critic_target_state_dict":     self.critic_target.state_dict(),
                "critic_optimizer_state_dict":  self.critic_optim.state_dict(),
                "protagonist_optimizer_state_dict": self.protagonist_optim.state_dict(),
                "adversary_optimizer_state_dict":   self.adversary_optim.state_dict(),
            },
            ckpt_path,
        )

    def load_checkpoint(self, modelIter, ckpt_path, evaluate=True):
        pro_ckpt_path = os.path.join(ckpt_path, "pro_model", "model_{}.pt".format(modelIter))
        # critic_ckpt_path = os.path.join(ckpt_path, "pro_model", "critic_{}.pt".format(modelIter))
        adv_ckpt_path = os.path.join(ckpt_path, "adv_model", "model_{}.pt".format(modelIter))
        print("Loading models from {}".format(pro_ckpt_path))
        if ckpt_path is not None:
            self.protagonist.load_state_dict(torch.load(pro_ckpt_path, map_location=self.device))
            self.adversary.load_state_dict(torch.load(adv_ckpt_path, map_location=self.device))
            for i, critic in enumerate(self.critics):
                critic_ckpt_path_i = os.path.join(ckpt_path, "pro_model", "critic_{}".format(i), "critic_{}.pt".format(modelIter))
                critic.load_state_dict(torch.load(critic_ckpt_path_i, map_location=self.device))
        
            # self.critic_target.load_state_dict(ckpt["critic_target_state_dict"])
            # self.critic_optim.load_state_dict(ckpt["critic_optimizer_state_dict"])
            # self.protagonist_optim.load_state_dict(ckpt["protagonist_optimizer_state_dict"])
            # self.adversary_optim.load_state_dict(ckpt["adversary_optimizer_state_dict"])

            mode = "eval" if evaluate else "train"
            for net in [self.protagonist, self.adversary, self.critic]:
                getattr(net, mode)()
        
        self.Q_network = MeanEnsembleNet([c for c in self.critics])
        # self.Q_target  = MeanEnsembleNet([c for c in self.critic_targets])

    # ------------------------------------------------------------------
    # Update Hyperparameters
    # ------------------------------------------------------------------

    def updateHyperParam(self):
        """
        Updates the hypewr-parameters, such as learning rate, discount factor
        (GAMMA) and exploration-exploitation tradeoff (EPSILON)
        """
        lr = self.critic_optim.state_dict()["param_groups"][0]["lr"]
        if (lr <= self.LR_C_END):
            for param_group in self.critic_optim.param_groups:
                param_group["lr"] = self.LR_C_END
        else:
            self.scheduler.step()
            # self.protagonist_scheduler.step()
            # self.adversary_scheduler.step()

        self.GammaScheduler.step()
        self.GAMMA = self.GammaScheduler.get_variable()


    # ------------------------------------------------------------------
    # Batch unpacking  (unchanged from original)
    # ------------------------------------------------------------------

    def unpack_batch(self, batch):
        """Decomposes the batch into tensors ready for update().

        Returns:
            (non_final_mask, non_final_state_nxt, state, action,
             reward, g_x, l_x)
        """
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.s_)), dtype=torch.bool
        ).to(self.device)

        state = torch.FloatTensor(np.array(batch.s)).to(self.device)
        action = torch.FloatTensor(np.array(batch.a)).to(self.device)
        action_next = torch.FloatTensor(np.array(batch.a_)).to(self.device)
        disturbance = torch.FloatTensor(np.array(batch.d)).to(self.device)
        non_final_states = [s for s in batch.s_ if s is not None]
        non_final_state_nxt = (
            torch.from_numpy(np.vstack(non_final_states)).float().to(self.device)
            if non_final_states
            else torch.zeros(0, state.shape[1], device=self.device)
        )

        reward = torch.FloatTensor(np.array(batch.r)).to(self.device)
        g_x    = torch.FloatTensor([i["g_x"] for i in batch.info]).to(self.device).view(-1)
        l_x    = torch.FloatTensor([i["l_x"] for i in batch.info]).to(self.device).view(-1)

        return non_final_mask, non_final_state_nxt, state, action, action_next, disturbance, reward, g_x, l_x
