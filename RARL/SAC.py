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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.optim import Adam
import torch.optim as optim
from .utils import soft_update, hard_update, save_model
from .model import GaussianPolicy, QNetwork, DeterministicPolicy, StepLRMargin, StepResetLR

from collections import namedtuple
Transition = namedtuple("Transition", ["s", "a", "d", "r", "s_", "a_", "info"])

class SAC(object):
    def __init__(self, config, dimList, action_space, disturbance_space):

        # ----------------------------------------------------------------
        # Hyper-parameters  (kept as both lower- and UPPER-case so that
        # existing code referencing either form continues to work)
        # ----------------------------------------------------------------
        self.CONFIG     = config
        self.gamma      = config.GAMMA
        self.GAMMA      = config.GAMMA          
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

        self.dimList    = dimList

        self.policy_type           = config.POLICY
        self.target_update_interval = config.TARGET_UPDATE_INTERVAL

        self.device = torch.device(config.DEVICE)

        # Discount factor: anneal to one
        self.GammaScheduler = StepLRMargin(
            initValue=self.CONFIG.GAMMA,
            period=self.CONFIG.GAMMA_PERIOD,
            decay=self.CONFIG.GAMMA_DECAY,
            endValue=self.CONFIG.GAMMA_END,
            goalValue=1.0,
        )
        self.GAMMA = self.GammaScheduler.get_variable()

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
        q_dimList = dimList #[dimList[0], 64, 64, 1]
        self.critic = QNetwork(config, q_dimList, action_space.shape[0], disturbance_space.shape[0]).to(self.device)
        self.critic_optim = optim.AdamW(self.critic.parameters(), lr=config.LR_C, weight_decay=1e-3)

        self.scheduler = optim.lr_scheduler.StepLR(
            self.critic_optim, step_size=self.LR_C_PERIOD, gamma=self.LR_C_DECAY
        )
        self.max_grad_norm = 1
        self.cntUpdate = 0

        self.critic_target = QNetwork(config, q_dimList, action_space.shape[0], disturbance_space.shape[0]).to(self.device)
        hard_update(self.critic_target, self.critic)

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
    # Gradient update
    # ------------------------------------------------------------------

    def update(self, memory, batch_size, updates, ep_unc=None, batch=None):
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
        if len(memory) < self.BATCH_SIZE * 20:
            return None

        # ---- sample from replay buffer ---------------------------------
        if batch is None:
            transitions = memory.sample(self.BATCH_SIZE)
            batch = Transition(*zip(*transitions))

        (
            non_final_mask,
            non_final_state_nxt,
            state,
            action,
            disturbance,
            _,
            g_x,
            l_x,
        ) = self.unpack_batch(batch)

        # ----------------------------------------------------------------
        # 1.  Critic update (reach-avoid Bellman target)
        # ----------------------------------------------------------------
        self.critic.train()

        qf1, qf2 = self.critic(state, action, disturbance)

        min_qf_next_target = torch.zeros(self.BATCH_SIZE).to(self.device)

        with torch.no_grad():
            next_action, next_log_pi_a, _ = self.protagonist.sample(non_final_state_nxt)
            next_disturb, next_log_pi_d, _ = self.adversary.sample(non_final_state_nxt)
            qf1_next, qf2_next = self.critic_target(non_final_state_nxt, next_action, next_disturb)
            # protagonist minimises → take the minimum of the two Q-heads
        min_qf_next_target[non_final_mask] = (torch.min(qf1_next, qf2_next) + (self.alpha_pro * next_log_pi_a + self.alpha_adv * next_log_pi_d)/2).view(-1)

        # Epistemic-uncertainty weight (FIX 3 – use self.CONFIG.TIME_STEP)
        _lambda  = 1.0 / ep_unc if ep_unc is not None else 1.0
        eu_weight = torch.exp(
            torch.tensor(_lambda * self.CONFIG.TIME_STEP, device=self.device)
        )

        eu_weight = 1
        # Reach-avoid backup
        terminal     = torch.max(l_x, g_x)
        non_terminal = torch.max(g_x[non_final_mask],
            torch.min(l_x[non_final_mask], eu_weight * min_qf_next_target[non_final_mask]))

        next_q_value = torch.zeros(self.BATCH_SIZE).float().to(self.device)
        final_mask = torch.logical_not(non_final_mask)
        next_q_value[non_final_mask] = (
            (1 - self.GAMMA) * terminal[non_final_mask] + self.GAMMA * non_terminal
        )
        next_q_value[final_mask] = terminal[final_mask]

        qf1_loss = F.smooth_l1_loss(qf1, next_q_value.unsqueeze(-1).detach())
        qf2_loss = F.smooth_l1_loss(qf2, next_q_value.unsqueeze(-1).detach())
        qf_loss  = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optim.step()

        self.protagonist.train()
        pi_pro, log_pi_pro, _ = self.protagonist.sample(state)

        self.adversary.train()
        pi_adv, log_pi_adv, _ = self.adversary.sample(state)

        # ----------------------------------------------------------------
        # 2.  Protagonist update  (minimise Q)
        # ----------------------------------------------------------------
        qf1_pi, qf2_pi = self.critic(state, pi_pro, pi_adv.detach())
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        # Protagonist wants to *minimise* Q  → minimise  Q - α·H
        pro_loss = (min_qf_pi + self.alpha_pro * log_pi_pro).mean()

        self.protagonist_optim.zero_grad()
        pro_loss.backward()
        self.protagonist_optim.step()
        # ----------------------------------------------------------------
        # 3.  Adversary update  (maximise Q)   
        # ----------------------------------------------------------------
        qf1_pi, qf2_pi = self.critic(state, pi_pro.detach(), pi_adv)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        # The adversary feeds its action into the *same* critic but wants
        # to drive Q *up* → maximise  Q + α·H  (entropy regularised) -loss
        adv_loss = (-min_qf_pi + self.alpha_adv * log_pi_adv).mean()

        self.adversary_optim.zero_grad()
        adv_loss.backward()
        self.adversary_optim.step()

        # ----------------------------------------------------------------
        # 4.  Alpha / entropy tuning (disabled – kept for future use)
        # ----------------------------------------------------------------
        if self.autoAlphaTuning:
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
        # else:
        #     pro_alpha_loss = torch.tensor(0.0, device=self.device)

        alpha_tlogs = torch.tensor(float(self.alpha_pro))

        # ----------------------------------------------------------------
        # 5.  Soft-update of target critic
        # ----------------------------------------------------------------
        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return (
            qf1_loss.item(),
            qf2_loss.item(),
            pro_loss.item(),
            adv_loss.item(),
            alpha_tlogs.item(),
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
        critic_ckpt_path = os.path.join(ckpt_path, "pro_model", "critic_{}.pt".format(modelIter))
        adv_ckpt_path = os.path.join(ckpt_path, "adv_model", "model_{}.pt".format(modelIter))
        print("Loading models from {}".format(ckpt_path))
        if ckpt_path is not None:
            # ckpt = torch.load(ckpt_path, map_location=self.device)
            self.protagonist.load_state_dict(torch.load(pro_ckpt_path, map_location=self.device))
            self.adversary.load_state_dict(torch.load(adv_ckpt_path, map_location=self.device))
            self.critic.load_state_dict(torch.load(critic_ckpt_path, map_location=self.device))
            # self.critic_target.load_state_dict(ckpt["critic_target_state_dict"])
            # self.critic_optim.load_state_dict(ckpt["critic_optimizer_state_dict"])
            # self.protagonist_optim.load_state_dict(ckpt["protagonist_optimizer_state_dict"])
            # self.adversary_optim.load_state_dict(ckpt["adversary_optimizer_state_dict"])

            mode = "eval" if evaluate else "train"
            for net in [self.protagonist, self.adversary, self.critic]:
                getattr(net, mode)()

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

        return non_final_mask, non_final_state_nxt, state, action, disturbance, reward, g_x, l_x
