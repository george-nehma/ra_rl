"""
SACTrainer.py

A trainer for reach-avoid reinforcement learning using Soft Actor-Critic (SAC).
Mirrors the structure of Trainer.py (DDQN-based) but works with the continuous-
action SAC agent defined in SAC.py.

Reach-avoid Bellman backup (same objective as Trainer.py):
    V(s) = (1 - γ) * max{ g(s), l(s) }
           + γ * max{ g(s), min{ l(s), V(s') } }
    loss  = E[ ( V(f(s,a)) - Q(s,a) )² ]

The protagonist tries to minimise this value; the adversary tries to maximise it.
"""

from csv import writer
import os
import time

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from .SAC import SAC, Transition  # noqa: F401  (imported so callers can do: from SACTrainer import SACTrainer)
from .ReplayMemory import ReplayMemory

# ---------------------------------------------------------------------------
# Transition tuple – identical layout to the one used in Trainer.py so that
# a shared ReplayMemory can be reused unchanged.
# Fields
#   s      : current state  (np.ndarray)
#   a      : protagonist action  (np.ndarray, continuous)
#   d      : adversary  action  (np.ndarray, continuous)
#   r      : scalar reward / cost
#   s_     : next state (np.ndarray or None if terminal)
#   a_     : protagonist action at s_ (np.ndarray or None if terminal)
#   info   : dict with at least {"g_x": float, "l_x": float}
# ---------------------------------------------------------------------------



class SACTrainer:
    """
    Trainer for a two-player reach-avoid game using SAC.

    The SAC object owns both a *protagonist* policy (minimiser) and an
    *adversary* policy (maximiser), as well as a shared twin-Q critic.

    Args:
        sac_agent (SAC): fully constructed SAC instance.
        CONFIG (object): configuration object – must expose at least
            MEMORY_CAPACITY, BATCH_SIZE, and optionally DEVICE.
    """

    def __init__(self, sac_agent: SAC, CONFIG):
        self.agent  = sac_agent
        self.CONFIG = CONFIG
        self.memory = ReplayMemory(CONFIG.MEMORY_CAPACITY) # shared ReplayMemory

    # ------------------------------------------------------------------
    # Replay buffer helpers
    # ------------------------------------------------------------------

    def store_transition(self, *args):
        """Push one transition into the replay buffer."""
        self.memory.update(Transition(*args))

    def initBuffer(self, env):
        """
        Fill the replay buffer to capacity using uniformly random actions.

        Args:
            env (gym.Env): environment following the (s, info) = env.reset()
                / (s_, r, done, _, info) = env.step((u, d)) interface.
        """
        cnt = 0
        s, info = env.reset()
        while len(self.memory) < self.memory.capacity:
            cnt += 1
            print("\rWarmup Buffer [{:d}]".format(cnt), end="")

            # Random continuous actions sampled from the action space
            u = env.action_space.sample()   # protagonist
            d = env.action_space.sample()   # adversary

            s_, r, done, _, info = env.step((u, d))

            # s_store  = None if done else s_

            # Pre-compute the next protagonist action (needed for certain
            # RA-backup variants that use a_ just like the DDQN trainer).
            # if done:
                # a_next = None
            # else:
            a_next, _ = self.agent.select_action(s_, explore=True)

            self.store_transition(s, u, d, r, s_, a_next, done, info)

            if done:
                s, info = env.reset()
            else:
                s = s_

        print(" --- Warmup Buffer Ends")

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def learn(
        self,
        env,
        MAX_UPDATES    = 2_000_000,
        MAX_EP_STEPS   = 100,
        warmupQ        = False,
        doneTerminate  = True,
        runningCostThr = None,
        curUpdates     = None,
        checkPeriod    = 50_000,
        plotFigure     = True,
        storeFigure    = False,
        plotTrainValue = True,
        showBool       = False,
        vmin           = -1,
        vmax           = 1,
        numRndTraj     = 200,
        itr_init       = 0,
        storeModel     = True,
        storeBest      = False,
        outFolder      = "SAC_RA",
        verbose        = True,
    ):
        """
        Learn the Q-function (and policies) using SAC for reach-avoidance.

        Args:
            env (gym.Env): the environment.
            MAX_UPDATES (int): maximum number of critic gradient updates.
            MAX_EP_STEPS (int): maximum steps per episode.
            warmupBuffer (bool): pre-fill the replay buffer before training.
            doneTerminate (bool): end episode on 'done' flag.
            runningCostThr (float | None): stop early when running cost
                drops below this threshold.
            curUpdates (int | None): resume from this update count.
            checkPeriod (int): evaluate & (optionally) save every N updates.
            plotFigure (bool): render value-function plots interactively.
            storeFigure (bool): save value-function plots to disk.
            plotTrainValue (bool): include value-function in check-period plots.
            showBool (bool): plot sign(V) instead of V.
            vmin (float): colour-bar minimum.
            vmax (float): colour-bar maximum.
            numRndTraj (int): trajectories used to compute success ratio.
            storeModel (bool): save checkpoints to disk.
            storeBest (bool): only overwrite checkpoint when success improves.
            outFolder (str): root folder for model/ and figure/ sub-dirs.
            verbose (bool): print progress messages.

        Returns:
            trainingRecords (np.ndarray): shape (N, 5) –
                [qf1_loss, qf2_loss, pro_loss, adv_loss, alpha] per update.
            trainProgress (np.ndarray): shape (M, 3) –
                [success, failure, unfinished] ratio at each check point.
        """

        # ----------------------------------------------------------------
        # Warmup buffer
        # ----------------------------------------------------------------
        self.initBuffer(env)

        # ----------------------------------------------------------------
        # Bookkeeping / folder creation
        # ----------------------------------------------------------------
        cntUpdate = curUpdates if curUpdates is not None else 0
        if curUpdates is not None:
            print("Resuming from {:d} updates.".format(cntUpdate))

        if storeModel:
            pro_modelFolder = os.path.join(outFolder, "pro_model")
            adv_modelFolder = os.path.join(outFolder, "adv_model")
            os.makedirs(pro_modelFolder, exist_ok=True)
            os.makedirs(adv_modelFolder, exist_ok=True)
        if storeFigure:
            figureFolder = os.path.join(outFolder, "figure")
            os.makedirs(figureFolder, exist_ok=True)

        trainingRecords  = []
        trainProgress    = []
        runningCost      = 0.0
        checkPointSucc   = 0.0
        ep               = 0
        epistemic_uncertainty = 1.0
        # ----------------------------------------------------------------
        # Tensorboard for logging
        # ----------------------------------------------------------------
        logger  = SummaryWriter(log_dir=os.path.join(outFolder, "logs"))

        # ----------------------------------------------------------------
        # Main loop
        # ----------------------------------------------------------------
        t0_learn = time.time()

        while cntUpdate <= MAX_UPDATES:
            s, info = env.reset()
            epCost  = 0.0
            ep     += 1

            for step_num in range(MAX_EP_STEPS):
                # ---- action selection ----------------------------------
                # explore=True → stochastic (reparameterised) sample
                # explore=False → deterministic mean action
                u, d = self.agent.select_action(s, explore=True)

                # ---- environment step ----------------------------------
                s_, r, done, _, info = env.step((u, d))
                epCost += r

                # terminal_step = done or (step_num == MAX_EP_STEPS - 1)
                # s_store  = None if terminal_step else s_
                # a_next   = None
                # if not terminal_step:
                a_next, _ = self.agent.select_action(s_, explore=True)

                self.store_transition(s, u, d, r, s_, a_next, done, info)
                s = s_

                # ---- periodic evaluation --------------------------------
                if cntUpdate != 0 and cntUpdate % checkPeriod == 0:
                    results = env.unwrapped.simulate_trajectories(
                        self.agent,
                        T=MAX_EP_STEPS,
                        num_rnd_traj=numRndTraj,
                    )[1]

                    success  = np.sum(results ==  1) / results.shape[0]
                    failure  = np.sum(results == -1) / results.shape[0]
                    unfinish = np.sum(results ==  0) / results.shape[0]
                    trainProgress.append([success, failure, unfinish])

                    if verbose:
                        pro_lr = self.agent.critic_optim.state_dict(
                        )["param_groups"][0]["lr"]
                        print("\nAfter [{:d}] updates:".format(cntUpdate))
                        print(
                            "  - epi_uncertainty={:.4f}, pro_alpha={:.7f},"
                            " pro_lr={:.7e}, gamma={:.7e}.".format(
                                epistemic_uncertainty,
                                float(self.agent.alpha_pro),
                                pro_lr,
                                self.agent.GAMMA,
                            )
                        )
                        print("  - success/failure/unfinished:", end=" ")
                        with np.printoptions(
                            formatter={"float": "{: .3f}".format}
                        ):
                            print(np.array([success, failure, unfinish]))

                    if storeModel:
                        if storeBest:
                            if success > checkPointSucc:
                                checkPointSucc = success
                                self._save_models(
                                    cntUpdate+itr_init,
                                    pro_modelFolder,
                                    adv_modelFolder,
                                )
                        else:
                            self._save_models(
                                cntUpdate+itr_init, pro_modelFolder, adv_modelFolder
                            )

                    if (plotFigure or storeFigure) and plotTrainValue:
                        self.agent.Q_network.eval()
                        env.unwrapped.visualize(
                            self.agent,
                            vmin=0 if showBool else vmin,
                            vmax=vmax,
                            boolPlot=showBool,
                            cmap="seismic",
                        )
                        if storeFigure:
                            figurePath = os.path.join(
                                figureFolder, "{:d}.png".format(cntUpdate+itr_init)
                            )
                            plt.savefig(figurePath)
                        if plotFigure:
                            plt.pause(0.001)
                        plt.clf()
                        plt.close("all")

                # ---- gradient update -----------------------------------
                losses = self.agent.update(
                    self.memory,
                    batch_size=self.CONFIG.BATCH_SIZE,
                    updates=cntUpdate
                )
                self.agent.updateHyperParam()
                if losses is not None:
                    (
                        qf1_loss,
                        qf2_loss,
                        pro_loss,
                        adv_loss,
                        alpha_tlogs,
                        epistemic_uncertainty
                    ) = losses
                    # Use critic loss spread as a proxy for epistemic
                    # uncertainty (larger disagreement → higher uncertainty).
                    # epistem_uncertainty = abs(qf1_loss - qf2_loss) + 1e-8
                    trainingRecords.append(
                        [
                            qf1_loss,
                            qf2_loss,
                            pro_loss,
                            adv_loss,
                            float(alpha_tlogs),
                            float(epistemic_uncertainty)
                        ]
                    )
                
                # Log losses
                logger.add_scalar("Loss/protagonist", pro_loss, cntUpdate+itr_init)
                logger.add_scalar("Loss/adversary", adv_loss, cntUpdate+itr_init)
                logger.add_scalar("Loss/critic1", qf1_loss, cntUpdate+itr_init)
                logger.add_scalar("Loss/critic2", qf2_loss, cntUpdate+itr_init)
                logger.add_scalar("HyperParam/alpha_pro", float(alpha_tlogs), cntUpdate+itr_init)
                logger.add_scalar("HyperParam/Epistemic_uncertainty", float(epistemic_uncertainty), cntUpdate+itr_init)

                cntUpdate += 1

                if done and doneTerminate:
                    break

            # ---- episode report ----------------------------------------
            runningCost = runningCost * 0.9 + epCost * 0.1
            if verbose:
                print(
                    "\r[ep {:d} | upd {:d}]: running/ep cost ="
                    " ({:.2f}/{:.2f}), {:d} steps.".format(
                        ep, cntUpdate, runningCost, epCost, step_num + 1
                    ),
                    end="",
                )

            # ---- early stopping ----------------------------------------
            if runningCostThr is not None and runningCost <= runningCostThr:
                print(
                    "\nSolved at update {:d}!"
                    " Running cost = {:.2f}.".format(cntUpdate+itr_init, runningCost)
                )
                env.close()
                logger.close()
                break

        t1_learn = time.time()

        # ---- final save ------------------------------------------------
        if storeModel:
            self._save_models(cntUpdate+itr_init, pro_modelFolder, adv_modelFolder)

        print(
            "\nLearning: {:.1f}s".format(t1_learn - t0_learn))

        trainingRecords = np.array(trainingRecords) if trainingRecords else np.empty((0, 5))
        trainProgress   = np.array(trainProgress)   if trainProgress   else np.empty((0, 3))
        return trainingRecords, trainProgress

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _save_models(self, cntUpdate, pro_folder, adv_folder):
        """Save protagonist and adversary policy checkpoints."""
        pro_path = os.path.join(pro_folder, "model_{:d}.pt".format(cntUpdate))
        adv_path = os.path.join(adv_folder, "model_{:d}.pt".format(cntUpdate))
        torch.save(self.agent.protagonist.state_dict(), pro_path)
        torch.save(self.agent.adversary.state_dict(),   adv_path)
        if hasattr(self.agent, "critic"):
            for i, c in enumerate(self.agent.critics):
                sub = os.path.join(pro_folder, f"critic_{i}")
                os.makedirs(sub, exist_ok=True)
                critic_path = os.path.join(
                    sub, "critic_{:d}.pt".format(cntUpdate)
                )
                torch.save(c.state_dict(), critic_path)
