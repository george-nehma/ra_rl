"""
Please contact the author(s) of this library if you have any questions.
Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

This module implements reach-avoid reinforcement learning with double deep
Q-network. It also supports the standard sum of discounted rewards (Lagrange
cost) reinforcement learning.

Here we aim to minimize the reach-avoid cost, given by the Bellman backup:
    - a' = argmin_a' Q_network(s', a')
    - V(s') = Q_target(s', a')
    - V(s) = gamma ( max{ g(s), min{ l(s), V(s') } }
             + (1-gamma) max{ g(s), l(s) }
    - loss = E[ ( V(f(s,a)) - Q_network(s,a) )^2 ]
"""

import torch
import torch.nn as nn
from torch.nn.functional import mse_loss, smooth_l1_loss

from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import os
import time

from .model import Model
from .ReplayMemory import ReplayMemory
from .DDQN import DDQN, Transition

Transition = namedtuple("Transition", ["s", "a", "d", "r", "s_", "a_", "info"])


class Trainer():
  """
  Implements the double deep Q-network algorithm. Supports minimizing the
  reach-avoid cost or the standard sum of discounted costs.

  Args:
      DDQN (object): an object implementing the basic utils functions.
  """

  def __init__(
      self, CONFIG
  ):
    """
    Initializes with a configuration object, environment information, neural
    network architecture, reinforcement learning algorithm type and type of
    the terminal value for reach-avoid reinforcement learning.

    Args:
        CONFIG (object): configuration.
    """
    self.memory = ReplayMemory(CONFIG.MEMORY_CAPACITY)
    # super(Trainer, self).__init__(CONFIG)

  def store_transition(self, *args):
    """Stores the transition into the replay buffer.
    """
    self.memory.update(Transition(*args))

  def initBuffer(self, env, protagonist, adversary):
    """Adds some transitions to the replay memory (buffer) randomly.

    Args:
        env (gym.Env): the environment we interact with.
        protagonist (DDQNSingle): Protagonist
        adversary (DDQNSingle): Adversary agent
    """
    cnt = 0
    while len(self.memory) < self.memory.capacity:
      cnt += 1
      print("\rWarmup Buffer [{:d}]".format(cnt), end="")
      s, info = env.reset()
      u_idx = protagonist.select_action(s, explore=True)
      d_idx = adversary.select_action(np.concatenate([s, [u_idx]]), explore=True)
    #   u, d, u_idx, d_idx = self.select_action(s, explore=True)
      s_, r, done, info = env.step((u_idx, d_idx))
      s_ = None if done else s_
      u_idx_next = None if done else protagonist.select_action(s_, explore=True)
      self.store_transition(s, u_idx, d_idx, r, s_, u_idx_next, info)
      if done:
        s, info = env.reset()
      else:
        s = s_
    print(" --- Warmup Buffer Ends")

  def initQ(
      self, env, warmupIter, outFolder, num_warmup_samples=200, vmin=-1,
      vmax=1, plotFigure=True, storeFigure=True
  ):
    """
    Initalizes the Q-network given that the environment can provide warmup
    examples with heuristic values.

    Args:
        env (gym.Env): the environment we interact with.
        warmupIter (int, optional): the number of iterations in the
            Q-network warmup.
        outFolder (str, optional): the path of the parent folder of model/ and
            figure/.
        num_warmup_samples (int, optional): the number of warmup samples.
            Defaults to 200.
        vmin (float, optional): the minmum value in the colorbar.
            Defaults to -1.
        vmax (float, optional): the maximum value in the colorbar.
            Defaults to 1.
        plotFigure (bool, optional): plot figures if True.
            Defaults to True.
        storeFigure (bool, optional): store figures if True.
            Defaults to False.

    Returns:
        np.ndarray: loss of fitting Q-values to heuristic values.
    """
    lossList = np.empty(warmupIter, dtype=float)
    for ep_tmp in range(warmupIter):
      states, heuristic_v = env.get_warmup_examples(
          num_warmup_samples=num_warmup_samples
      )

      self.Q_network.train()
      heuristic_v = torch.from_numpy(heuristic_v).float().to(self.device)
      states = torch.from_numpy(states).float().to(self.device)
      v = self.Q_network(states)
      loss = mse_loss(input=v, target=heuristic_v, reduction="sum")

      self.optimizer.zero_grad()
      loss.backward()
      nn.utils.clip_grad_norm_(self.Q_network.parameters(), self.max_grad_norm)
      self.optimizer.step()
      lossList[ep_tmp] = loss.detach().cpu().numpy()
      print(
          "\rWarmup Q [{:d}]. MSE = {:f}".format(ep_tmp + 1, loss),
          end="",
      )

    print(" --- Warmup Q Ends")
    if plotFigure or storeFigure:
      env.visualize(self.Q_network, vmin=vmin, vmax=vmax, cmap="seismic")
      if storeFigure:
        figureFolder = os.path.join(outFolder, "figure")
        os.makedirs(figureFolder, exist_ok=True)
        figurePath = os.path.join(figureFolder, "initQ.png")
        plt.savefig(figurePath)
      if plotFigure:
        plt.pause(0.001)
      plt.clf()
      plt.close('all')
    self.target_network.load_state_dict(
        self.Q_network.state_dict()
    )  # hard replace
    self.build_optimizer()

    return lossList

  def learn(
      self, protagonist, adversary, env, MAX_UPDATES=2000000, MAX_EP_STEPS=100, warmupBuffer=True,
      warmupQ=False, warmupIter=10000, addBias=False, doneTerminate=True,
      runningCostThr=None, curUpdates=None, checkPeriod=50000, plotFigure=True,
      storeFigure=False, showBool=False, vmin=-1, vmax=1, numRndTraj=200,
      storeModel=True, storeBest=False, outFolder="RA", verbose=True
  ):
    """Learns the Q function given the training hyper-parameters.

    Args:
        protagonist (DDQNSingle): Protagonist
        adversary (DDQNSingle): Adversary agent
        env (gym.Env): the environment we interact with.
        MAX_UPDATES (int, optional): the maximum number of gradient updates.
            Defaults to 2000000.
        MAX_EP_STEPS (int, optional): the number of steps in an episode.
            Defaults to 100.
        warmupBuffer (bool, optional): fill the replay buffer if True.
            Defaults to True.
        warmupQ (bool, optional): train the Q-network by (l_x, g_x) if
            True. Defaults to False.
        warmupIter (int, optional): the number of iterations in the
            Q-network warmup. Defaults to 10000.
        addBias (bool, optional): use biased version of value function if
            True. Defaults to False.
        doneTerminate (bool, optional): end the episode when the agent
            crosses the boundary if True. Defaults to True.
        runningCostThr (float, optional): end the training if the running
            cost is smaller than this threshold. Defaults to None.
        curUpdates (int, optional): set the current number of updates
            (usually used when restoring trained models). Defaults to None.
        checkPeriod (int, optional): the period we check the performance.
            Defaults to 50000.
        plotFigure (bool, optional): plot figures if True. Defaults to True.
        storeFigure (bool, optional): store figures if True. Defaults to False.
        showBool (bool, optional): plot the sign of value function if True.
            Defaults to False.
        vmin (float, optional): the minimum value in the colorbar.
            Defaults to -1.
        vmax (float, optional): the maximum value in the colorbar.
            Defaults to 1.
        numRndTraj (int, optional): the number of random trajectories used
            to obtain the success ratio. Defaults to 200.
        storeModel (bool, optional): store models if True. Defaults to True.
        storeBest (bool, optional): only store the best model if True.
            Defaults to False.
        outFolder (str, optional): the path of the parent folder of model/ and
            figure/. Defaults to 'RA'.
        verbose (bool, optional): print the messages if True. Defaults to True.

    Returns:
        trainingRecords (np.ndarray): loss for every Q-network update.
        trainProgress (np.ndarray): each entry consists of the
            (success, failure, unfinished) ratio of random trajectories, which
            are checked periodically.
    """

    # == Warmup Buffer ==
    startInitBuffer = time.time()
    if warmupBuffer:
      self.initBuffer(env, protagonist, adversary)
    endInitBuffer = time.time()

    # == Warmup Q ==
    # startInitQ = time.time()
    # if warmupQ:
    #   self.initQ(
    #       env, warmupIter=warmupIter, outFolder=outFolder,
    #       plotFigure=plotFigure, storeFigure=storeFigure, vmin=vmin, vmax=vmax
    #   )
    # endInitQ = time.time()

    # == Main Training ==
    startLearning = time.time()
    trainingRecords = []
    runningCost = 0.0
    trainProgress = []
    checkPointSucc = 0.0
    ep = 0

    if curUpdates is not None:
      protagonist.cntUpdate = curUpdates
      print("starting from {:d} updates".format(protagonist.cntUpdate))

    if storeModel:
      pro_modelFolder = os.path.join(outFolder, "pro_model")
      os.makedirs(pro_modelFolder, exist_ok=True)
      adv_modelFolder = os.path.join(outFolder, "adv_model")
      os.makedirs(adv_modelFolder, exist_ok=True)
    if storeFigure:
      figureFolder = os.path.join(outFolder, "figure")
      os.makedirs(figureFolder, exist_ok=True)

    while protagonist.cntUpdate <= MAX_UPDATES:
      s, info = env.reset()
      u_idx = protagonist.select_action(s, explore=True)
      epCost = 0.0
      ep += 1
      # Rollout
      for step_num in range(MAX_EP_STEPS):
        # Select action
        # u, d, u_idx, d_idx = self.select_action(s,explore=True)
        u_idx = protagonist.select_action(s, explore=True)
        d_idx = adversary.select_action(np.concatenate([s, [u_idx]]), explore=True)

        # Interact with env
        s_, r, done, info = env.step((u_idx, d_idx))
        s_ = None if done else s_
        epCost += r

        # Store the transition in shared memory
        u_idx_next = None if done else protagonist.select_action(s_, explore=True)
        self.store_transition(s, u_idx, d_idx, r, s_, u_idx_next, info)
        s = s_
        
        # Check after fixed number of gradient updates
        if protagonist.cntUpdate != 0 and protagonist.cntUpdate % checkPeriod == 0:
          results = env.simulate_trajectories(
              protagonist.Q_network, adversary.Q_network, T=MAX_EP_STEPS, num_rnd_traj=numRndTraj,
          )[1]
          success = np.sum(results == 1) / results.shape[0]
          failure = np.sum(results == -1) / results.shape[0]
          unfinish = np.sum(results == 0) / results.shape[0]
          trainProgress.append([success, failure, unfinish])
          if verbose:
            lr = protagonist.optimizer.state_dict()["param_groups"][0]["lr"]
            print("\nAfter [{:d}] updates:".format(protagonist.cntUpdate))
            print(
                "  - eps={:.2f}, gamma={:.6f}, protagonist lr={:.1e}.".format(
                    protagonist.EPSILON, protagonist.GAMMA, lr
                )
            )
            print("  - success/failure/unfinished ratio:", end=" ")
            with np.printoptions(formatter={"float": "{: .3f}".format}):
              print(np.array([success, failure, unfinish]))

          if storeModel:
            if storeBest:
              if success > checkPointSucc:
                checkPointSucc = success
                protagonist.save(protagonist.cntUpdate, pro_modelFolder)
                adversary.save(protagonist.cntUpdate, adv_modelFolder)
            else:
              protagonist.save(protagonist.cntUpdate, pro_modelFolder)
              adversary.save(protagonist.cntUpdate, adv_modelFolder)

          if plotFigure or storeFigure:
            # self.Q_network.eval()
            if showBool:
              env.visualize(
                  protagonist.Q_network, vmin=0, boolPlot=True, addBias=addBias
              )
            #   env.visualize(
            #       adversary.Q_network, vmin=0, boolPlot=True, addBias=addBias
            #   )
            else:
              env.visualize(
                  protagonist.Q_network, vmin=vmin, vmax=vmax, cmap="seismic",
                  addBias=addBias
              )
            #   env.visualize(
            #       adversary.Q_network, vmin=vmin, vmax=vmax, cmap="seismic",
            #       addBias=addBias
            #   )
            if storeFigure:
              figurePath = os.path.join(
                  figureFolder, "{:d}.png".format(protagonist.cntUpdate)
              )
              plt.savefig(figurePath)
            if plotFigure:
              plt.pause(0.001)
            plt.clf()
            plt.close('all')

        # Perform one step of the optimization (on the target network)
        lossC_pro = protagonist.update(addBias=addBias)
        lossC_adv = adversary.update(addBias=addBias)
        trainingRecords.append(lossC_pro)
        trainingRecords.append(lossC_adv)
        protagonist.cntUpdate += 1
        protagonist.updateHyperParam()
        adversary.updateHyperParam()

        # Terminate early
        if done and doneTerminate:
          break

      # Rollout report
      runningCost = runningCost*0.9 + epCost*0.1
      if verbose:
        print(
            "\r[{:d}-{:d}]: ".format(ep, protagonist.cntUpdate)
            + "This episode gets running/episode cost = "
            + "({:3.2f}/{:.2f}) after {:d} steps.".
            format(runningCost, epCost, step_num + 1),
            end="",
        )

      # Check stopping criteria
      if runningCostThr is not None:
        if runningCost <= runningCostThr:
          print(
              "\n At Updates[{:3.0f}] Solved!".format(protagonist.cntUpdate)
              + " Running cost is now {:3.2f}!".format(runningCost)
          )
          env.close()
          break
    endLearning = time.time()
    timeInitBuffer = endInitBuffer - startInitBuffer
    timeInitQ = 0 # endInitQ - startInitQ
    timeLearning = endLearning - startLearning
    protagonist.save(protagonist.cntUpdate, pro_modelFolder)
    adversary.save(protagonist.cntUpdate, adv_modelFolder)
    print(
        "\nInitBuffer: {:.1f}, InitQ: {:.1f}, Learning: {:.1f}".format(
            timeInitBuffer, timeInitQ, timeLearning
        )
    )
    trainingRecords = np.array(trainingRecords)
    trainProgress = np.array(trainProgress)
    return trainingRecords, trainProgress
