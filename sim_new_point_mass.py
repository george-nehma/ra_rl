"""
Please contact the author(s) of this library if you have any questions.
Authors: George Nehma ( gnehma2020@fit.edu )

This experiment runs double deep Q-network with the discounted reach-avoid
Bellman equation (DRABE) proposed in [RSS21] on a 2-dimensional point mass
problem. We use this script to generate Fig. 2 and Fig. 3 in the paper.

Examples:
    RA:
        python3 sim_naive.py -w -sf -of scratch -a -g 0.99 -n anneal
        python3 sim_naive.py -w -sf -of scratch -n 9999
        python3 sim_naive.py -w -sf -of scratch -g 0.999 -dt fail -n 999
    Lagrange:
        python3 sim_naive.py -sf -m lagrange -of scratch -g 0.95 -n 95
        python3 sim_naive.py -sf -m lagrange -of scratch -dt TF -g 0.95 -n 95
    test: python3 sim_naive.py -w -sf -of scratch -wi 100 -mu 100 -cp 40
"""

import os
import argparse
import time
from warnings import simplefilter
import gymnasium as gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

import yaml
from types import SimpleNamespace

from RARL.DDQNSingle import DDQNSingle
# from RARL.DDQNCriticEnsemble import CriticEnsemble
from RARL.Trainer import Trainer
from RARL.config import dqnConfig, ceConfig
from RARL.utils import save_obj
from utils.utils import plot_protagonist_adversary_actions, plot_RA_eval, plot_protagonist_adversary_values, PlotConfig
from gym_reachability import gym_reachability  # Custom Gym env.

matplotlib.use('Agg')
simplefilter(action='ignore', category=FutureWarning)
timestr = time.strftime("%Y-%m-%d-%H_%M_%S")

# == ARGS ==

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config.yaml", type=str)
script_args = parser.parse_args()

with open(script_args.config, "r") as f:
    config = yaml.safe_load(f)

args = SimpleNamespace(**{k: v for section in config.values() for k, v in section.items()})
print(args)

# == CONFIGURATION ==
env_name = "zermelo_show-v0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
maxUpdates = args.maxUpdates
updateTimes = args.updateTimes
updatePeriod = int(maxUpdates / updateTimes)
maxSteps = 250
storeFigure = args.storeFigure
plotFigure = args.plotFigure

fn = args.name + 'TF'
if args.showTime:
  fn = fn + '-' + timestr

outFolder = os.path.join(args.outFolder, 'point-mass', 'SAC', args.mode, fn)
print(outFolder)
figureFolder = os.path.join(outFolder, 'figure')
os.makedirs(figureFolder, exist_ok=True)

if args.mode == 'RA':
  agentMode = 'RA'
  if args.annealing:
    GAMMA_END = 0.999999
    EPS_PERIOD = int(updatePeriod / 10)
    EPS_RESET_PERIOD = updatePeriod
  else:
    GAMMA_END = args.gamma
    EPS_PERIOD = updatePeriod
    EPS_RESET_PERIOD = maxUpdates

elif args.mode == 'AARA':
  agentMode = 'AARA'
  if args.annealing:
    GAMMA_END = 0.999999
    EPS_PERIOD = int(updatePeriod / 10)
    EPS_RESET_PERIOD = updatePeriod
  else:
    GAMMA_END = args.gamma
    EPS_PERIOD = updatePeriod
    EPS_RESET_PERIOD = maxUpdates

sample_inside_obs = False

# == Environment ==
print("\n== Environment Information ==")
env = gym.make(
    env_name, config=args, device=device,
    sample_inside_obs=sample_inside_obs, envType="basic"
)

stateDim = env.unwrapped.state.shape[0]
actionNum = env.unwrapped.action_space.n
action_list = np.arange(actionNum)
print(
    "State Dimension: {:d}, ActionSpace Dimension: {:d}".format(
        stateDim, actionNum
    )
)
print(f"Discrete Controls: {env.unwrapped.discrete_controls}")

env.unwrapped.set_costParam(args.penalty, args.reward, args.costType, args.scaling) # only needed for Lagrange
env.unwrapped.set_seed(args.randomSeed)


if plotFigure or storeFigure:
  nx, ny = 101, 101
  vmin = -1 * args.scaling
  vmax = 1 * args.scaling

  v = np.zeros((nx, ny))
  l_x = np.zeros((nx, ny))
  g_x = np.zeros((nx, ny))
  xs = np.linspace(env.unwrapped.bounds[0, 0], env.unwrapped.bounds[0, 1], nx)
  ys = np.linspace(env.unwrapped.bounds[1, 0], env.unwrapped.bounds[1, 1], ny)

  it = np.nditer(v, flags=['multi_index'])

  while not it.finished:
    idx = it.multi_index
    x = xs[idx[0]]
    y = ys[idx[1]]

    l_x[idx] = env.unwrapped.target_margin(np.array([x, y]))
    g_x[idx] = env.unwrapped.safety_margin(np.array([x, y]))

    v[idx] = np.maximum(l_x[idx], g_x[idx])
    it.iternext()

  axStyle = env.unwrapped.get_axes()

  fig, axes = plt.subplots(1, 3, figsize=(12, 6))

  ax = axes[0]
  im = ax.imshow(
      l_x.T, interpolation='none', extent=axStyle[0], origin="lower",
      cmap="seismic", vmin=vmin, vmax=vmax
  )
  cbar = fig.colorbar(
      im, ax=ax, pad=0.01, fraction=0.05, shrink=.95, ticks=[vmin, 0, vmax]
  )
  cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=24)
  ax.set_title(r'$\ell(x)$', fontsize=18)

  ax = axes[1]
  im = ax.imshow(
      g_x.T, interpolation='none', extent=axStyle[0], origin="lower",
      cmap="seismic", vmin=vmin, vmax=vmax
  )
  cbar = fig.colorbar(
      im, ax=ax, pad=0.01, fraction=0.05, shrink=.95, ticks=[vmin, 0, vmax]
  )
  cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=24)
  ax.set_title(r'$g(x)$', fontsize=18)

  ax = axes[2]
  im = ax.imshow(
      v.T, interpolation='none', extent=axStyle[0], origin="lower",
      cmap="seismic", vmin=vmin, vmax=vmax
  )
  env.unwrapped.plot_reach_avoid_set(ax)
  cbar = fig.colorbar(
      im, ax=ax, pad=0.01, fraction=0.05, shrink=.95, ticks=[vmin, 0, vmax]
  )
  cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=24)
  ax.set_title(r'$v(x)$', fontsize=18)

  for ax in axes:
    env.unwrapped.plot_target_failure_set(ax=ax)
    env.unwrapped.plot_formatting(ax=ax)

  fig.tight_layout()
  if storeFigure:
    figurePath = os.path.join(figureFolder, 'env.png')
    fig.savefig(figurePath)
  if plotFigure:
    plt.show()
    plt.pause(0.001)
  plt.close()

# == Agent CONFIG ==
print("\n== Agent Information ==")
PRO_CONFIG = dqnConfig(
    DEVICE=device, ENV_NAME=env_name, SEED=args.randomSeed,
    MAX_UPDATES=maxUpdates, MAX_EP_STEPS=maxSteps, BATCH_SIZE=64,
    MEMORY_CAPACITY=args.memoryCapacity, ARCHITECTURE=args.architecture,
    ACTIVATION=args.actType, GAMMA=args.gamma, GAMMA_PERIOD=updatePeriod,
    GAMMA_END=GAMMA_END, EPS_PERIOD=EPS_PERIOD, EPS_DECAY=0.7,
    EPS_RESET_PERIOD=EPS_RESET_PERIOD, LR_C=args.learningRate,
    LR_C_PERIOD=updatePeriod, LR_C_DECAY=0.8, MAX_MODEL=100,
    SELECT_WORST_Q=args.selectWorstQ, FIND_MAX_Q=args.findMaxQ,
    SIM_MAX_Q=args.simMaxQ
)

# args.architecture = [120,20]
# ADV_CONFIG = ceConfig(
#     DEVICE=device, ENV_NAME=env_name, SEED=args.randomSeed,
#     MAX_UPDATES=maxUpdates, MAX_EP_STEPS=maxSteps, BATCH_SIZE=64,
#     MEMORY_CAPACITY=args.memoryCapacity, ARCHITECTURE=args.architecture,
#     ACTIVATION=args.actType, GAMMA=args.gamma, GAMMA_PERIOD=updatePeriod,
#     GAMMA_END=GAMMA_END, EPS_PERIOD=EPS_PERIOD, EPS_DECAY=0.7,
#     EPS_RESET_PERIOD=EPS_RESET_PERIOD, LR_C=args.learningRate,
#     LR_C_PERIOD=updatePeriod, LR_C_DECAY=0.8, MAX_MODEL=100, NUM_CRITICS=args.numCritics
# )

# == TRAINER ==
trainer = Trainer(PRO_CONFIG)

# == PROTAGONIST AGENT ==
dimList = [stateDim] + PRO_CONFIG.ARCHITECTURE + [actionNum]
protagonist = DDQNSingle(
    PRO_CONFIG, actionNum, trainer.memory, dimList=dimList, mode=agentMode,
    terminalType=args.terminalType
)
print("We want to use: {}, and Agent uses: {}".format(device, protagonist.device))
print("Critic is using cuda: ", next(protagonist.Q_network.parameters()).is_cuda)

if args.warmup:
  print("\n== Warmup Q ==")
  lossList = protagonist.initQ(
      env, args.warmupIter, outFolder, num_warmup_samples=200, vmin=vmin,
      vmax=vmax, plotFigure=plotFigure, storeFigure=storeFigure
  )

# == ADVERSARY AGENT ==
# dimList = [stateDim + 1] + ADV_CONFIG.ARCHITECTURE + [actionNum] # +1 for sending the action into the adversary network 
# adversary = CriticEnsemble(
#     ADV_CONFIG, actionNum, trainer.memory, dimList=dimList, mode=agentMode,
#     terminalType=args.terminalType
# )
# adversary = DDQNSingle(
#     ADV_CONFIG, actionNum, trainer.memory, dimList=dimList, mode=agentMode,
#     terminalType=args.terminalType
# )
# print("We want to use: {}, and Agent uses: {}".format(device, adversary.device))
# print("Critic is using cuda: ", next(adversary.Q_network.parameters()).is_cuda)

# if args.warmup:
#   print("\n== Warmup Q ==")
#   lossList = adversary.initQ(
#       env, args.warmupIter, outFolder, num_warmup_samples=200, vmin=vmin,
#       vmax=vmax, plotFigure=plotFigure, storeFigure=storeFigure
#   )



#   if plotFigure or storeFigure:
#     fig, ax = plt.subplots(1, 1, figsize=(4, 4))
#     tmp = np.arange(500, args.warmupIter)
#     # tmp = np.arange(args.warmupIter)
#     ax.plot(tmp, lossList[tmp], 'b-')
#     ax.set_xlabel('Iteration', fontsize=18)
#     ax.set_ylabel('Loss', fontsize=18)
#     plt.tight_layout()

#     if storeFigure:
#       figurePath = os.path.join(figureFolder, 'initQ_Loss.png')
#       fig.savefig(figurePath)
#     if plotFigure:
#       plt.show()
#       plt.pause(0.001)
#     plt.close()

print("\n== Training Information ==")
vmin = -1 * args.scaling
vmax = 1 * args.scaling
checkPeriod = args.checkPeriod
trainRecords, trainProgress = trainer.learn(protagonist,
    env, MAX_UPDATES=maxUpdates, MAX_EP_STEPS=maxSteps, warmupQ=False,
    doneTerminate=True, vmin=vmin, vmax=vmax, numRndTraj=10000,
    checkPeriod=checkPeriod, outFolder=outFolder, plotFigure=args.plotFigure,
    storeFigure=args.storeFigure
)

trainDict = {}
trainDict['trainRecords'] = trainRecords
trainDict['trainProgress'] = trainProgress
filePath = os.path.join(outFolder, 'train')

if plotFigure or storeFigure:
  # = loss
  fig, axes = plt.subplots(1, 2, figsize=(8, 4))

  data = trainRecords
  ax = axes[0]
  ax.plot(data, 'b:')
  ax.set_xlabel('Iteration (x 1e5)', fontsize=18)
  ax.set_xticks(np.linspace(0, maxUpdates, 5))
  ax.set_xticklabels(np.linspace(0, maxUpdates, 5) / 1e5)
  ax.set_title('loss_critic', fontsize=18)
  ax.set_xlim(left=0, right=maxUpdates)

  data = trainProgress[:, 0]
  ax = axes[1]
  x = np.arange(data.shape[0]) + 1
  ax.plot(x, data, 'b-o')
  ax.set_xlabel('Index', fontsize=18)
  ax.set_xticks(x)
  # ax.set_xticklabels(np.arange(data.shape[0]) + 1)
  ax.set_title('Success Rate', fontsize=18)
  ax.set_xlim(left=1, right=data.shape[0])

  fig.tight_layout()
  if storeFigure:
    figurePath = os.path.join(figureFolder, 'train_loss_success.png')
    fig.savefig(figurePath)
  if plotFigure:
    plt.show()
    plt.pause(0.001)
  plt.close()

  # = value_rollout_action
  idx = np.argmax(trainProgress[:, 0]) + 1
  successRate = np.amax(trainProgress[:, 0])
  print('We pick model with success rate-{:.3f}'.format(successRate))
  protagonist.restore(idx * args.checkPeriod, outFolder, prefix="pro_model")
#   adversary.restore(idx * args.checkPeriod, outFolder, prefix="adv_model")

  nx = 41
  ny = 121
  xs = np.linspace(env.unwrapped.bounds[0, 0], env.unwrapped.bounds[0, 1], nx)
  ys = np.linspace(env.unwrapped.bounds[1, 0], env.unwrapped.bounds[1, 1], ny)

  resultMtx = np.empty((nx, ny), dtype=int)
  actDistMtx = np.empty((nx, ny), dtype=int)
  disturbDistMtx = np.empty((nx, ny), dtype=int)
  it = np.nditer(resultMtx, flags=['multi_index'])
  analytic_max_fail = 0
  analytic_min_fail = 0

  while not it.finished:
    idx = it.multi_index
    print(idx, end='\r')
    x = xs[idx[0]]
    y = ys[idx[1]]

    state = np.array([x, y])
    stateTensor = torch.FloatTensor(state).to(device).unsqueeze(0)
    action_index = protagonist.Q_network(stateTensor).min(dim=1)[1].cpu().item()
    if args.testMaxQ:
        disturb_index = protagonist.Q_network(stateTensor).max(dim=1)[1].cpu().item()
    elif not args.testMaxQ:
        disturb_index = protagonist.select_action(state, env, agent='adv', explore=False)
    actDistMtx[idx] = action_index
    disturbDistMtx[idx] = disturb_index

    _, _, result = env.unwrapped.simulate_one_trajectory(
        protagonist.Q_network, T=250, state=state
    )
    
    g_x = env.unwrapped.safety_margin(state)
    inside_max_diag = env.unwrapped.is_inside_diagonal_region(state)
    inside_min_diag = env.unwrapped.is_inside_diagonal_region(state, min=True)
    if g_x > 0:
      analytic_min_fail += 1
      analytic_max_fail += 1
    else:
      if inside_max_diag:
        analytic_max_fail += 1
      if inside_min_diag: 
        analytic_min_fail += 1

    resultMtx[idx] = result
    it.iternext()

  print('Analytical Success Rate for Maximal Disturbances-{:.3f}'.format(1 - analytic_max_fail/(nx*ny)))
  print('Analytical Success Rate for No Disturbances-{:.3f}'.format(1 - analytic_min_fail/(nx*ny)))
  print('Best RA Success Rate with Adversarial Disturbances-{:.3f}'.format((resultMtx == 1).sum()/(nx*ny)))

  cfg = PlotConfig(nx=nx, ny=ny, xs=xs, ys=ys, vmin=vmin, vmax=vmax, resultMtx=resultMtx, 
                   actDistMtx=actDistMtx, disturbDistMtx=disturbDistMtx, figureFolder=figureFolder, 
                   plotFigure=plotFigure, storeFigure=storeFigure, actionNum=actionNum)

  plot_RA_eval(env, protagonist, cfg)

  plot_protagonist_adversary_actions(env, cfg)

#   plot_protagonist_adversary_values(env, protagonist, adversary, cfg)
  

  trainDict['resultMtx'] = resultMtx
  trainDict['actDistMtx'] = actDistMtx
  trainDict['disturbDistMtx'] = disturbDistMtx

save_obj(trainDict, filePath)
