"""
Please contact the author(s) of this library if you have any questions.
Authors: George Nehma ( gnehma2020@fit.edu )

Ensemble variant of sim_new_point_mass.py. Protagonist is a DDQNEnsemble
(N critics, seed-diversified) instead of DDQNSingle. Everything else —
Trainer, adversary, env, config loading — is identical to sim_new_point_mass.py.

After the standard training / eval block an extra section computes and saves
the epistemic uncertainty maps (variance of Q across critics, conservative
min-Q value, and safe/unsafe classification disagreement).

Examples:
    python3 sim_ensemble.py --config config.yaml
    python3 sim_ensemble.py --config config_5critics.yaml
"""

import os
import shutil
import time
from warnings import simplefilter

import gymnasium as gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import yaml
from types import SimpleNamespace

# CHANGED: DDQNEnsemble as protagonist
from RARL.SACTrainer import SACTrainer
from RARL.SAC import SAC
from RARL.config import ceConfig
from RARL.utils import save_obj
from utils.utils import (
    plot_protagonist_adversary_actions,
    plot_RA_eval,
    PlotConfig,
)
from gym_reachability import gym_reachability    # noqa: F401 — registers envs

import argparse

matplotlib.use('Agg')
simplefilter(action='ignore', category=FutureWarning)
timestr = time.strftime("%Y-%m-%d-%H_%M_%S")

# == ARGS — YAML config, same as sim_new_point_mass.py ==
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config.yaml", type=str)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to load (for evaluation or resuming training).")
parser.add_argument("--checkpointVal", type=int, default=50000, help="Checkpoint iteration to load for evaluation (ignored if --checkpoint not provided).")
parser.add_argument(
    "--num_jobs", type=int, default=4,
    help="Total number of sim_ensemble.py processes running in parallel. "
         "Used to divide CPU threads evenly across jobs."
)
script_args = parser.parse_args()

# -- Pin PyTorch threads so parallel jobs don't fight over cores.
# With N jobs on a C-core machine each job gets C/N threads.
# e.g. 16 jobs on 64 cores → 4 threads each → all 64 cores busy, no contention.
_total_cores = os.cpu_count() or 1
_threads_per_job = max(1, _total_cores // script_args.num_jobs)
torch.set_num_threads(_threads_per_job)
torch.set_num_interop_threads(max(1, _threads_per_job // 2))
print(f"[Thread config] {_total_cores} cores / {script_args.num_jobs} jobs "
      f"= {_threads_per_job} threads per job")

if script_args.checkpoint is not None:
    with open(os.path.join(script_args.checkpoint, "init_configs.yaml"), "r") as f:
        config_dict = yaml.safe_load(f)
else:
    with open(script_args.config, "r") as f:
        config_dict = yaml.safe_load(f)

args = SimpleNamespace(**{k: v for section in config_dict.values() for k, v in section.items()})
print(args)

# == CONFIGURATION ==
env_name     = args.envName
device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_envs      = args.numEnvs

if env_name =="zermelo_show-v0":
    env_title = "point-mass" 
elif env_name == "one_player_reach_avoid_lunar_lander":
    env_title = "lunar-lander"
elif env_name == "cont-obs-avoid-v0":
    env_title = "cont-obs-avoid"

updatePeriod = int(args.maxUpdates / args.updateTimes)
                           
storeFigure  = args.storeFigure
plotFigure   = args.plotFigure

if script_args.checkpoint is not None:
    outFolder = script_args.checkpoint
elif args.showTime:
    fn = args.name + args.doneType
    fn = fn + '-' + timestr
    outFolder    = os.path.join(args.outFolder, 'ensemble', str(args.numCritics)+ "critics", env_title, 'SAC', args.mode, fn)

figureFolder = os.path.join(outFolder, 'figure')
os.makedirs(figureFolder, exist_ok=True)
print(outFolder)

# == Epsilon / gamma schedule — identical to sim_new_point_mass.py ==
if args.mode == 'RA':
    agentMode = 'RA'
    if args.annealing:
        GAMMA_END        = 0.999999
        EPS_PERIOD       = int(args.updatePeriodEps / 10)
        EPS_RESET_PERIOD = args.updatePeriodEps
    else:
        GAMMA_END        = 0.999
        EPS_PERIOD       = args.updatePeriodEps
        EPS_RESET_PERIOD = args.maxUpdates

elif args.mode == 'AARA':
    agentMode = 'AARA'
    if args.annealing:
        GAMMA_END        = 0.999999
        EPS_PERIOD       = int(args.updatePeriodEps / 10)
        EPS_RESET_PERIOD = args.updatePeriodEps
    else:
        GAMMA_END        = 0.999
        EPS_PERIOD       = args.updatePeriodEps
        EPS_RESET_PERIOD = args.maxUpdates

sample_inside_obs = False

# == Environment — passes config=args like updated sim_new_point_mass.py ==
print("\n== Environment Information ==")
train_envs = gym.make_vec(
    env_name, num_envs=num_envs, config=args, device=device,
    sample_inside_obs=sample_inside_obs
)
# train_envs = gym.vector.SyncVectorEnv(
#     [lambda: gym.make(env_name, config=args, device=device,
#     sample_inside_obs=sample_inside_obs) for _ in range(num_envs)],
#     autoreset_mode=gym.vector.AutoresetMode.SAME_STEP
# )
eval_env = gym.make(
    env_name, config=args, device=device,
    sample_inside_obs=sample_inside_obs
)

stateDim    = eval_env.unwrapped.state.shape[0]
actionNum   = eval_env.unwrapped.action_space.shape[0]
action_list = np.arange(actionNum)
print("State Dimension: {:d}, ActionSpace Dimension: {:d}".format(stateDim, actionNum))
print(f"Control Range: low = {eval_env.unwrapped.action_space.low}, high = {eval_env.unwrapped.action_space.high}")
print(f"Disturbance Range: low = {eval_env.unwrapped.disturbance_space.low}, high = {eval_env.unwrapped.disturbance_space.high}")

eval_env.unwrapped.set_costParam(args.penalty, args.reward, args.costType, args.scaling)

train_envs.reset(seed=args.randomSeed)
eval_env.reset(seed=args.randomSeed)

# == Environment margin plots (unchanged from sim_new_point_mass.py) ==
if plotFigure or storeFigure:
    nx, ny = 101, 101
    vmin = -1 * args.scaling
    vmax =  1 * args.scaling
    v    = np.zeros((nx, ny))
    l_x  = np.zeros((nx, ny))
    g_x  = np.zeros((nx, ny))
    xs = np.linspace(eval_env.unwrapped.bounds[0, 0], eval_env.unwrapped.bounds[0, 1], nx)
    ys = np.linspace(eval_env.unwrapped.bounds[1, 0], eval_env.unwrapped.bounds[1, 1], ny)
    it = np.nditer(v, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        x, y       = xs[idx[0]], ys[idx[1]]
        g_x[idx]   = eval_env.unwrapped.safety_margin(np.array([x, y]))
        l_x[idx]   = eval_env.unwrapped.target_margin(np.array([x, y])) # + g_x[idx]
        v[idx]     = np.maximum(l_x[idx], g_x[idx])
        it.iternext()

    vmin = round(max(abs(g_x.min()),abs(l_x.max())),1)
    vmax = -vmin
    axStyle = eval_env.unwrapped.get_axes()
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    for ax, data, title in zip(
        axes, [l_x, g_x, v],
        [r'$\ell(x)$', r'$g(x)$', r'$v(x)$'],
    ):
        # vmin = round(data.min(),1)
        # vmax = round(data.max(),1)
        im = ax.imshow(
            data.T, interpolation='none', extent=axStyle[0], origin="lower",
            cmap="seismic", vmin=vmin, vmax=vmax,
        )
        cbar = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.05, shrink=.95,
                            ticks=[vmin, 0, vmax])
        cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=24)
        ax.set_title(title, fontsize=18)
        eval_env.unwrapped.plot_target_failure_set(ax=ax)
        # env.unwrapped.plot_formatting(ax=ax)
    # env.unwrapped.plot_reach_avoid_set(axes[2])
    fig.tight_layout()
    if storeFigure:
        fig.savefig(os.path.join(figureFolder, 'env.png'))
    if plotFigure:
        plt.show()
        plt.pause(0.001)
    plt.close()

# == Agent CONFIG ==
print("\n== Agent Information ==")

# CHANGED: protagonist uses ceConfig (superset of dqnConfig with NUM_CRITICS)
# New fields SELECT_WORST_Q, FIND_MAX_Q, SIM_MAX_Q match updated dqnConfig

CONFIG = ceConfig(
    DEVICE=device, 
    ENV_NAME=env_name, 
    SEED=args.randomSeed,
    MAX_UPDATES=args.maxUpdates, 
    MAX_EP_STEPS=args.maxSteps,
    BATCH_SIZE=args.batchSize,
    MEMORY_CAPACITY=args.memoryCapacity, 
    ARCHITECTURE=args.architecture,
    ACTIVATION=args.actType, 
    # =================== LEARNING RATE .
    GAMMA=args.gamma, 
    GAMMA_PERIOD=args.updatePeriodGamma,
    GAMMA_END=GAMMA_END, 
    GAMMA_DECAY=args.gammaDecay,
    # =================== EXPLORATION PARAMS.
    EPSILON=args.eps,
    EPS_END=args.epsEnd,
    EPS_PERIOD=EPS_PERIOD, 
    EPS_DECAY=args.epsDecay,
    EPS_RESET_PERIOD=EPS_RESET_PERIOD, 
    # =================== LEARNING RATE PARAMS.
    LR_C= args.learningRate,
    LR_C_END= args.learningRate * 0.5,
    LR_C_PERIOD=args.updatePeriodLr, 
    LR_C_DECAY=args.learningRateDecay, 
    # =================== LEARNING RATE PARAMS.
    LR_A= args.learningRateActor,
    LR_A_END= args.learningRateActor * 0.5,
    LR_A_PERIOD=args.updatePeriodLr, 
    LR_A_DECAY=args.learningRateDecay, 
    # ===================
    MAX_MODEL=args.maxModel,
    NUM_CRITICS=args.numCritics,
    SELECT_WORST_Q=args.selectWorstQ,
    FIND_MAX_Q=args.findMaxQ,
    SIM_MAX_Q=args.simMaxQ, 
    TIME_STEP=args.timeStep,
    TAU=args.tau,
    HARD_UPDATE=args.hardUpdate,
    SOFT_UPDATE=args.softUpdate,
    RENDER=args.render,
    DOUBLE=args.double,
    REWARD=args.reward,
    PENALTY=args.penalty,
    ALPHA=args.alpha,
    POLICY=args.policy,
    TARGET_UPDATE_INTERVAL=args.targetUpdateInterval,
    AUTO_ALPHA_TUNING=args.autoAlphaTuning
)
if script_args.checkpoint is None:
    with open(os.path.join(outFolder,'init_configs.yaml'), "w") as f:
        yaml.dump(config_dict, f, sort_keys=False)

# == PROTAGONIST — DDQNEnsemble ==========================================
# CHANGED: DDQNSingle → DDQNEnsemble; dimList and call signature identical
dimList     = [stateDim] + CONFIG.ARCHITECTURE + [actionNum]
sacAgent = SAC(CONFIG, dimList=dimList, action_space=eval_env.unwrapped.action_space, disturbance_space=eval_env.unwrapped.disturbance_space)  
print(sacAgent)
print("We want to use: {}, and Agent uses: {}".format(device, sacAgent.device))
print("Critic is using cuda: ", next(sacAgent.critic.parameters()).is_cuda)

if script_args.checkpoint is not None:
    sacAgent.load_checkpoint(script_args.checkpointVal, outFolder, evaluate=True)
    print(f"Loaded checkpoint from {script_args.checkpoint} at iteration {script_args.checkpointVal}")
    itr_init = script_args.checkpointVal
    args.maxUpdates += 1000000
else:
    itr_init = 0


# if args.warmup:
#     print("\n== Warmup Q (protagonist ensemble) ==")
#     lossList = protagonist.initQ(
#         env, args.warmupIter, outFolder,
#         num_warmup_samples=200, vmin=vmin, vmax=vmax,
#         plotFigure=plotFigure, storeFigure=storeFigure,
#     )

# == TRAINER ==
trainer = SACTrainer(sacAgent, CONFIG)

# == TRAINING — Trainer.learn unchanged ==================================
print("\n== Training Information ==")
vmin        = -1 * args.scaling
vmax        =  1 * args.scaling
checkPeriod = args.checkPeriod

trainRecords, trainProgress = trainer.learn(
    train_envs, eval_env, MAX_UPDATES=args.maxUpdates, MAX_EP_STEPS=args.maxSteps, warmupQ=False,
    doneTerminate=True, vmin=vmin, vmax=vmax, numRndTraj=10000,
    checkPeriod=checkPeriod, outFolder=outFolder, storeBest=args.storeBest,
    plotFigure=args.plotFigure, storeFigure=args.storeFigure, 
    plotTrainValue=args.plotTrainValue, verbose=True, runningCostThr=None, itr_init=itr_init,
)

trainDict = {
    'trainRecords':  trainRecords,
    'trainProgress': trainProgress,
}
filePath = os.path.join(outFolder, 'train')

# == POST-TRAINING: loss / success curves + rollout eval =================
# Mirrors sim_new_point_mass.py: everything below is inside plotFigure or
# storeFigure guard, plus ensemble-specific uncertainty maps appended after.
if plotFigure or storeFigure:

    # -- training curves ---------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    ax = axes[0]
    ax.plot(trainRecords[:,0], 'b:')
    ax.set_xlabel('Iteration (x 1e5)', fontsize=18)
    ax.set_xticks(np.linspace(0, args.maxUpdates, 5))
    ax.set_xticklabels(np.linspace(0, args.maxUpdates, 5) / 1e5)
    ax.set_title('loss_critic', fontsize=18)
    ax.set_xlim(left=0, right=args.maxUpdates)

    ax = axes[1]
    ax.plot(trainRecords[:,1], 'r')
    ax.set_xlabel('Iteration (x 1e5)', fontsize=18)
    ax.set_xticks(np.linspace(0, args.maxUpdates, 5))
    ax.set_xticklabels(np.linspace(0, args.maxUpdates, 5) / 1e5)
    ax.set_title('Epistemic Uncertainty', fontsize=18)
    ax.set_xlim(left=0, right=args.maxUpdates)

    data = trainProgress[:, 0]
    ax   = axes[2]
    x    = np.arange(data.shape[0]) + 1
    ax.plot(x, data, 'b-o')
    ax.set_xlabel('Index', fontsize=18)
    ax.set_xticks(x)
    ax.set_title('Success Rate', fontsize=18)
    ax.set_xlim(left=1, right=data.shape[0])
    fig.tight_layout()
    if storeFigure:
        fig.savefig(os.path.join(figureFolder, 'train_loss_value_success.png'))
    if plotFigure:
        plt.show()
        plt.pause(0.001)
    plt.close()

    # -- restore best checkpoint ------------------------------------------
    idx         = np.argmax(trainProgress[:, 0]) + 1
    successRate = np.amax(trainProgress[:, 0])
    print('We pick model with success rate-{:.3f}'.format(successRate))
    # CHANGED: ensemble restore saves per-critic with unique prefixes
    sacAgent.load_checkpoint(idx * args.checkPeriod, outFolder, evaluate=True)

    # -- grid rollout eval (identical to sim_new_point_mass.py) -----------
    nx = 41
    ny = 121
    na = env.unwrapped.action_space.shape[0]
    nd = env.unwrapped.disturbance_space.shape[0]
    xs = np.linspace(env.unwrapped.bounds[0, 0], env.unwrapped.bounds[0, 1], nx)
    ys = np.linspace(env.unwrapped.bounds[1, 0], env.unwrapped.bounds[1, 1], ny)

    resultMtx      = np.empty((nx, ny), dtype=int)
    actDistMtx     = np.empty((nx, ny, na), dtype=float)
    disturbDistMtx = np.empty((nx, ny, nd), dtype=float)
    # ADDED: uncertainty matrices
    varMtx         = np.empty((nx, ny))
    disAgreeMtx    = np.empty((nx, ny))

    analytic_max_fail = 0
    analytic_min_fail = 0

    it = np.nditer(resultMtx, flags=['multi_index'])
    while not it.finished:
        idx_cell = it.multi_index
        print(idx_cell, end='\r')
        x, y  = xs[idx_cell[0]], ys[idx_cell[1]]
        state = np.concatenate([np.array([x, y, 0, 0]), env.unwrapped.obs_list])

        stateTensor  = torch.FloatTensor(state).to(device).unsqueeze(0)
        _, _, action = sacAgent.protagonist.sample(stateTensor)
        _, _, disturbance = sacAgent.adversary.sample(stateTensor)

        actDistMtx    [idx_cell] = action.cpu().detach().numpy()
        disturbDistMtx[idx_cell] = disturbance.cpu().detach().numpy()

        # ADDED: epistemic uncertainty at this state
        unc = sacAgent.get_uncertainty(stateTensor, action, disturbance)
        varMtx    [idx_cell] = unc["epistemic_uncertainty"].item()
        disAgreeMtx[idx_cell] = unc["safe_disagreement"].item()

        _, result, _, _ = env.unwrapped.simulate_one_trajectory(
            sacAgent, T=250, state=state
        )

        g_x_val         = env.unwrapped.safety_margin(state)
        # inside_max_diag = env.unwrapped.is_inside_diagonal_region(state)
        # inside_min_diag = env.unwrapped.is_inside_diagonal_region(state, min=True)
        if g_x_val > 0:
            analytic_min_fail += 1
            analytic_max_fail += 1
        # else:
        #     if inside_max_diag: analytic_max_fail += 1
        #     if inside_min_diag: analytic_min_fail += 1

        resultMtx[idx_cell] = result
        it.iternext()

    print('Analytical Success Rate for Maximal Disturbances-{:.3f}'.format(
        1 - analytic_max_fail / (nx * ny)))
    print('Analytical Success Rate for No Disturbances-{:.3f}'.format(
        1 - analytic_min_fail / (nx * ny)))
    print('Best RA Success Rate with Adversarial Disturbances-{:.3f}'.format(
        (resultMtx == 1).sum() / (nx * ny)))

    cfg = PlotConfig(
        nx=nx, ny=ny, xs=xs, ys=ys, vmin=vmin, vmax=vmax,
        resultMtx=resultMtx, actDistMtx=actDistMtx,
        disturbDistMtx=disturbDistMtx, figureFolder=figureFolder,
        plotFigure=plotFigure, storeFigure=storeFigure, actionNum=actionNum,
    )
    plot_RA_eval(env, sacAgent, cfg)
    # plot_protagonist_adversary_actions(env, cfg)

    # -- ADDED: 4-panel epistemic uncertainty figure ----------------------
    sacAgent.plot_uncertainty_maps(
        env,
        out_folder=figureFolder,
        nx=41, ny=ny,
        vmin=vmin, vmax=vmax,
        store=storeFigure,
        show=plotFigure,
    )

    # -- ADDED: uncertainty overlay on rollout grid -----------------------
    axStyle = env.unwrapped.get_axes()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    ax = axes[0]
    im = ax.imshow(
        varMtx.T, interpolation='none', extent=axStyle[0],
        origin='lower', cmap='YlOrRd', zorder=-1,
    )
    fig.colorbar(im, ax=ax, pad=0.01, fraction=0.05, shrink=.95)
    ax.set_title(r'Epistemic Uncertainty  Var$_k[Q]$', fontsize=14)
    env.unwrapped.plot_target_failure_set(ax=ax)
    # env.unwrapped.plot_reach_avoid_set(ax=ax)
    env.unwrapped.plot_formatting(ax=ax)

    ax = axes[1]
    im = ax.imshow(
        disAgreeMtx.T, interpolation='none', extent=axStyle[0],
        origin='lower', cmap='PuRd', vmin=0, vmax=1, zorder=-1,
    )
    fig.colorbar(im, ax=ax, pad=0.01, fraction=0.05, shrink=.95)
    ax.set_title('Safe / Unsafe Disagreement', fontsize=14)
    env.unwrapped.plot_target_failure_set(ax=ax)
    # env.unwrapped.plot_reach_avoid_set(ax=ax)
    env.unwrapped.plot_formatting(ax=ax)

    fig.suptitle(
        f"Protagonist Ensemble ({args.numCritics} critics | mode={args.mode})",
        fontsize=13,
    )
    fig.tight_layout()
    if storeFigure:
        fig.savefig(os.path.join(figureFolder, 'uncertainty_overlay.png'), dpi=150)
    if plotFigure:
        plt.show()
        plt.pause(0.001)
    plt.close()

    # -- save rollout matrices --------------------------------------------
    trainDict['resultMtx']      = resultMtx
    trainDict['actDistMtx']     = actDistMtx
    trainDict['disturbDistMtx'] = disturbDistMtx
    trainDict['varMtx']         = varMtx         # ADDED
    trainDict['disAgreeMtx']    = disAgreeMtx    # ADDED
    trainDict['numCritics']     = args.numCritics # ADDED

# Save uncertainty metrics pickle alongside the main train dict
# protagonist.save_metrics(
#     {'varMtx': varMtx if (plotFigure or storeFigure) else None,
#      'disAgreeMtx': disAgreeMtx if (plotFigure or storeFigure) else None},
#     out_folder=outFolder, tag='_final',
# )

save_obj(trainDict, filePath)
