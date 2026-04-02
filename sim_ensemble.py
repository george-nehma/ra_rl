"""
sim_ensemble.py
---------------
Train a critic ensemble for reach-avoid RL and evaluate epistemic uncertainty.

Mirrors ``sim_naive.py`` exactly in structure; the only additions are:
  -nc / --numCritics   number of ensemble members (default: 3)
  -es / --ensembleStrategy  action-selection strategy: mean | conservative | vote

Usage examples
--------------
# Default 3-critic ensemble, RA mode with gamma annealing
python3 sim_ensemble.py -w -sf -a -g 0.9 -mu 12000000 -cp 600000 -ut 20 -n anneal

# 5-critic ensemble, conservative action selection
python3 sim_ensemble.py -sf -g 0.9999 -n 9999 -nc 5 -es conservative

# Quick smoke test
python3 sim_ensemble.py -w -sf -wi 100 -mu 5000 -cp 2500 -nc 3 -n smoke
"""

import os
import argparse
import time
from warnings import simplefilter

import gymnasium as gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch

from RARL.DDQNSingle import DDQNSingle
from RARL.DDQNEnsemble import DDQNEnsemble
from RARL.config import dqnConfig
from RARL.utils import save_obj
from gym_reachability import gym_reachability  # noqa: F401 – registers envs

matplotlib.use('Agg')
simplefilter(action='ignore', category=FutureWarning)

timestr = time.strftime("%Y-%m-%d-%H_%M")

# =============================================================================
# CLI
# =============================================================================
parser = argparse.ArgumentParser()

# -- environment --------------------------------------------------------------
parser.add_argument(
    "-dt", "--doneType", help="when to raise done flag", default='toEnd', type=str
)
parser.add_argument(
    "-ct", "--costType", help="cost type", default='sparse', type=str
)
parser.add_argument(
    "-rnd", "--randomSeed", help="random seed", default=0, type=int
)
parser.add_argument(
    "-r", "--reward", help="reward when entering target set", default=-1, type=float
)
parser.add_argument(
    "-p", "--penalty", help="penalty when entering failure set", default=1, type=float
)
parser.add_argument(
    "-s", "--scaling", help="scaling of ell/g", default=4, type=float
)

# -- training -----------------------------------------------------------------
parser.add_argument("-w",  "--warmup",       help="warmup Q-network",      action="store_true")
parser.add_argument("-wi", "--warmupIter",   help="warmup iterations",      default=2000,   type=int)
parser.add_argument("-mu", "--maxUpdates",   help="max gradient updates",   default=400000, type=int)
parser.add_argument("-ut", "--updateTimes",  help="#hyper-param steps",     default=10,     type=int)
parser.add_argument("-mc", "--memoryCapacity", help="replay buffer size",   default=10000,  type=int)
parser.add_argument("-cp", "--checkPeriod",  help="checkpoint period",      default=20000,  type=int)

# -- network ------------------------------------------------------------------
parser.add_argument("-a",   "--annealing",    help="gamma annealing",        action="store_true")
parser.add_argument("-arc", "--architecture", help="hidden layer sizes",     default=[100, 20], nargs="*", type=int)
parser.add_argument("-lr",  "--learningRate", help="learning rate",          default=1e-3,  type=float)
parser.add_argument("-g",   "--gamma",        help="discount / contraction", default=0.9999, type=float)
parser.add_argument("-act", "--actType",      help="activation function",    default='Tanh', type=str)

# -- RL mode ------------------------------------------------------------------
parser.add_argument("-m",  "--mode",         help="RA or lagrange",         default='AARA',  type=str)
parser.add_argument("-tt", "--terminalType", help="terminal value type",    default='g',   type=str)

# -- ensemble -----------------------------------------------------------------
parser.add_argument(
    "-nc", "--numCritics",
    help="number of ensemble critics (default: 3)",
    default=3, type=int,
)
parser.add_argument(
    "-es", "--ensembleStrategy",
    help="action selection: mean | conservative | vote",
    default='mean', type=str,
    choices=['mean', 'conservative', 'vote'],
)

# -- file / output ------------------------------------------------------------
parser.add_argument("-st", "--showTime",    help="append timestamp to name", action="store_true")
parser.add_argument("-n",  "--name",        help="extra name tag",           default='',        type=str)
parser.add_argument("-of", "--outFolder",   help="output folder",            default='experiments', type=str)
parser.add_argument("-pf", "--plotFigure",  help="show figures interactively", action="store_true")
parser.add_argument("-sf", "--storeFigure", help="save figures to disk",      action="store_true")

args = parser.parse_args()
print(args)

# =============================================================================
# Config
# =============================================================================
env_name   = "zermelo_show-v0"
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
maxUpdates = args.maxUpdates
updateTimes = args.updateTimes
updatePeriod = int(maxUpdates / updateTimes)
maxSteps   = 250

fn = args.name + ('-' + args.doneType) + ('-' + args.costType if args.mode == 'lagrange' else '')
if args.showTime:
    fn += '-' + timestr

outFolder = os.path.join(
    args.outFolder, 'ensemble', f"{args.numCritics}critics", 'DDQN', args.mode, fn
)
figureFolder = os.path.join(outFolder, 'figure')
os.makedirs(figureFolder, exist_ok=True)
print("Output folder:", outFolder)

# -- gamma / epsilon schedule mirrors sim_naive.py exactly -------------------
if args.mode == 'lagrange':
    envMode   = 'normal'
    agentMode = 'normal'
    GAMMA_END = args.gamma
    EPS_PERIOD = updatePeriod
    EPS_RESET_PERIOD = maxUpdates
elif args.mode == 'AARA':
    envMode   = 'AARA'
    agentMode = 'AARA'
    if args.annealing:
        GAMMA_END        = 0.999999
        EPS_PERIOD       = int(updatePeriod / 10)
        EPS_RESET_PERIOD = updatePeriod
    else:
        GAMMA_END        = args.gamma
        EPS_PERIOD       = updatePeriod
        EPS_RESET_PERIOD = maxUpdates

sample_inside_obs = args.doneType == 'toEnd'

# =============================================================================
# Environment
# =============================================================================
print("\n== Environment Information ==")
env = gym.make(
    env_name, config=args, device=device, mode=envMode, doneType=args.doneType,
    sample_inside_obs=sample_inside_obs, envType='basic'
)
stateDim  = env.unwrapped.state.shape[0]
actionNum = env.unwrapped.action_space.n
action_list = np.arange(actionNum)
print(f"State Dim: {stateDim}, Action Dim: {actionNum}")
print(env.unwrapped.discrete_controls)

env.unwrapped.set_costParam(args.penalty, args.reward, args.costType, args.scaling)
env.unwrapped.set_seed(args.randomSeed)

vmin = -1 * args.scaling
vmax =  1 * args.scaling

# -- optional: plot environment margins (same as sim_naive.py) ----------------
if args.plotFigure or args.storeFigure:
    nx_env, ny_env = 101, 101
    v_env = np.zeros((nx_env, ny_env))
    l_x   = np.zeros((nx_env, ny_env))
    g_x   = np.zeros((nx_env, ny_env))
    xs_env = np.linspace(env.unwrapped.bounds[0, 0], env.unwrapped.bounds[0, 1], nx_env)
    ys_env = np.linspace(env.unwrapped.bounds[1, 0], env.unwrapped.bounds[1, 1], ny_env)
    it = np.nditer(v_env, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        x, y = xs_env[idx[0]], ys_env[idx[1]]
        l_x[idx] = env.unwrapped.target_margin(np.array([x, y]))
        g_x[idx] = env.unwrapped.safety_margin(np.array([x, y]))
        v_env[idx] = np.maximum(l_x[idx], g_x[idx])
        it.iternext()
    axStyle = env.unwrapped.get_axes()
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    for ax, data, title in zip(
        axes, [l_x, g_x, v_env],
        [r'$\ell(x)$', r'$g(x)$', r'$v(x)$']
    ):
        im = ax.imshow(
            data.T, interpolation='none', extent=axStyle[0], origin='lower',
            cmap='seismic', vmin=vmin, vmax=vmax,
        )
        fig.colorbar(im, ax=ax, pad=0.01, fraction=0.05, shrink=.95,
                     ticks=[vmin, 0, vmax])
        ax.set_title(title, fontsize=18)
        env.unwrapped.plot_target_failure_set(ax=ax)
        env.unwrapped.plot_formatting(ax=ax)
    if hasattr(env, 'plot_reach_avoid_set'):
        env.unwrapped.plot_reach_avoid_set(axes[2])
    fig.tight_layout()
    if args.storeFigure:
        fig.savefig(os.path.join(figureFolder, 'env.png'))
    if args.plotFigure:
        plt.show(); plt.pause(0.001)
    plt.close()

# =============================================================================
# Agent config (shared base; ensemble copies & bumps SEED per critic)
# =============================================================================
print("\n== Agent / Ensemble Configuration ==")
CONFIG = dqnConfig(
    DEVICE=device, ENV_NAME=env_name, SEED=args.randomSeed,
    MAX_UPDATES=maxUpdates, MAX_EP_STEPS=maxSteps, BATCH_SIZE=64,
    MEMORY_CAPACITY=args.memoryCapacity,
    ARCHITECTURE=args.architecture,
    ACTIVATION=args.actType,
    GAMMA=args.gamma, GAMMA_PERIOD=updatePeriod,
    GAMMA_END=GAMMA_END, EPS_PERIOD=EPS_PERIOD, EPS_DECAY=0.7,
    EPS_RESET_PERIOD=EPS_RESET_PERIOD,
    LR_C=args.learningRate, LR_C_PERIOD=updatePeriod, LR_C_DECAY=0.8,
    MAX_MODEL=100,
)

dimList = [stateDim] + CONFIG.ARCHITECTURE + [actionNum]

print(f"\nBuilding ensemble with {args.numCritics} critics ...")
ensemble = DDQNEnsemble(
    CONFIG, actionNum, action_list,
    dim_list=dimList,
    num_critics=args.numCritics,
    mode=agentMode,
    terminal_type=args.terminalType,
)
print(f"\n{ensemble}")

# =============================================================================
# Warmup (optional)
# =============================================================================
if args.warmup:
    print("\n== Warming up ensemble critics ==")
    all_warmup_losses = ensemble.init_q(
        env, args.warmupIter, outFolder,
        num_warmup_samples=200,
        vmin=vmin, vmax=vmax,
        plot_figure=args.plotFigure,
        store_figure=args.storeFigure,
    )

    if args.plotFigure or args.storeFigure:
        fig, axes = plt.subplots(1, args.numCritics,
                                 figsize=(5 * args.numCritics, 4))
        if args.numCritics == 1:
            axes = [axes]
        for i, (ax, losses) in enumerate(zip(axes, all_warmup_losses)):
            tmp = np.arange(500, args.warmupIter)
            ax.plot(tmp, losses[tmp], 'b-')
            ax.set_xlabel('Iteration', fontsize=14)
            ax.set_ylabel('Loss', fontsize=14)
            ax.set_title(f'Critic {i} Warmup Loss', fontsize=14)
        fig.tight_layout()
        if args.storeFigure:
            fig.savefig(os.path.join(figureFolder, 'warmup_loss.png'))
        if args.plotFigure:
            plt.show(); plt.pause(0.001)
        plt.close()

# =============================================================================
# Training
# =============================================================================
print("\n== Training Ensemble ==")
all_records, all_progress = ensemble.learn(
    env,
    max_updates=maxUpdates,
    max_ep_steps=maxSteps,
    warmup_q=False,
    done_terminate=True,
    vmin=vmin, vmax=vmax,
    num_rnd_traj=10000,
    check_period=args.checkPeriod,
    out_folder=outFolder,
    plot_figure=args.plotFigure,
    store_figure=args.storeFigure,
)

# =============================================================================
# Restore best checkpoint per critic
# =============================================================================
print("\n== Restoring best checkpoints ==")
best_rates = ensemble.restore_best(all_progress, args.checkPeriod, outFolder)
print(f"  Mean success rate across ensemble: {np.mean(best_rates):.3f} "
      f"± {np.std(best_rates):.3f}")

# =============================================================================
# Training diagnostics – per-critic loss & success rate curves
# =============================================================================
if args.plotFigure or args.storeFigure:
    fig, axes = plt.subplots(2, args.numCritics,
                             figsize=(6 * args.numCritics, 8))
    if args.numCritics == 1:
        axes = axes.reshape(2, 1)

    for i, (records, progress) in enumerate(zip(all_records, all_progress)):
        # loss
        ax = axes[0, i]
        ax.plot(records, 'b:', alpha=0.8)
        ax.set_title(f'Critic {i} — Training Loss', fontsize=12)
        ax.set_xlabel('Update', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)
        ax.set_xlim(left=0, right=maxUpdates)

        # success rate
        ax = axes[1, i]
        x = np.arange(progress.shape[0]) + 1
        ax.plot(x, progress[:, 0], 'b-o', markersize=5)
        ax.axhline(best_rates[i], color='r', linestyle='--', alpha=0.7,
                   label=f'best={best_rates[i]:.3f}')
        ax.set_title(f'Critic {i} — Success Rate', fontsize=12)
        ax.set_xlabel('Checkpoint', fontsize=11)
        ax.set_ylabel('Success Rate', fontsize=11)
        ax.legend(fontsize=10)

    fig.tight_layout()
    if args.storeFigure:
        fig.savefig(os.path.join(figureFolder, 'ensemble_training.png'))
    if args.plotFigure:
        plt.show(); plt.pause(0.001)
    plt.close()

# =============================================================================
# Epistemic uncertainty maps
# =============================================================================
print("\n== Computing epistemic uncertainty maps ==")
ensemble.plot_uncertainty_maps(
    env,
    out_folder=figureFolder,
    nx=41, ny=41,
    store=args.storeFigure,
    show=args.plotFigure,
)

# =============================================================================
# Grid-level rollout evaluation using ensemble action selection
# =============================================================================
print(f"\n== Rollout evaluation (strategy='{args.ensembleStrategy}') ==")
nx_eval, ny_eval = 41, 41
xs_eval = np.linspace(env.unwrapped.bounds[0, 0], env.unwrapped.bounds[0, 1], nx_eval)
ys_eval = np.linspace(env.unwrapped.bounds[1, 0], env.unwrapped.bounds[1, 1], ny_eval)

resultMtx   = np.empty((nx_eval, ny_eval), dtype=int)
actDistMtx  = np.empty((nx_eval, ny_eval), dtype=int)
varMtx      = np.empty((nx_eval, ny_eval))
disAgreeMtx = np.empty((nx_eval, ny_eval))

it = np.nditer(resultMtx, flags=['multi_index'])
while not it.finished:
    idx = it.multi_index
    x, y = xs_eval[idx[0]], ys_eval[idx[1]]
    state = np.array([x, y], dtype=np.float32)
    st = torch.FloatTensor(state).unsqueeze(0).to(ensemble.device)

    # uncertainty
    metrics = ensemble.get_uncertainty(st)
    varMtx[idx]      = metrics["epistemic_uncertainty"].item()
    disAgreeMtx[idx] = metrics["safe_disagreement"].item()

    # action via ensemble strategy
    action_index = ensemble.act(st, strategy=args.ensembleStrategy)
    actDistMtx[idx] = action_index

    # rollout using mean Q network (first critic as proxy – or you can
    # build a thin wrapper; here we use critic 0 which is consistent with
    # the existing simulate_one_trajectory API)
    _, _, result = env.unwrapped.simulate_one_trajectory(
        ensemble.critics[0].Q_network, T=250, state=state, toEnd=False
    )
    resultMtx[idx] = result
    it.iternext()

# =============================================================================
# Rollout visualisation
# =============================================================================
if args.plotFigure or args.storeFigure:
    axStyle = env.unwrapped.get_axes()
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharex=True, sharey=True)

    # Value (mean Q)
    ax = axes[0]
    unc_maps = ensemble.get_uncertainty_map(env, nx=nx_eval, ny=ny_eval)
    im = ax.imshow(
        unc_maps['mean_v'].T, interpolation='none', extent=axStyle[0],
        origin='lower', cmap='seismic', vmin=vmin, vmax=vmax, zorder=-1,
    )
    ax.contour(xs_eval, ys_eval, unc_maps['mean_v'].T,
               levels=[0], colors='k', linewidths=2, linestyles='dashed')
    ax.set_xlabel('Mean Value', fontsize=14)

    # Rollout result
    ax = axes[1]
    ax.imshow(
        (resultMtx != 1).T, interpolation='none', extent=axStyle[0],
        origin='lower', cmap='seismic', vmin=0, vmax=1, zorder=-1,
    )
    env.unwrapped.plot_trajectories(
        ensemble.critics[0].Q_network, states=env.unwrapped.visual_initial_states,
        toEnd=True, ax=ax, c='w', lw=1.5,
    )
    ax.set_xlabel('Rollout RA', fontsize=14)

    # Epistemic uncertainty
    ax = axes[2]
    im = ax.imshow(
        varMtx.T, interpolation='none', extent=axStyle[0],
        origin='lower', cmap='YlOrRd', zorder=-1,
    )
    fig.colorbar(im, ax=ax, pad=0.01, fraction=0.05, shrink=.95)
    ax.set_xlabel(r'Epistemic Var$_k[Q]$', fontsize=14)

    # Critic disagreement on safe/unsafe
    ax = axes[3]
    im = ax.imshow(
        disAgreeMtx.T, interpolation='none', extent=axStyle[0],
        origin='lower', cmap='PuRd', vmin=0, vmax=1, zorder=-1,
    )
    fig.colorbar(im, ax=ax, pad=0.01, fraction=0.05, shrink=.95)
    ax.set_xlabel('Safe/Unsafe Disagreement', fontsize=14)

    for ax in axes:
        env.plot_target_failure_set(ax=ax)
        if hasattr(env, 'plot_reach_avoid_set'):
            env.plot_reach_avoid_set(ax)
        env.plot_formatting(ax=ax)

    fig.suptitle(
        f"Ensemble ({args.numCritics} critics) | strategy='{args.ensembleStrategy}'",
        fontsize=14,
    )
    fig.tight_layout()
    if args.storeFigure:
        fig.savefig(os.path.join(figureFolder, 'value_rollout_uncertainty.png'), dpi=150)
    if args.plotFigure:
        plt.show(); plt.pause(0.001)
    plt.close()

# =============================================================================
# Save all results
# =============================================================================
trainDict = {
    'numCritics':       args.numCritics,
    'ensembleStrategy': args.ensembleStrategy,
    'bestRates':        best_rates,
    'allRecords':       all_records,
    'allProgress':      all_progress,
    'resultMtx':        resultMtx,
    'actDistMtx':       actDistMtx,
    'varMtx':           varMtx,
    'disAgreeMtx':      disAgreeMtx,
}
save_obj(trainDict, os.path.join(outFolder, 'ensemble_train'))
print("\nDone. Results saved to:", outFolder)
