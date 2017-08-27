from __future__ import division

import argparse

import matplotlib
matplotlib.use('Qt4Agg')

from matplotlib import pyplot as plt
plt.style.use('fivethirtyeight')

import numpy as np

from funzo.domains.objectworld import ObjectWorld, ObjectWorldMDP
from funzo.domains.objectworld import OWTransition, OWReward#, OWRewardLFA
from funzo.planners.dp import PolicyIteration, ValueIteration


def main(map_name, planner):
    gmap = np.loadtxt(map_name)

    with ObjectWorld(gmap=gmap) as world:
        R = OWReward(rmax=10.0)
##        R = GRewardLFA(weights=[-0.01, -10.0, 1.0], rmax=1.0)
        T = OWTransition(wind=0.1)
        g_mdp = ObjectWorldMDP(reward=R, transition=T, discount=0.95)

        # ------------------------
        mdp_planner = PolicyIteration(max_iter=200, random_state=None)
        if planner == 'VI':
            mdp_planner = ValueIteration()

        res = mdp_planner.solve(g_mdp)
        V = res['V']
        print('Policy: ', res['pi'])

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()
    ax = world.visualize(ax, policy=res['pi'])

    # Plot value function
    plt.figure(figsize=(8, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(V.reshape(gmap.shape),
               interpolation='nearest', cmap='viridis', origin='lower',
               vmin=np.min(V), vmax=np.max(V))
    plt.grid(False)
    plt.title('Value function')
    plt.colorbar(orientation='horizontal')

    # Plot groundtruth reward
    xticks = np.arange(gmap.shape[0])
    yticks = np.arange(gmap.shape[1])
    plt.subplot(1, 2, 2)
    plt.imshow(R._R.reshape(gmap.shape), interpolation='nearest', origin='lower',
               cmap='viridis',extent=[0,gmap.shape[0],0,gmap.shape[1]])
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.grid(False)
    plt.title("Groundtruth reward")
    plt.colorbar(orientation='horizontal')
    
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--map", type=str, required=True,
                        help="Grid Map file")
    parser.add_argument("-p", "--planner", type=str, default="PI",
                        help="Planner to use: [PI, VI], default: PI")

    args = parser.parse_args()
    main(args.map, args.planner)
