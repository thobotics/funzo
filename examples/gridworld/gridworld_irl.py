

from matplotlib import pyplot as plt
plt.style.use('fivethirtyeight')

import numpy as np

from funzo.domains.gridworld import GridWorld, GRewardLFA, GReward
from funzo.planners.dp import policy_iteration

from funzo.irl.birl.map_birl import MAPBIRL
from funzo.irl.birl.base import GaussianRewardPrior
from funzo.irl.base import PolicyLoss, RewardLoss

SEED = None


def main():
    gmap = np.loadtxt('maps/map_b.txt')
    # w = np.array([0.001, -0.1, 1.0])
    w = np.array([-0.001, -0.1, 1.0])

    g = GridWorld(gmap, reward_function=None, discount=0.9)
    # rfunc = GRewardLFA(g, weights=w)

    rfunc = GReward(g)
    g._reward = rfunc
    w = rfunc._R

    # ------------------------
    plan = policy_iteration(g, verbose=1)
    policy = plan['pi']
    print(policy)

    # fig = plt.figure(figsize=(8, 8))
    # ax = fig.gca()
    # ax = g.visualize(ax, policy=policy)
    # plt.show()

    demos = g.generate_trajectories(policy, num=50, random_state=SEED)
    # demos = g.generate_trajectories(policy, starts=[1, 4, 3], random_state=SEED)
    # np.save('demos.npy', demos)
    # demos = np.load('demos.npy')
    # print(demos)

    # IRL
    r_prior = GaussianRewardPrior(sigma=0.15)
    irl_solver = MAPBIRL(mdp=g, prior=r_prior, demos=demos,
                         planner=policy_iteration,
                         beta=0.6)
    r, data = irl_solver.run(V_E=plan['V'], random_state=SEED)

    g.reward.weights = r
    r_plan = policy_iteration(g)
    print(r_plan['pi'])
    print(r)
    V = r_plan['V']

    # compute the loss
    # loss_func = PolicyLoss(mdp=g, planner=policy_iteration, order=1)
    loss_func = RewardLoss(order=1)
    pi_loss = [loss_func(w, w_pi) for w_pi in data['rewards']]

    # ------------------------
    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()
    ax = g.visualize(ax, policy=r_plan['pi'])

    plt.figure(figsize=(8, 8))
    plt.imshow(V.reshape(gmap.shape),
               interpolation='nearest', cmap='viridis', origin='lower',
               vmin=np.min(V), vmax=np.max(V))
    plt.grid(False)
    plt.title('Value function')
    plt.colorbar()

    plt.figure(figsize=(8, 6))
    plt.plot(data['iter'], pi_loss)
    plt.ylabel('Loss function $\mathcal{L}_{\pi}$')
    plt.xlabel('Iteration')
    plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    main()
