"""
Action set: (LEFT, RIGHT, UP, DOWN and STAY).
"""

from __future__ import division

import math
from itertools import product

import numpy as np
import numpy.random as rn

from six.moves import range
from collections import Iterable
from matplotlib.patches import Rectangle

from .base import Domain, model_domain

from ..models.mdp import MDP
from ..models.mdp import TabularRewardFunction, LinearRewardFunction
from ..models.mdp import MDPTransition, MDPState, MDPAction

from ..utils.validation import check_random_state

from funzo.domains.gridworld import GridWorld, GTransition

__all__ = [
    'ObjectWorldMDP',
    'ObjectWorld',
    'OWReward',
    'GRewardLFA',
]

#############################################################################

# Cell status
FREE = 'free'
OBSTACLE = 'obstacle'
TERMINAL = 'terminal'

class OWReward(TabularRewardFunction):
    """ Grid world MDP reward function """
    def __init__(self, rmax=1.0, domain=None):
        super(OWReward, self).__init__(domain=domain,
                                      rmax=rmax)

        self._domain = model_domain(domain, ObjectWorld)
        self.grid_size = self._domain.grid_size
        self.objects = self._domain.objects

        R = np.zeros(len(self))
        for s in self._domain.states:
            state_ = self._domain.states[s]
            x, y = state_.cell

            # Note that it only support 2 colors for now
            near_c0 = False
            near_c1 = False

            for (dx, dy) in product(range(-3, 4), range(-3, 4)):
                if 0 <= x + dx < self.grid_size and 0 <= y + dy < self.grid_size:
                    if (abs(dx) + abs(dy) <= 3 and
                            (x+dx, y+dy) in self.objects and
                            self.objects[x+dx, y+dy].outer_colour == 0):
                        near_c0 = True
                    if (abs(dx) + abs(dy) <= 2 and
                            (x+dx, y+dy) in self.objects and
                            self.objects[x+dx, y+dy].outer_colour == 1):
                        near_c1 = True

            if near_c0 and near_c1:
                R[s] = 1
            elif near_c0:
                R[s] = -1
            else:
                R[s] = 0

        self.update_parameters(reward=R)

    def __call__(self, state, action):
        """ Evaluate reward function """
        return self._R[state]

    def __len__(self):
        return len(self._domain.states)

#############################################################################
class OWTransition(MDPTransition):
    """ GridWorld MDP controller """
    def __init__(self, wind=0.2, domain=None):
        super(OWTransition, self).__init__(domain)
        self._wind = wind
        self._domain = model_domain(domain, ObjectWorld)

    def __call__(self, state, action, **kwargs):
        """ Transition

        Returns
        --------
        A list of all possible next states [(prob, state)]

        """
        state_ = self._domain.states[state]
        action_ = self._domain.actions[action]
        p_s = 1.0 - self._wind
        p_f = self._wind / 2.0
        A = self._domain.actions.values()
        return [(p_s, self._move(state_, action_)),
                (p_f, self._move(state_, self._right(action_, A))),
                (p_f, self._move(state_, self._left(action_, A)))]

    def _move(self, state, action):
        """ Return the state that results from going in this direction.

        Stay in the same state if action os leading to go outside the world or
        to obstacles

        Returns
        --------
        new_state : int
            Id of the new state after transition (which can be the current
            state, if transition leads to outside of the world)

        """
        new_coords = (state.cell[0] + action.direction[0],
                      state.cell[1] + action.direction[1])

        if new_coords in self._domain.state_map:
            return self._domain.state_map[new_coords]

        return self._domain.state_map[state.cell]

    def _heading(self, heading, inc, directions):
        return directions[(directions.index(heading) + inc) % len(directions)]

    def _right(self, heading, directions):
        return self._heading(heading, -1, directions)

    def _left(self, heading, directions):
        return self._heading(heading, +1, directions)


class OWObject(object):
    """
    Object in objectworld.
    """

    def __init__(self, inner_colour, outer_colour):
        """
        inner_colour: Inner colour of object. int.
        outer_colour: Outer colour of object. int.
        -> OWObject
        """

        self.inner_colour = inner_colour
        self.outer_colour = outer_colour

    def __str__(self):
        """
        A string representation of this object.

        -> __str__
        """

        return "<OWObject (In: {}) (Out: {})>".format(self.inner_colour,
                                                      self.outer_colour)

class ObjectWorld(GridWorld):
    """
    Objectworld MDP.
    """

    def __init__(self, gmap):
        """
        gmap: path to map file.
        n_objects: Number of objects in the world. int.
        n_colours: Number of colours to colour objects with. int.
        -> Objectworld
        """

        super(ObjectWorld, self).__init__(gmap)

        # Load object
        self._obj = np.flipud(np.asarray(np.loadtxt("maps/map_a_object.txt")))
        self._inner = np.flipud(np.asarray(np.loadtxt("maps/map_a_inner.txt")))
        self._outer = np.flipud(np.asarray(np.loadtxt("maps/map_a_outer.txt")))

        self.grid_size = gmap.shape[0]
        self.n_objects = np.sum(self._obj == 1)

        self.objects = {}
        for x in range(0, gmap.shape[1]):
            for y in range(0, gmap.shape[0]):
                if self._obj[x, y] == 1:
                    obj = OWObject(int(self._inner[x,y]),int(self._outer[x,y]))
                    self.objects[y, x] = obj


class ObjectWorldMDP(MDP):
    """ Grid world MDP representing the decision making process """
    def __init__(self, reward, transition, discount=0.9, domain=None):
        super(ObjectWorldMDP, self).__init__(reward,
                                           transition,
                                           discount,
                                           domain)
        self._domain = model_domain(domain, ObjectWorld)

    @property
    def S(self):
        """ States of the MDP in an indexable container """
        return self._domain.states.keys()

    @property
    def A(self):
        """ Actions of the MDP in an indexable container """
        return self._domain.actions.keys()

    def actions(self, state):
        """ Get the set of actions available at a state """
        if self._domain.terminal(state):
            return [4]
        return self._domain.actions.keys()
