import logging
from enum import Enum, IntEnum

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp

#DEFAULT_TRANSITION_PROBA = np.array([[0.2, 0.75, 0.05], [0.75, 0.05, 0.2], [0.05, 0.2, 0.75]]) # Low uncertainty
#DEFAULT_TRANSITION_PROBA = np.array([[0.2, 0.3, 0.5], [0.3, 0.5, 0.2], [0.5, 0.2, 0.3]]) # High uncertainty
DEFAULT_TRANSITION_PROBA = np.array([[0.75, 0.3, 0.05], [0.2, 0.4, 0.2], [0.05, 0.3, 0.75]])

class Action(IntEnum):
    LEFT = 0
    RIGHT = 1
    """MIDDLE = 1
    RIGHT = 2"""

class Tokens(IntEnum):
    RED = 0
    GREEN = 1
    BLUE = 2

class States(IntEnum):
    LEFT_UNAVAILABLE = 0
    MIDDLE_UNAVAILABLE = 1
    RIGHT_UNAVAILABLE = 2

class RGB_TOKEN:
    actions = [Action.LEFT, Action.RIGHT]#[Action.LEFT, Action.MIDDLE, Action.RIGHT]
    states = [States.LEFT_UNAVAILABLE, States.MIDDLE_UNAVAILABLE, States.RIGHT_UNAVAILABLE]
    def __init__(self, transition_probabilities=DEFAULT_TRANSITION_PROBA, valuable_tokens=[Tokens.RED], unavailable_action=None):
        self.M = transition_probabilities
        self.unavailable_action = unavailable_action
        self.current_state = unavailable_action
        ### Make states:
        """I = np.eye(3)
        I[unavailable_action] = 0.
        self.S = self.M @ I"""
        S = []
        for i in range(3):
            """I = np.eye(3)
            I[i] = 0.
            z = np.zeros((1, 3))
            z[:, i] = 1.
            S.append(np.concatenate((self.M @ I, z), axis=0))"""
            S.append(self.M[:, [j for j in range(3) if j != i]])
        ###
        self.S = np.array(S)
        ###
        self.goal_tokens = valuable_tokens
        ###
        self.token_rewards = np.zeros(3)#np.zeros(4)
        for vt in valuable_tokens:
            self.token_rewards[vt] = 10.
        ###
        self.token_rewards -= 10.
        ###
        self.Q = dict()
        self.__V = np.zeros(3)
        self._n_iterations = 1

    def reset(self, state):
        self.current_state = state # Availability of the slot machines
        return self.S[state]

    def execute(self, action):
        return self.__execute(action)

    def __execute(self, action):
        """ Execute action and collect the reward or penalty.

            :param Action action: Chosen slot
            :return float: reward which results from the chosen slot
        """
        assert self.current_state is not None, "current_state is None. Call reset with a valid state index before calling execute."
        tokens = []
        rewards = []
        S = self.S[self.current_state]
        for token, transition_proba in enumerate(S[:, action]):
            if transition_proba > 0.:
                tokens.append(token)
                rewards.append(self.token_rewards[token])
        ####
        return tokens, rewards

    def iter_policy_eval(self, n_iterations=None, eps=0.1, verbose=True, save=True):
        n_iterations = self._n_iterations if n_iterations is None else n_iterations
        for iter in range(n_iterations):
            D = 0.
            for state in RGB_TOKEN.states:
                v = self.__V[state]
                q_vals = []
                for action in RGB_TOKEN.actions:
                    val = np.dot(self.S[state, :, action], self.token_rewards)
                    self.Q[(state, action)] = val
                    q_vals.append(self.Q[(state, action)])
                ####
                self.__V[state] = np.mean(q_vals)
                D = max(D, np.abs(v - self.__V[state]))
            ###
            if verbose:
                print("-----------------")
                print("k =", iter+1)
                print(np.round(self.__V, 4), D)
            ####
            self.__V_target = self.__V.copy()
            if D < eps:
                break
        ####
        ##### Save Q-table
        filename = './rgbToken_qtable.npy'
        if verbose:
            print("Save Q-table as", filename)
        if save:
            np.save(filename, self.Q)
        #####
        return self.Q

    def get_target_optimal_joint_dist(self, unavail_slot_idx):
        """
         Following the control as inference setting,
        the target distribution p(\tau|O_{1:T}) \prop p(\tau, O_{1:T})
        with p(\tau, O_{1:T}) = p(s_1) \prod_{t=1}^{T} p(s_{t+1}|s_{t}, a_{t}) p(O_{t}=1|s_{t}, a_{t}) p(a_{t}|s_{t})
        which implies:
        p(s_{t+1}, a_{t}, O_{t}=1 | s_{t}) = p(s_{t+1}|s_{t}, a_{t}) p(O_{t}=1|s_{t}, a_{t}) p(a_{t}|s_{t})
         Note that:
        p(O_{t}=1 | s_{t}, a_{t}) = exp[r(s_{t}, a_{t})] where r(s_{t}, a_{t}) < 0, \forall t
        ####
        ++ In the RGB case, we have only one state. This is taken to be the initial state s_{1} = s_{t} and p(s_1) = 1.
        ++ The colored tokens are then taken to be the next states, so s_{t+1} = s_{2} \in \{ R, G, B \}
        ++ We have two actions (left or right slot). And the action probability p(a_{t}|s_{t}) is initially taken to be uniform.
        ++ The reward mostly depends on the next state (i.e. the obtained token), so exp[r(s_{t}, a_{t})] = exp[r(s_{t+1})]
         Hence, we get that the optimal target distribution is given by:
        p(s_{2}, a_{1} | s_{1}, O_{1}=1) \prop p(s_{2}, a_{1}, O_{1}=1 | s_{1})
        with:
            p(s_{2}, a_{1}, O_{1}=1 | s_{1}) = p(s_{2}|s_{1}, a_{1}) exp[r(s_{t+1})] / N_a
        where
            N_a = number of actions (here N_a = 2)
        """
        joint_p = np.zeros((3, 2, 2))
        max_r = max(self.token_rewards)
        for s in range(3):
            for a, action in enumerate(RGB_TOKEN.actions):
                for O in range(2):
                    r = self.token_rewards[s] - max_r # Adjust to values smaller than 0
                    p_O = ((1. - np.exp(r)) if O == 0 else np.exp(r))
                    joint_p[s, a, O] = self.S[unavail_slot_idx, s, action] * p_O * 0.5
                ####
            #####
        #####
        ## Reduce to the optimal intent and action target dist.: p(s_{2}, a_{1} | s_{1}, O_{1}=1)
        joint_p = joint_p[:, :, 1] / joint_p[:, :, 1].sum()
        #####
        return joint_p
