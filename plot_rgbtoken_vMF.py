import os
from rgb_token import RGB_TOKEN
import numpy as np
from scipy.special import logsumexp
import torch as th
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.categorical import Categorical
from torch.distributions.independent import Independent
#from hyperspherical_vae.distributions import VonMisesFisher
from torch.distributions.von_mises import VonMises

from scipy.special import i0, i1

import matplotlib as mp
mp.use("Agg")
#mp.rcParams["pdf.fonttype"] = 42
#mp.rcParams["ps.fonttype"] = 42
mp.rcParams["text.usetex"] = True
mp.rcParams.update({"font.size": 17})
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from matplotlib.patches import Rectangle

import argparse

################################################################################
parser = argparse.ArgumentParser(description="Parser")
parser.add_argument("--case", default=1, type=int, help="Task configurations/cases in [1, 2, 3 ,4] (default: 1)")
parser.add_argument("--hu", action="store_true", help="Run with high uncertainty transition probabilities")

################################################################################
LOW_UNCERTAINTY_CASES = {
    ## Case 1: Only one valuable token and only one correct the slot machine
    1: ("LU_case1_", np.array([[0.2, 0.75, 0.05], [0.75, 0.05, 0.2], [0.05, 0.2, 0.75]]), [1]),
    ## Case 2: Only one valuable token and the slot machines offer multiple ways to get them (here 2 ways to get green)
    2: ("LU_case2_", np.array([[0.2, 0.75, 0.05], [0.75, 0.05, 0.75], [0.05, 0.2, 0.2]]), [1]),
    ## Case 3: Multiple valuable tokens and only one correct slot machine for each valuable token
    3: ("LU_case3_", np.array([[0.2, 0.75, 0.05], [0.75, 0.05, 0.2], [0.05, 0.2, 0.75]]), [0, 2]),
    ## Case 4: Multiple valuable tokens and the slot machines offer multiple ways to get them
    #4: ("LU_case4_", np.array([[0.75, 0.3, 0.05], [0.2, 0.4, 0.2], [0.05, 0.3, 0.75]]), [0, 2]),
    4: ("LU_case4_", np.array([[0.15, 0.1, 0.75], [0.1, 0.8, 0.1], [0.75, 0.1, 0.15]]), [0, 2]),
}

HIGH_UNCERTAINTY_CASES = {
    ## Case 1: Only one valuable token and only one correct the slot machine
    1: ("HU_case1_", np.array([[0.2, 0.3, 0.5], [0.3, 0.5, 0.2], [0.5, 0.2, 0.3]]), [1]),
    ## Case 2: Only one valuable token and the slot machines offer multiple ways to get them (here 2 ways to get green)
    2: ("HU_case2_", np.array([[0.2, 0.3, 0.5], [0.3, 0.5, 0.3], [0.5, 0.2, 0.2]]), [1]),
    ## Case 3: Multiple valuable tokens and only one correct slot machine for each valuable token
    #3: ("HU_case3_", np.array([[0.2, 0.3, 0.5], [0.3, 0.5, 0.2], [0.5, 0.2, 0.3]]), [0, 2]),
    ## Case 3: Multiple valuable tokens and the slot machines offer multiple ways to get them
    3: ("HU_case3_", np.array([[0.3, 0.3, 0.5], [0.2, 0.5, 0.2], [0.5, 0.2, 0.3]]), [0, 2]),
    ## Case 4: Multiple valuable tokens and the slot machines offer multiple ways to get them
    4: ("HU_case4_", np.array([[0.5, 0.3, 0.2], [0.3, 0.4, 0.3], [0.2, 0.3, 0.5]]), [0, 2]),
}

################################################################################
def flatten(xss):
    return [x for xs in xss for x in xs]

def full_extent(ax, pad=0.0):
    """Get the full extent of an axes, including axes labels, tick labels, and
    titles."""
    # For text objects, we need to draw the figure first, otherwise the extents
    # are undefined.
    ax.figure.canvas.draw()
    items = []#ax.get_xticklabels() + ax.get_yticklabels()
    # items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
    items += [ax]#, ax.title]
    bbox = Bbox.union([item.get_window_extent() for item in items])
    return bbox.expanded(1.0 + pad, 1.0 + pad)

def get_argmin_solutions(KL_list):
    min_val = min(KL_list)
    min_ind = KL_list.argsort()[:4]
    minimum_values = []
    for idx in min_ind:
        if KL_list[idx] == min_val:
            minimum_values.append(x[idx])
    return min_val, np.array(minimum_values)

def get_analytical_argmin_forward_KL(mixture_means, mixture_kappas, F, N=100):
    ### mu
    A = [i1(mixture_kappas[i]) / (i0(mixture_kappas[i]) + 1e-8) for i in range(len(mixture_means))]
    S = sum([F[i] * A[i] * np.sin(mixture_means[i]) for i in range(len(mixture_means))])
    C = sum([F[i] * A[i] * np.cos(mixture_means[i]) for i in range(len(mixture_means))])
    fkl_sol_mu = np.arctan2(S, C)
    ### kappa
    R_bar = sum([F[i] * A[i] * np.cos(fkl_sol_mu - mixture_means[i]) for i in range(len(mixture_means))])
    fkl_sol_kappa = R_bar * (2. - R_bar**2) / (1. - R_bar**2)
    A_kappa = i1(fkl_sol_kappa) / (i0(fkl_sol_kappa) + 1e-8)
    #### Newton method iterations to get a more accurate inversion
    for i in range(N):
        fkl_sol_kappa -= (A_kappa - R_bar) / (1. - A_kappa**2 - A_kappa / fkl_sol_kappa)
        A_kappa = i1(fkl_sol_kappa) / (i0(fkl_sol_kappa) + 1e-8)
        if (A_kappa - R_bar) < 1e-6:
            break
    ###
    return fkl_sol_mu, fkl_sol_kappa + 1e-6 # add 1e-6 for numerical stability to avoid kappa being too small

def get_analytical_argmin_reverse_KL(mixture_means, mixture_kappas, F):
    ### This simply selects the global mode (It returns the first one if there are multiple ones)
    #### The modes occur at x = mu, so first, compute the likelihood for each component's mode, weighted by
    #### the mixture weights F
    p = [F[i] * np.exp(mixture_kappas[i]) / (2. * np.pi * (i0(mixture_kappas[i]) + 1e-8)) for i in range(len(mixture_means))]
    #### Then select the highest one as the solution
    idx = np.argmax(F)
    ###
    rkl_sol_mu, rkl_sol_kappa = mixture_means[idx], mixture_kappas[idx]
    ###
    return rkl_sol_mu, rkl_sol_kappa

def get_forward_reverse_KL(mix_d, type="both", bs=int(1e5)):
    FKL_list = []
    RKL_list = []
    ##
    if type in ["both", "fkl"]:
        s_for_fkl = mix_d.sample([bs])
        v_for_fkl = th.cat((s_for_fkl.cos(), s_for_fkl.sin()), dim=-1)
        mix_d_logprob = mix_d.log_prob(s_for_fkl).mean()
    ##
    if USE_VMISES:
        for j, mu in enumerate(x):
            mu = th.tensor([mu])
            d = VonMises(loc=mu, concentration=th.tensor([kappa]))
            #VonMisesFisher(loc=th.cat((mu.cos(), mu.sin())), scale=th.tensor([kappa]))
            #### Compute Forward KL divergence
            if type in ["both", "fkl"]:
                FKL = (mix_d_logprob - d.log_prob(s_for_fkl).mean()).item()
                FKL_list.append(FKL)
            #### Compute Reverse KL divergence
            if type in ["both", "rkl"]:
                s = d.sample([bs])
                rmix_d_logprob = mix_d.log_prob(s).mean()
                RKL = (d.log_prob(s).mean() - rmix_d_logprob).item()
                RKL_list.append(RKL)
            print("Element {}/{} done.\r".format(j+1, x.shape[0]), end="")
        ##
        fkl_minimum_values = []
        if type in ["both", "fkl"]:
            FKL_list = np.array(FKL_list)
            _, fkl_minimum_values = get_argmin_solutions(FKL_list)
        #####
        rkl_minimum_values = []
        if type in ["both", "rkl"]:
            RKL_list = np.array(RKL_list)
            _, rkl_minimum_values = get_argmin_solutions(RKL_list)
    else:
        direction = v_for_fkl.mean(0)
        fkl_minimum_values = [direction]#[th.atan2(direction[1], (direction[0] + 1e-6)).item()]
        #fkl_minimum_values = [s_for_fkl.mean().item()]
        rkl_minimum_values = [s_for_fkl[th.argmax(mix_d_logprob)].item()]
    #####
    return fkl_minimum_values, rkl_minimum_values


#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
def plot_von_mises_with_analytical_solutions(ax, fkl_minimum_values, rkl_minimum_values, plot_intent, vals=None):
    if vals is None:
        vals = th.FloatTensor(x).unsqueeze(-1)
    ####
    colors = ['red', '#4fada5']
    minimum_values = [fkl_minimum_values, rkl_minimum_values]
    for minimum_value, color in zip(minimum_values, colors):
        if minimum_value is not None:
            kl_sol_mu, kl_sol_kappa = minimum_value
            ####
            d = VonMises(loc=th.tensor([kl_sol_mu]), concentration=th.tensor([kl_sol_kappa]))
            ####
            y = d.log_prob(vals).squeeze().exp().numpy()
            y = 1.5 * (y - np.min(y)) / (np.max(y) - np.min(y))
            line = ax.plot(x, y, linewidth=2, color=color, zorder=3 )
            ####
            if not plot_intent:
                ax.fill_between(x, y1=np.zeros_like(x), y2=y, color=color, alpha=0.5)
            ####
            arr2 = ax.arrow(x=kl_sol_mu, y=-1.5, dx=0, dy=0.8, alpha=1., width = 0.01, head_width=4*0.1, head_length=2*4*0.1,
                             edgecolor = color, facecolor = color, lw = 1, ls="--", zorder = 3)
        else:
            pass
    ######

def plot_von_mises_mixture_with_analytical_solutions(
        ax, F, I, plot_intent, fkl_minimum_values=None, rkl_minimum_values=None,
        mixture_means=[-np.pi / 2., np.pi / 2.],
        mixture_kappas=[30]*2,
        verbose=False
    ):
    if isinstance(F, np.ndarray):
        F = th.FloatTensor(F)
    ###
    if F.sum() > 0.:
        comp = Independent(VonMises(th.tensor(mixture_means).unsqueeze(-1), th.tensor(mixture_kappas).unsqueeze(-1)), 1)
        mix = Categorical(F / F.sum())
        mix_d = MixtureSameFamily(mix, comp)
        ###
        vals = th.FloatTensor(x).unsqueeze(-1)
        ###
        radii = mix_d.log_prob(vals).squeeze().exp().numpy()
        radii = 1.5 * (radii - np.min(radii)) / (np.max(radii) - np.min(radii))
        ###
        y = mix_d.log_prob(vals).squeeze().exp().numpy()
        y = 1.5 * (y - np.min(y)) / (np.max(y) - np.min(y))
        ### Angles increase anticlockwise from East
        ax.set_theta_zero_location('E'); ax.set_theta_direction(1);
        ###
        if not plot_intent:
            ax.fill_between(x, y1=np.zeros_like(x), y2=y, color=mixture_color, alpha=0.3)
        #else:
        #    ### Plot external black circle
        #    ax.plot(x, np.zeros_like(x)+0.98, linewidth=2, color='black')
        #########
        ### Color the zones of decidability (in #ed702d) and indecision (in gray or white)
        for i, (mu, k) in enumerate(zip(mixture_means, mixture_kappas)):
            tol = 0.5*3.09*np.sqrt(1./k)
            mu_lb, mu_ub = mu - tol - 0.15, mu + tol + 0.15
            #print("(mu_lb, mu_ub) = ", mu_lb, mu_ub)
            mu_x = np.linspace(min(mu_lb, mu_ub), max(mu_lb, mu_ub), num=50, endpoint=False)
            ax.fill_between(mu_x, y1=np.zeros_like(mu_x), y2=-1.5+np.zeros_like(mu_x), color=mixture_color, alpha=0.3)
            ####
            if plot_intent:
                idx = (i+1) % len(mixture_means)
                tol = 0.5*3.09*np.sqrt(1./mixture_kappas[idx])
                mu_b1 = np.mod(max(mu_lb, mu_ub) + 2.*np.pi, 2.*np.pi)
                mu_lb, mu_ub = mixture_means[idx] - tol - 0.15, mixture_means[idx] + tol + 0.15
                mu_b2 = np.mod(min(mu_lb, mu_ub) + 2.*np.pi, 2.*np.pi)
                #print("(mu_b1, mu_b2) = ", mu_b1, mu_b2)
                mu_x = np.linspace(min(mu_b1, mu_b2), max(mu_b1, mu_b2), num=50, endpoint=False)
                ax.fill_between(mu_x, y1=1.+np.zeros_like(mu_x), y2=np.zeros_like(mu_x), color="white", alpha=1.)
                ax.plot(mu_x, np.zeros_like(mu_x)+0.98, linewidth=2, color='black')
                ax.axvline(x=mu_b1, ymin=0.5, ymax=0.82, linewidth=2, color='black')
                ax.axvline(x=mu_b2, ymin=0.5, ymax=0.82, linewidth=2, color='black')
        #########
        ### Plot black circle
        line = ax.plot(x, np.zeros_like(x), linewidth=2, color='black', zorder=2 )
        ### plot mixture distribution
        line = ax.plot(x, y, linewidth=2, color=mixture_color, zorder=3, alpha=mixture_transparency )
        ### Plot KL solutions
        if mixture_transparency < 1.:
            plot_von_mises_with_analytical_solutions(ax, fkl_minimum_values, rkl_minimum_values, plot_intent, vals)
            #####
            if I is None:
                Intents_str = ["SLOT 1", "SLOT 2", "SLOT 3"]
                mask = []
                minimum_value = fkl_minimum_values[0] if TYPE == "fkl" else rkl_minimum_values[0]
                #print(minimum_value, F, TYPE, )
                for idx, m in enumerate(mixture_means):
                    d = VonMises(th.tensor([[m]]), th.tensor([[kappa]]))
                    proba = d.log_prob(th.tensor([minimum_value])).squeeze().exp().item()
                    if proba > 5e-2:
                        I = Intents_str[idx]
                        break
                ####
                if I is None:
                    I = "N/A"
            #####
        ######
    else:
        ### Plot black circle
        line = ax.plot(x, np.zeros_like(x), linewidth=2, color='black', zorder=2 )
    #####
    if I is None:
        I = ""
    #####
    ax.text(1., 1.2, "{}".format(I), color=("red" if TYPE == "fkl" else "#4fada5"))
    ### 'Trick': This will display Zero as a circle. Fitted Von-Mises function will lie along zero.
    ax.set_ylim(-1.5, 1.5)
    ###
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_axis_off()

def get_intent_inference(n, N, state, game, plot_intent, Q):
    mixture_means = [0., 2. * np.pi / 3., 4. * np.pi / 3.]
    mixture_kappas = [kappa]*len(mixture_means)
    ####
    theta_for_directions = np.array(mixture_means)
    directions = np.array([[np.cos(theta), np.sin(theta)] for theta in theta_for_directions])
    Intents = {
        tuple(directions[0]): 0, tuple(directions[1]): 1, tuple(directions[2]): 2 # "R", "G", "B"
    }
    ####
    #if plot_intent:
    Intents_str = ["Red Token", "Green Token", "Blue Token", "N/A"]
    ####
    P = np.zeros((3, 2)) # joint distribution; rows are for intent and columns for actions
    actual_state = game.reset(state) # Actual state with unavailable slot and transitions
    q = np.array([Q.get((state, action), 0.0) for action in game.actions])
    v = logsumexp(q)
    p = np.exp(q - v)
    ##
    for act_id, action in enumerate(game.actions):
        _ = game.reset(state)
        next_states, _ = game.execute(action)
        #print(action, next_states)
        for transition_id, ns in enumerate(next_states):
            P[transition_id, act_id] += actual_state[ns, act_id] * p[act_id] * np.exp(game.token_rewards[ns])
    ####
    print("++ State {}/{}: {}; q = {}; p = {}".format(n+1, N, state, q, p))
    # Plot von Mises distribution
    intent = None
    fkl_minimum_values = None
    rkl_minimum_values = None
    if mixture_transparency < 1.:
        if TYPE == "fkl":
            theta_for_direction, kappa_for_direction = get_analytical_argmin_forward_KL(
                mixture_means, mixture_kappas,
                F=P.sum(-1), N=100
            )
            fkl_minimum_values = (theta_for_direction, kappa_for_direction)
        elif TYPE == "rkl":
            theta_for_direction, kappa_for_direction = get_analytical_argmin_reverse_KL(
                mixture_means, mixture_kappas,
                F=P.sum(-1)
            )
            rkl_minimum_values = (theta_for_direction, kappa_for_direction)
        ####
        #print(theta_for_direction, kappa_for_direction)
        mask = []
        for idx, (m, k) in enumerate(zip(mixture_means, mixture_kappas)):
            d = VonMises(th.tensor([[m]]), th.tensor([[k]]))
            proba = d.log_prob(th.tensor([[theta_for_direction]])).squeeze().exp().item()
            #print(proba)
            mask.append(proba > 5e-2)
        #####
        directions = np.array(list(Intents.keys()))
        intent = directions[
            mask
        ].reshape(-1)
        #####
        intent_id = Intents.get(tuple(intent), None)
        if intent_id is None:
            p = np.zeros_like(P[0])
            intent = "N/A" if plot_intent else ""
        else:
            p = P[intent_id]
            intent = Intents_str[intent_id]
    ####
    return P, intent, p, fkl_minimum_values, rkl_minimum_values, mixture_means, mixture_kappas

def get_action_inference(p):
    mixture_means = [-np.pi / 2., np.pi / 2.]
    mixture_kappas = [kappa]*len(mixture_means)
    ####
    Intents_str = ["SLOT 1", "SLOT 2", "SLOT 3"]
    ####
    I = None
    fkl_minimum_values = None
    rkl_minimum_values = None
    if p.sum() > 0. and mixture_transparency < 1.:
        if TYPE == "fkl":
            theta_for_direction, kappa_for_direction = get_analytical_argmin_forward_KL(
                mixture_means, mixture_kappas,
                F=p
            )
            fkl_minimum_values = (theta_for_direction, kappa_for_direction)
        elif TYPE == "rkl":
            theta_for_direction, kappa_for_direction = get_analytical_argmin_reverse_KL(
                mixture_means, mixture_kappas,
                F=p
            )
            rkl_minimum_values = (theta_for_direction, kappa_for_direction)
        ####
        mask = []
        minimum_value = fkl_minimum_values[0] if TYPE == "fkl" else rkl_minimum_values[0]
        #print(minimum_value, F, TYPE, )
        for idx, m in enumerate(mixture_means):
            d = VonMises(th.tensor([[m]]), th.tensor([[kappa]]))
            proba = d.log_prob(th.tensor([minimum_value])).squeeze().exp().item()
            if proba > 5e-2:
                I = Intents_str[idx]
                break
        ####
        if I is None:
            I = "N/A"
    ####
    return I, fkl_minimum_values, rkl_minimum_values, mixture_means, mixture_kappas

def run_with_analytical_solutions(plot_intent=False):
    ######
    if LOW_UNCERTAINTY:
        suffix, transition_probabilities, valuable_tokens = LOW_UNCERTAINTY_CASES[CASE]
    else:
        suffix, transition_probabilities, valuable_tokens = HIGH_UNCERTAINTY_CASES[CASE]
    ######
    game = RGB_TOKEN(transition_probabilities=transition_probabilities, valuable_tokens=valuable_tokens)
    ######
    suffix += "_".join(["{}".format(e).replace(".", "p") for e in game.M[0]])
    ######
    ## Plot the slot machines with the color proportions corresponding to the transition probabilities
    if ReEvaluate:
        """
        fig, ax = plt.subplots(figsize=(9.2, 5))
        ax.yaxis.set_visible(False)
        ax.set_ylim(0., 1.)
        ####
        category_colors = reversed(["red", "green", "blue"])
        names = ["Slot 1", "Slot 2", "Slot 3"]
        ####
        data = np.flip(game.M, axis=0)
        print(data)
        data_cum = data.cumsum(axis=0)
        print(data_cum)
        for i, color in enumerate(category_colors):
            heights = data[i, :]
            print(heights)
            starting_point = data_cum[i, :] - heights
            print(starting_point)
            print("#########")
            rects = ax.bar(names, heights, bottom=starting_point, width=0.5,
                        color=color)#label=colname,
            ax.bar_label(rects, label_type='center', color='white', fontsize='x-small')
        """
        fig, ax = plt.subplots(figsize=(9.2, 5))
        ax.yaxis.set_visible(False)
        ax.set_ylim(0., 1.)
        ####
        category_colors = reversed(["#ea3323", "#8afc63", "#4866ac"])
        names = ["Slot 1", "Slot 2", "Slot 3"]
        ####
        data = np.flip(game.M, axis=0)
        print(data)
        data_cum = data.cumsum(axis=0)
        print(data_cum)
        for i, color in enumerate(category_colors):
            heights = data[i, :]
            starting_point = data_cum[i, :] - heights
            rects = ax.bar(names, heights, bottom=starting_point, width=0.4,
                        color=color, edgecolor="black")
            print("#########")
            ax.bar_label(rects, label_type='center', color='black', fontsize='x-large')
        ####
        #plt.show()
        plt.savefig("./sim_res/rgb_token_world/" + "machine_" + suffix + ".pdf")
    ######
    filename = './rgbToken_qtable.npy'
    if os.path.isfile(filename) and not ReEvaluate:
        Q = np.load(filename, allow_pickle='TRUE').item()
    else:
        game.iter_policy_eval(n_iterations=200, eps=1e-5)
        Q = np.load(filename, allow_pickle='TRUE').item()
    ######
    h, w = 1, 3
    #fig, axs = plt.subplots(h, w, polar=True)
    WIDTH_SIZE, HEIGHT_SIZE = 12, 9
    fig = plt.figure(figsize=(WIDTH_SIZE, HEIGHT_SIZE))#, facecolor='black')
    gs = fig.add_gridspec(h, w, hspace=0.05, wspace=0.05)
    axs = np.array([[None]*w]*h)
    k = 1
    for i in range(h):
        for j in range(w):
            axs[i, j] = fig.add_subplot(gs[i, j], polar=True)
            ###
            if plot_intent:
                ###
                norm = mp.colors.Normalize(0., 2.*np.pi)
                #quant_steps = 2056
                cb = mp.colorbar.ColorbarBase(
                    axs[i, j],
                    cmap=mp.colormaps.get_cmap('hsv'),# quant_steps),#
                    norm=norm,#
                    orientation='horizontal',
                    alpha=0.5
                )
                ## For aesthetics - Get rid of border and axis labels
                #cb.outline.set_visible(False)
                axs[i, j].set_axis_off()
            ###
            _ = axs[i, j].set_xticklabels([])
            _ = axs[i, j].set_yticklabels([])
            k += 1
    ######
    start_list = game.states.copy()
    N = len(start_list)
    assert N == 3, "Expected 3 states, got {} states".format(N)
    ##########################################
    ####
    for n, state in enumerate(start_list):
        P, intent, p, fkl_minimum_values, rkl_minimum_values, mixture_means, mixture_kappas = get_intent_inference(n, N, state, game, plot_intent, Q)
        ###########
        if plot_intent:
            plot_von_mises_mixture_with_analytical_solutions(
                axs[0, n], P.sum(-1), intent, plot_intent,
                fkl_minimum_values, rkl_minimum_values,
                mixture_means, mixture_kappas
            )
        else:
            ######
            I, fkl_minimum_values, rkl_minimum_values, mixture_means, mixture_kappas = get_action_inference(p)
            print(p, fkl_minimum_values)
            ####
            plot_von_mises_mixture_with_analytical_solutions(
                axs[0, n], p, None, plot_intent,
                fkl_minimum_values, rkl_minimum_values,
                mixture_means, mixture_kappas
            )
        if state == 0:
            print(P, P.sum())
            print(P.sum(-1))
            print(P.sum(0))
        else:
            print(P.sum(-1))
            print(P.sum(0))
    ###########
    if plot_intent:
        filename = "Intents_{}".format(TYPE)
    else:
        filename = "Affordances_{}".format(TYPE)
    ###########
    extent = Bbox.union(flatten([[full_extent(axs[i, j]) for j in range(w)] for i in range(h)]))
    # It's best to transform this back into figure coordinates. Otherwise, it won't
    # behave correctly when the size of the plot is changed.
    extent = extent.transformed(fig.transFigure.inverted())
    # We can now make the rectangle in figure coords using the "transform" kwarg.
    rect = Rectangle([extent.xmin, extent.ymin], extent.width, extent.height,
                     facecolor='none', edgecolor='black', zorder=0,
                     transform=fig.transFigure)
    fig.patches.append(rect)
    ###########
    #plt.show()
    plt.savefig("./sim_res/rgb_token_world/" + filename + "_" + suffix + ".pdf")

if __name__ == "__main__":
    USE_VMISES = False
    args = parser.parse_args()
    LOW_UNCERTAINTY = not args.hu
    CASE = args.case
    ####
    print("####################")
    kappa = 30.
    x = np.linspace(-np.pi, np.pi, num=500)
    theta = np.linspace(-np.pi, np.pi, num=50, endpoint=False)
    ####
    ReEvaluate = True
    TYPE = "both"
    mixture_transparency = 1.
    mixture_color = "black"
    run_with_analytical_solutions()
    ####
    print("####################")
    ReEvaluate = False
    TYPE = "fkl"
    mixture_transparency = 0.3
    mixture_color = "black"
    run_with_analytical_solutions(plot_intent=True)
    mixture_color = "#ed702d"
    run_with_analytical_solutions(plot_intent=False)
    ####
    print("####################")
    TYPE = "rkl"
    mixture_transparency = 0.3
    mixture_color = "black"
    run_with_analytical_solutions(plot_intent=True)
    mixture_color = "#ed702d"
    run_with_analytical_solutions(plot_intent=False)
