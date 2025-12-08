import os
import sys
import errno
import numpy as np
import pandas as pd
import argparse
from rgb_token import RGB_TOKEN
from scipy.special import logsumexp

import torch as th
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.categorical import Categorical
from torch.distributions.independent import Independent
from torch.distributions.multivariate_normal import MultivariateNormal

import matplotlib as mpl
mpl.rcParams["text.usetex"] = True
mpl.rcParams.update({"font.size": 20})
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import itertools
from scipy.optimize import minimize
import torch.multiprocessing as mp
import seaborn as sns
from tqdm import tqdm

################################################################################
N_PROCESS = mp.cpu_count() - 5#// 2
USE_CUDA = False

IS_RUN = False
PROBA_ROUND_PRECISION = 3

N_TRIALS = 10**PROBA_ROUND_PRECISION
PROBA_ROUND_MULT = 10**PROBA_ROUND_PRECISION

USE_INHIB = True
RAND_START = False
START_BIAS = 0.01

R = 3 ## Separation of the vertices
### transition_probabilities = [[Row1Col1, Row1Col2, Row1Col3], [Row2Col1, Row2Col2, Row2Col3], [Row3Col1, Row3Col2, Row3Col3]]
### Sum over rows must be equal to 1: Row1Col1 + Row2Col1 + Row3Col1 = 1 (same for Col2 and Col3)
### Row1 = Red; Row2 = Green; Row3 = Blue
### Col1 = Slot1; Col2 = Slot2; Col3 = Slot3
CASES = [
    ("LU_case1_", np.array([[0.2, 0.75, 0.05], [0.75, 0.05, 0.2], [0.05, 0.2, 0.75]]), [1], 2),
    ("LU_case2_", np.array([[0.2, 0.75, 0.05], [0.75, 0.05, 0.75], [0.05, 0.2, 0.2]]), [1], 1),
    ##("HU_case3_", np.array([[0.2, 0.3, 0.5], [0.3, 0.5, 0.2], [0.5, 0.2, 0.3]]), [0, 2], 1),
    ("HU_case3_", np.array([[0.3, 0.3, 0.5], [0.2, 0.5, 0.2], [0.5, 0.2, 0.3]]), [0, 2], 1),
    ("LU_case4_", np.array([[0.15, 0.1, 0.75], [0.1, 0.8, 0.1], [0.75, 0.1, 0.15]]), [0, 2], 1),
]
KL_TYPES = [
    "rkl",
    "fkl",
    "jkl"
]
LAMBDA_VALUES = [
    0., 0.5, 1.
]

################################################################################
def fetch_args():
    parser = argparse.ArgumentParser(description="Parser")
    parser.add_argument("--R", default=3, type=int, help="Separation of the vertices (default: 3)")
    return parser.parse_args()

################################################################################
def make_dirs(d):
    for i, p in enumerate(d.split("/")):
        p = "/".join(d.split("/")[:i]) + "/" + p
        if not os.path.isdir(p):
            try:
                os.mkdir(p)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
                pass

################################################################################
def angle_between_vectors(v1, v2):
    """
    Calculates the angle in radians between two PyTorch vectors.
    Args:
        v1 (torch.Tensor): The first vector.
        v2 (torch.Tensor): The second vector.
    Returns:
        torch.Tensor: The angle in radians.
    """
    # Calculate the dot product
    dot_product = th.dot(v1, v2)
    # Calculate the magnitudes of the vectors
    magnitude_v1 = th.norm(v1)
    magnitude_v2 = th.norm(v2)
    # Calculate the cosine of the angle
    cosine_angle = dot_product / (magnitude_v1 * magnitude_v2)
    # Handle potential floating point errors that might push cosine_angle outside [-1, 1]
    cosine_angle = th.clamp(cosine_angle, -1.0, 1.0)
    # Calculate the angle using arc cosine
    angle_radians = th.acos(cosine_angle)
    ##
    return angle_radians

################################################################################
def symsqrt(matrix):
    """Compute the square root of a positive definite matrix."""
    # perform the decomposition
    # s, v = matrix.symeig(eigenvectors=True)
    _, s, v = matrix.svd()  # passes th.autograd.gradcheck()
    # truncate small components
    above_cutoff = s > s.max() * s.size(-1) * th.finfo(s.dtype).eps
    s = s[..., above_cutoff]
    v = v[..., above_cutoff]
    # compose the square root matrix
    return (v * s.sqrt().unsqueeze(-2)) @ v.transpose(-2, -1)

################################################################################
def get_mahal_dist(x, mixture_means, mixture_sigmas, n_choices):
    mahal_dist = []
    for j in range(n_choices):
        #diff = (mixture_means[j] - x) @ th.linalg.pinv(mixture_sigmas[j])
        diff = (mixture_means[j] - x)
        mahal_dsq = th.einsum('i,ik,k->', diff, th.linalg.pinv(mixture_sigmas[j]), diff)
        mahal_dist.append(
            #diff.square().sum().item()
            mahal_dsq.item()
        )
    return np.array(mahal_dist)

################################################################################
def is_inside_simplex(x, vertices, tol=1e-8):
    """
    Test if the point x is inside the simplex defined by 'vertices'.
    Parameters:
      x        : array_like, shape (n,)
                 The point to test.
      vertices : array_like, shape (n+1, n)
                 The vertices of the simplex. Each row is a vertex.
      tol      : float, optional
                 Tolerance for numerical non-negativity.
    Returns:
      inside   : bool
                 True if x is inside the simplex (within tolerance), False otherwise.
      bary     : array, shape (n+1,)
                 The barycentric coordinates of x.
    """
    vertices = np.asarray(vertices)
    x = np.asarray(x)
    n = vertices.shape[1]  # each vertex is in R^n
    # Use the first vertex as a reference
    v0 = vertices[0]
    # Form the matrix of differences: columns are (v_i - v0) for i=1,...,n
    T = (vertices[1:] - v0).T  # shape (n, n)
    # Compute the coefficients for the difference
    y = x - v0
    try:
        # Solve T * lambda = y for lambda = [lambda_1, ..., lambda_n]
        lambdas = np.linalg.solve(T, y)
    except np.linalg.LinAlgError:
        raise ValueError("The provided vertices do not form a non-degenerate simplex.")
    #
    proj_x = (T @ np.clip(lambdas, a_min=-tol, a_max=100)) + v0
    # Compute the barycentric coordinate for v0
    lambda0 = 1 - np.sum(lambdas)
    bary = np.concatenate(([lambda0], lambdas))
    # Check if all barycentric coordinates are >= -tol
    inside = np.all(bary >= -tol)
    return inside, bary, proj_x

################################################################################
def project_onto_simplex(vertices, x, tol=1e-9):
    """
    Project point x onto the convex hull of the simplex defined by 'vertices'.
    #####
    Parameters
    ----------
    vertices : array_like, shape (n+1, n)
        The rows are the n+1 vertices v_i in R^n.
    x : array_like, shape (n,)
        The query point (possibly outside the simplex).
    tol : float
        Tolerance for the solver.
    #####
    Returns
    -------
    y : ndarray, shape (n,)
        The projection of x onto the simplex.
    w : ndarray, shape (n+1,)
        Barycentric weights of y = sum_i w[i] * vertices[i].
    """
    V = np.asarray(vertices)  # shape (n+1, n)
    m, n = V.shape
    assert m == n+1, "Need exactly n+1 vertices in R^n"
    #####
    # Precompute for objective
    G = V.dot(V.T)       # shape (m, m)
    c = V.dot(x)         # shape (m,)
    #####
    def objective(w):
        # ½ w^T (2G) w - 2 c^T w + const
        diff = G.dot(w) - c
        return float(w.dot(diff) - np.dot(x, x))  # returns ||Vw - x||^2
    #####
    def jac(w):
        # ∇_w [||Vw - x||^2] = 2 V (V^T w - x) = 2 (G w - c)
        return 2*(G.dot(w) - c)
    #####
    # constraints: sum(w)=1, w_i >= 0
    cons = [
        {'type': 'eq',
         'fun':    lambda w: np.sum(w) - 1.0,
         'jac':    lambda w: np.ones(m)},
        {'type': 'ineq',
         'fun':    lambda w: w,
         'jac':    lambda w: np.eye(m)}
    ]
    #####
    # initial guess: uniform weights
    w0 = np.ones(m) / m
    #####
    sol = minimize(objective, w0,
                   method='SLSQP',
                   jac=jac,
                   constraints=cons,
                   tol=tol,
                   options={'ftol': tol})
    #####
    if not sol.success:
        raise RuntimeError(f"Projection failed: {sol.message}")
    #####
    w_opt = sol.x
    y = w_opt.dot(V)     # projected point
    #####
    return y, w_opt

################################################################################
def get_vertices(n, r=1., w=None):
    a = r * np.sqrt(n / (n-1))
    I = np.eye(n)
    ones_ = np.ones(n)
    ######
    if w is None or (w == ones_).all():
        Q = np.eye(n)
    else:
        Q = np.eye(n)
        """ones_norm = np.linalg.norm(ones_)
        x = ones_# / ones_norm
        y = w# / np.linalg.norm(w)
        diff = (x - y)
        diff = (diff / np.linalg.norm(diff)).reshape(n, 1)
        Q = np.eye(n) - 2. * np.dot(diff, diff.T)# / np.linalg.norm(diff)
        Q = (1. / np.linalg.det(Q)) * Q
        print(Q, x, y)
        print(Q @ ones_)
        print(Q @ w)
        print(np.linalg.det(Q))"""
    ######
    k = 1 / n
    ######
    v = []
    for i, e in enumerate(I):
        v.append(
            a * (Q @ (e - k * ones_))
        )
        #print(np.dot(v[-1], ones_))
    ######
    return np.stack(v)

################################################################################
def project_vertices_onto_hyperplane(V):
    """
    Given an array V of shape (n+1, n+1), where each row V[i] is the
    coordinate of the i-th vertex of an n-simplex in R^{n+1},
    return an array of shape (n+1, n) giving their coordinates
    in the n-dimensional hyperplane sum(x_i) = 0.

    That is, we find a basis B for the hyperplane and solve
        B @ alpha_i = v_i
    for each vertex v_i. The result alpha_i are the 'intrinsic'
    coordinates in that hyperplane.
    """
    # Number of vertices is n+1, each in R^{n+1}:
    n_plus_1 = V.shape[1]   # dimension is n+1
    n = n_plus_1 - 1        # the hyperplane is n-dimensional
    # --- 1) Build a spanning set for the hyperplane ---
    #
    # A standard choice: b_i = e_i - e_{n+1}  for i=1..n,
    # each clearly satisfies sum(b_i) = 0.
    # We'll put these as columns of a matrix B.
    #
    # B has shape ((n+1) x n), i.e. each column is b_i in R^{n+1}.
    B = np.zeros((n_plus_1, n))
    for i in range(n):
        B[i, i]   = 1.0
        B[-1, i]  = -1.0
    #####
    # --- 2) Solve for alpha_i in B alpha_i = v_i ---
    # Because v_i are rows of V, we can do alpha_i^T = v_i^T B_pinv.
    # This means alpha = V @ B_pinv for all vertices at once.
    """
    B_pinv = np.linalg.pinv(B)    # pseudo-inverse of B of shape (n, n+1)
    coords = (B_pinv @ V).T          # shape = (n+1, n)
    """
    # If instead we want an orthonormal basis for the hyperplane
    # rather than the ad-hoc b_i = e_i - e_{n+1} for i=1..n,
    # we can do a QR factorization on B and then apply it
    Q, R = np.linalg.qr(B)    # Q has shape (n+1, n)
    coords = V @ Q
    #####
    return coords, Q

################################################################################
def get_joint_probabilities(game, Q, unavail_slot_idx):
    start_list = game.states.copy()
    N = len(start_list)
    if N != 3:
        print("Expected 3 states, got {} states".format(N))
        return ""
    ######
    state = start_list[unavail_slot_idx]
    P = np.zeros((3, 2)) # joint distribution; rows are for intent and columns for actions
    actual_state = game.reset(state) # Actual state with unavailable slot and transitions
    q = np.array([Q.get((state, action), 0.0) for action in game.actions])
    p = np.exp(q - logsumexp(q))
    ####
    reward_normalizer = np.sum([np.exp(r) for r in game.token_rewards])
    ####
    for act_id, action in enumerate(game.actions):
        _ = game.reset(state)
        next_states, _ = game.execute(action)
        for transition_id, ns in enumerate(next_states):
            P[transition_id, act_id] += actual_state[ns, act_id] * p[act_id]# * np.exp(game.token_rewards[ns]) / reward_normalizer
    ####
    return P

################################################################################
def proba_flow_sde(p, sdir, kl_type, lambd=0., seed=1234):
    D = 1.#0.1
    dt = 1e-3
    ########
    T = 100.  # Total time.
    N = int(T / dt) # Number of time steps.
    #t = np.linspace(0., T, N)  # Vector of times.
    ########
    ref_mahal_dist = 1.
    sqrtdt = np.sqrt(2.*dt)
    sqrtD = np.sqrt(D)
    ########
    p = p / p.sum() ## Make sure it is normalized
    n_choices = p.shape[0]
    assert n_choices > 1, "n_choices = {}; p = {}; intent_idx = {}".format(n_choices,
        p, intent_idx
    )
    mixture_means = th.FloatTensor(
        project_vertices_onto_hyperplane(
            get_vertices(n_choices, r=R)
        )[0]
    )
    d = n_choices - 1
    mixture_sigmas = th.FloatTensor(np.eye(d).reshape(1, d, d).repeat(n_choices, axis=0))
    ####
    comp = MultivariateNormal(mixture_means, mixture_sigmas)
    mix = Categorical(th.FloatTensor(p))
    mixture_dist = MixtureSameFamily(mix, comp)
    ###########
    if kl_type == "rkl":
        idx = np.argmax(p)
        mu, sigma = mixture_means[idx], mixture_sigmas[idx]
    elif kl_type in ["fkl", "jkl"]:
        mu, sigma = th.zeros_like(mixture_means[0]), th.zeros_like(mixture_sigmas[0])
        for idx, l in enumerate(p):
            mu.add_(l * mixture_means[idx])
        for idx, l in enumerate(p):
            diff = (mu - mixture_means[idx]).reshape(d, 1)
            sigma.add_(
                l * (mixture_sigmas[idx] @ mixture_sigmas[idx].T
                + diff @ diff.T)
            )
        ####
        sigma = symsqrt(sigma)
    else:
        ## jkl
        raise NotImplementedError
    ########
    q = MultivariateNormal(mu, sigma)
    #############################
    #############################
    np.random.seed(seed)
    #####
    xs = []
    mahal_dist_list = []
    drift_direction_list = []
    x = th.zeros_like(mixture_means[0])
    for i in range(N):
        xs.append(x.numpy())
        mahal_dist = get_mahal_dist(x, mixture_means, mixture_sigmas, n_choices)
        mahal_dist_list.append(np.sqrt(mahal_dist))
        #####
        mask = (mahal_dist < ref_mahal_dist)
        if mask.any():# or not is_inside:
            xs.append(x.numpy())
            mahal_dist = get_mahal_dist(x, mixture_means, mixture_sigmas, n_choices)
            mahal_dist_list.append(np.sqrt(mahal_dist))
            #####
            break
        #####
        y = x.clone()
        y.requires_grad_()
        logq = q.log_prob(y)
        logq.backward()
        derX_logq = y.grad
        #####
        y = x.clone()
        y.requires_grad_()
        logp = mixture_dist.log_prob(y)
        logp.backward()
        derX_logp = y.grad
        #####
        ratio = 1.
        if kl_type == "fkl":
            ratio = (logp - logq).detach().exp()
        elif kl_type == "jkl":
            ratio = (logp - logq).detach().exp()
            #ratio = (1. - lambd) + lambd * ratio
            ratio = 1. + lambd * (ratio - 1.)
        #####
        vt = ratio * (derX_logp - derX_logq)
        drift = vt + D * derX_logq
        #####
        drift_direction = np.array([
            th.dot(drift, mixture_means[j]) / th.linalg.norm(mixture_means[j])
            for j in range(n_choices)
        ])
        drift_direction_list.append(drift_direction)
        #####
        x = x + dt * drift + sqrtD * sqrtdt * th.FloatTensor(np.random.normal(0.0, 1.0, size=d))
        assert th.isnan(x).sum() == 0, "vt = {}; Dxlogp = {}; Dxlogq = {}; ratio = {}; \
            logp = {}, logq = {}; y = {}; is_inside = {}".format(vt, derX_logp, derX_logq, ratio,
                logp, logq, y, is_inside_simplex(y.detach().numpy(), mixture_means.numpy())[0]
            )
        assert th.isinf(x).sum() == 0, "vt = {}; Dxlogp = {}; Dxlogq = {}; ratio = {}; \
            logp = {}, logq = {}; y = {}; is_inside = {}".format(vt, derX_logp, derX_logq, ratio,
                logp, logq, y, is_inside_simplex(y.detach().numpy(), mixture_means.numpy())[0]
            )
        #####
        is_inside, _, proj_x = is_inside_simplex(x.numpy(), mixture_means.numpy())
        if not is_inside:
            ## then you should make sure you stay on the boundary
            try:
                z, w = project_onto_simplex(mixture_means.numpy(), x.numpy(), tol=1e-8)
            except RuntimeError as e:
                z = proj_x
            x = th.FloatTensor(z)
        #####
    ########
    xs = np.array(xs)
    mahal_dist_list = np.array(mahal_dist_list)
    ########
    #plt.clf()
    fig, ax = plt.subplots(figsize=(12,9))
    ########
    ts = np.array([i for i in range(mahal_dist_list.shape[0])])*dt # Vector of times.
    ########
    if n_choices == 2:
        colors = ["#ed702d", "#00b0a5"]
        items_type = "actions"
        legend_texts = ["Slot 1", "Slot 2"]
    else:
        colors = ["#ea3323", "#8afc63", "#4866ac"]#['r', 'g', 'b']
        items_type = "intents"
        legend_texts = ["Red token", "Green token", "Blue token"]
    ######## Plot Mahalonobis distance
    ax.axhline(y=ref_mahal_dist, xmin=0., xmax=T, linestyle='dashed', color='black', label="Decision Threshold")
    for n in range(n_choices):
        ax.plot(ts, mahal_dist_list[:, n], alpha=1., c=colors[n], label=legend_texts[n])
        val = mahal_dist_list[-1, n]
        #if val <= ref_mahal_dist:
        if mask[n]:
            ax.scatter([ts[-1]], [val], s=30, marker='o', c=colors[n])
    ######## Plot the diffusion
    ## No plot
    ########
    chosen_idx = np.array(list(range(n_choices)))[mask]
    if chosen_idx.shape[0] == 0:
        #print(f"No choice for {kl_type} with p={p}.")
        print(f"No choice for {sdir}.")
        chosen_idx = None
    else:
        chosen_idx = chosen_idx[0]
    ########
    plt.legend()
    plt.savefig(sdir + "mahal_dist_diffusion_{}.pdf".format(items_type))
    plt.close()
    ########
    pd.DataFrame(
        {"RT": list(ts), "x": list(xs), "mahal_dist": list(mahal_dist_list)}
    ).to_csv(sdir + "diffusion_data_{}.csv".format(items_type), index=False)
    ########
    return chosen_idx

################################################################################
def proba_flow_sde2(p, sdir, kl_type, lambd=0., seed=1234):
    """
     This one implements a version with opponent-inhibition scheme where
    the mixture p(x) defines the excitatory populations and the complement
    \bar p(x) corresponds to the inhibitory populations.
    With p(x) = \sum^N_{k=1} \pi_k p_k(x) we have:
     \bar p(x) = \frac{1}{N-1} \sum^N_{k=1} [\sum_{j\neq k} \pi_j] p_k(x)
    """
    D = 1.
    dt = 1e-3
    ########
    T = 100.  # Total time.
    N = int(T / dt) # Number of time steps.
    ########
    ref_mahal_dist = 1.
    sqrtdt = np.sqrt(2.*dt)
    sqrtD = np.sqrt(D)
    ########
    p = p / p.sum() ## Make sure it is normalized
    n_choices = p.shape[0]
    assert n_choices > 1, "n_choices = {}; p = {}; intent_idx = {}".format(n_choices,
        p, intent_idx
    )
    mixture_means = th.FloatTensor(
        project_vertices_onto_hyperplane(
            get_vertices(n_choices, r=R)
        )[0]
    )
    d = n_choices - 1
    mixture_sigmas = th.FloatTensor(np.eye(d).reshape(1, d, d).repeat(n_choices, axis=0))
    ####
    comp = MultivariateNormal(mixture_means, mixture_sigmas)
    mix = Categorical(th.FloatTensor(p))
    mixture_dist = MixtureSameFamily(mix, comp)
    #### Complement
    #p_bar = []
    #for i in range(n_choices):
    #    p_bar.append(sum(p[:i]) + sum(p[i+1:]))
    #p_bar = np.array(p_bar)
    p_bar = 1. - p
    p_bar /= d
    ## Extra precaution (should not be necessary by construction):
    p_bar = p_bar / p_bar.sum()
    ####
    comp = MultivariateNormal(mixture_means, mixture_sigmas)
    mix = Categorical(th.FloatTensor(p_bar))
    mixture_dist_inhib = MixtureSameFamily(mix, comp)
    ###########
    if kl_type == "rkl":
        idx = np.argmax(p)
        mu, sigma = mixture_means[idx], mixture_sigmas[idx]
        ## inhibition static solution q
        idx = np.argmax(p_bar)
        mu_inhib, sigma_inhib = mixture_means[idx], mixture_sigmas[idx]
    elif kl_type in ["fkl", "jkl"]:
        mu, sigma = th.zeros_like(mixture_means[0]), th.zeros_like(mixture_sigmas[0])
        ## inhibition static solution q
        mu_inhib, sigma_inhib = th.zeros_like(mixture_means[0]), th.zeros_like(mixture_sigmas[0])
        for idx, l in enumerate(p):
            mu.add_(l * mixture_means[idx])
            mu_inhib.add_(p_bar[idx] * mixture_means[idx])
        for idx, l in enumerate(p):
            diff = (mu - mixture_means[idx]).reshape(d, 1)
            sigma.add_(
                l * (mixture_sigmas[idx] @ mixture_sigmas[idx].T
                + diff @ diff.T)
            )
            ##
            diff = (mu_inhib - mixture_means[idx]).reshape(d, 1)
            sigma_inhib.add_(
                p_bar[idx] * (mixture_sigmas[idx] @ mixture_sigmas[idx].T
                + diff @ diff.T)
            )
        ####
        sigma = symsqrt(sigma)
        sigma_inhib = symsqrt(sigma_inhib)
    else:
        ## jkl
        raise NotImplementedError
    ########
    q = MultivariateNormal(mu, sigma)
    q_inhib = MultivariateNormal(mu_inhib, sigma_inhib)
    #############################
    #############################
    def get_derX_logprob(x, proba):
        y = x.clone()
        y.requires_grad_()
        log_prob = proba.log_prob(y)
        log_prob.backward()
        return y.grad, log_prob
    #############################
    #############################
    #np.random.seed(seed)
    rng  = np.random.default_rng(seed)
    #####
    k_E = 2.
    rho = 1.#min(np.sqrt((k_E**2 - 1.) / k_E**2) + 0.01, 1.) # correlation
    k_I = k_E * rho - np.sqrt((k_E * rho)**2 - (k_E**2 - 1.))
    if k_I < 0.:
        k_I = k_E * rho + np.sqrt((k_E * rho)**2 - (k_E**2 - 1.))
    #####
    xs = []
    mahal_dist_list = []
    drift_direction_list = []
    exc_drift_direction_list = []
    inhib_drift_direction_list = []
    #####
    x = th.zeros_like(mixture_means[0])
    if RAND_START:
        x = rng.uniform(-START_BIAS, START_BIAS, size=mixture_means[0].shape)
        x = th.FloatTensor(x)
    #####
    for i in range(N):
        xs.append(x.numpy())
        mahal_dist = get_mahal_dist(x, mixture_means, mixture_sigmas, n_choices)
        mahal_dist_list.append(np.sqrt(mahal_dist))
        #####
        mask = (mahal_dist < ref_mahal_dist)
        if mask.any():# or not is_inside:
            xs.append(x.numpy())
            mahal_dist = get_mahal_dist(x, mixture_means, mixture_sigmas, n_choices)
            mahal_dist_list.append(np.sqrt(mahal_dist))
            #####
            break
        #####
        derX_logq, logq = get_derX_logprob(x, q)
        ##
        derX_logp, logp = get_derX_logprob(x, mixture_dist)
        #####
        derX_logq_inhib, logq_inhib = get_derX_logprob(-x, q_inhib)
        ##
        derX_logp_inhib, logp_inhib = get_derX_logprob(-x, mixture_dist_inhib)
        #####
        ratio = 1.
        ratio_inhib = 1.
        if kl_type == "fkl":
            ratio = (logp - logq).detach().exp()
            ratio_inhib = (logp_inhib - logq_inhib).detach().exp()
        elif kl_type == "jkl":
            ratio = (logp - logq).detach().exp()
            #ratio = (1. - lambd) + lambd * ratio
            ratio = 1. + lambd * (ratio - 1.)
            ##
            ratio_inhib = (logp_inhib - logq_inhib).detach().exp()
            ratio_inhib = 1. + lambd * (ratio_inhib - 1.)
        #####
        vt_exc = ratio * (derX_logp - derX_logq)
        drift_exc = vt_exc + D * derX_logq
        ##
        drift_direction = np.array([
            th.dot(drift_exc, mixture_means[j]) / th.linalg.norm(mixture_means[j])
            for j in range(n_choices)
        ])
        exc_drift_direction_list.append(drift_direction)
        #####
        vt_inhib = ratio_inhib * (derX_logp_inhib - derX_logq_inhib)
        drift_inhib = vt_inhib + D * derX_logq_inhib
        ##
        drift_direction = np.array([
            th.dot(drift_inhib, mixture_means[j]) / th.linalg.norm(mixture_means[j])
            for j in range(n_choices)
        ])
        inhib_drift_direction_list.append(drift_direction)
        #####
        drift = k_E * drift_exc - k_I * drift_inhib
        #####
        drift_direction = np.array([
            th.dot(drift, mixture_means[j]) / th.linalg.norm(mixture_means[j])
            for j in range(n_choices)
        ])
        drift_direction_list.append(drift_direction)
        #####
        #noise = np.random.normal(0.0, 1.0, size=d)
        noise = rng.normal(0.0, 1.0, size=d)
        x = x + dt * drift + sqrtD * sqrtdt * th.FloatTensor(noise)
        assert th.isnan(x).sum() == 0, "vt = {}; Dxlogp = {}; Dxlogq = {}; ratio = {}; \
            logp = {}, logq = {}; y = {}; is_inside = {}".format(vt, derX_logp, derX_logq, ratio,
                logp, logq, y, is_inside_simplex(y.detach().numpy(), mixture_means.numpy())[0]
            )
        assert th.isinf(x).sum() == 0, "vt = {}; Dxlogp = {}; Dxlogq = {}; ratio = {}; \
            logp = {}, logq = {}; y = {}; is_inside = {}".format(vt, derX_logp, derX_logq, ratio,
                logp, logq, y, is_inside_simplex(y.detach().numpy(), mixture_means.numpy())[0]
            )
        #####
        is_inside, _, proj_x = is_inside_simplex(x.numpy(), mixture_means.numpy())
        if not is_inside:
            ## then you should make sure you stay on the boundary
            try:
                z, w = project_onto_simplex(mixture_means.numpy(), x.numpy(), tol=1e-8)
            except RuntimeError as e:
                z = proj_x
            x = th.FloatTensor(z)
        #####
    ########
    xs = np.array(xs)
    mahal_dist_list = np.array(mahal_dist_list)
    padd_size = mahal_dist_list.shape[0] - len(drift_direction_list)
    assert padd_size >= 0, "Unexpected size difference: {} vs {}".format(mahal_dist_list.shape[0], len(drift_direction_list))
    padd = [np.zeros(n_choices) for _ in range(padd_size)]
    drift_direction_list = drift_direction_list + padd
    exc_drift_direction_list = exc_drift_direction_list + padd
    inhib_drift_direction_list = inhib_drift_direction_list + padd
    ###
    drift_direction_list = np.array(drift_direction_list)
    exc_drift_direction_list = np.array(exc_drift_direction_list)
    inhib_drift_direction_list = np.array(inhib_drift_direction_list)
    ########
    if n_choices == 2:
        colors = ["#ed702d", "#00b0a5"]
        items_type = "actions"
        legend_texts = ["Slot 1", "Slot 2"]
    else:
        colors = ["#ea3323", "#8afc63", "#4866ac"]#['r', 'g', 'b']
        items_type = "intents"
        legend_texts = ["Red token", "Green token", "Blue token"]
    ######## Plot Mahalonobis distance
    #plt.clf()
    fig, ax = plt.subplots(figsize=(12,9))
    ########
    ts = np.array([i for i in range(mahal_dist_list.shape[0])])*dt # Vector of times.
    ########
    ax.axhline(y=ref_mahal_dist, xmin=0., xmax=T, linestyle='dashed', color='black', label="Decision Threshold")
    for n in range(n_choices):
        ax.plot(ts, mahal_dist_list[:, n], alpha=1., c=colors[n], label=legend_texts[n])
        val = mahal_dist_list[-1, n]
        #if val <= ref_mahal_dist:
        if mask[n]:
            ax.scatter([ts[-1]], [val], s=30, marker='o', c=colors[n])
    ########
    plt.legend()
    plt.savefig(sdir + "mahal_dist_diffusion_{}.pdf".format(items_type))
    plt.close()
    ######## Plot the drift direction
    #plt.clf()
    drift_lists = [
        ("", drift_direction_list),
        ("exc_", exc_drift_direction_list),
        ("inhib_", inhib_drift_direction_list)
    ]
    for drift_type, drift_list in drift_lists:
        fig, ax = plt.subplots(figsize=(12,9))
        ########
        for n in range(n_choices):
            #ax.plot(ts, drift_direction_list[:, n], alpha=1., c=colors[n], label=legend_texts[n])
            ax.plot(ts, drift_list[:, n], alpha=1., c=colors[n], label=legend_texts[n])
        ########
        plt.legend()
        plt.savefig(sdir + "{}drift_direction_{}.pdf".format(drift_type, items_type))
        plt.close()
    ######## Plot the diffusion
    ## No plot
    ########
    chosen_idx = np.array(list(range(n_choices)))[mask]
    if chosen_idx.shape[0] == 0:
        print(f"No choice for {sdir}: p={p}; p_bar={p_bar}.")
        chosen_idx = None
        print(mu, sigma)
        print(mu_inhib, sigma_inhib)
    else:
        chosen_idx = chosen_idx[0]
    ########
    pd.DataFrame(
        {"RT": list(ts), "x": list(xs), "mahal_dist": list(mahal_dist_list),
        "drift_direction": list(drift_direction_list),
        "exc_drift_direction": list(exc_drift_direction_list),
        "inhib_drift_direction": list(inhib_drift_direction_list)}
    ).to_csv(sdir + "diffusion_data_{}.csv".format(items_type), index=False)
    ########
    return chosen_idx

################################################################################
def run_sde(case, transition_probabilities, valuable_tokens, unavail_slot_idx, kl_type, lambd, trial_id, is_run=True):
    #######
    game = RGB_TOKEN(transition_probabilities=transition_probabilities, valuable_tokens=valuable_tokens)
    #######
    if lambd == 0.:
        kl_type = "rkl"
    elif lambd == 1.:
        kl_type = "fkl"
    else:
        kl_type = "jkl"
    #######
    probab_flow_fct = proba_flow_sde
    suffix = case + "_".join(["{}".format(e).replace(".", "p") for e in game.M[0]])
    R_suff = ""
    if USE_INHIB:
        probab_flow_fct = proba_flow_sde2
        R_suff += "_inhib"
    if RAND_START:
        R_suff += "_randstart"
    sdir = "./reaction_times/RGB_TOKEN/simplex_radiusR{}/{}_{}l{}/{}/".format(str(R) + R_suff, suffix,
        kl_type, str(lambd).replace(".", "p"), trial_id
    )
    #######
    if is_run:
        make_dirs(sdir)
        ######
        #plt.clf()
        fig, ax = plt.subplots(figsize=(9.2, 5))
        ax.yaxis.set_visible(False)
        ax.set_ylim(0., 1.)
        ####
        #category_colors = reversed(["red", "green", "blue"])
        category_colors = reversed(["#ea3323", "#8afc63", "#4866ac"])
        names = ["Slot 1", "Slot 2"]#, "Slot 3"]
        ####
        alphas = np.ones(3)
        alphas[unavail_slot_idx] = 0.3
        ####
        data = np.flip(game.M, axis=0)
        data = np.delete(data, unavail_slot_idx, axis=1)
        data_cum = data.cumsum(axis=0)
        for i, color in enumerate(category_colors):
            heights = data[i, :]
            starting_point = data_cum[i, :] - heights
            rects = ax.bar(names, heights, bottom=starting_point, width=0.4,
                        color=color, edgecolor="black")
            #for bar, alpha in zip(rects, alphas):
            #    bar.set_alpha(alpha)
            ax.bar_label(rects, label_type='center', color='black', fontsize='x-large')
        ####
        plt.savefig(sdir + "machine.pdf")
        plt.close()
        ######
        print("++ Start sde for {}".format(sdir))
        ######
        '''
        filename = './rgbToken_qtable.npy'
        if os.path.isfile(filename):
            Q = np.load(filename, allow_pickle='TRUE').item()
        else:
            return ""
        '''
        #Q = game.iter_policy_eval(n_iterations=1, eps=1e-5, verbose=False, save=False)
        #joint_prob = get_joint_probabilities(game, Q, unavail_slot_idx)
        joint_prob = game.get_target_optimal_joint_dist(unavail_slot_idx)
        if not np.isclose(joint_prob.sum(), 1.):
            print("The joint probability is not quite complete and sums to: {} != 1".format(joint_prob.sum()))
        ### Intent selection
        p = joint_prob.sum(-1)
        chosen_intent = probab_flow_fct(p, sdir, kl_type, lambd, seed=trial_id)
        ### Action selection
        chosen_action = None
        p = None
        if chosen_intent is not None:
            p = joint_prob[chosen_intent]
            p = p / p.sum()
            chosen_action = probab_flow_fct(p, sdir, kl_type, lambd, seed=trial_id)
        #####
        pd.DataFrame(
            {"chosen_intent": [chosen_intent], "chosen_action": [chosen_action],
            "intent_proba": [joint_prob.sum(-1) / joint_prob.sum(-1).sum()],
            "action_proba": [joint_prob.sum(0) / joint_prob.sum(0).sum()],
            "joint_proba": [joint_prob / joint_prob.sum()]}
        ).to_csv(sdir + "trial_decision.csv", index=False)
    #####
    return sdir

################################################################################
def plot(sdirs):
    sns.set(context="paper", style="white", palette="tab10", font="Arial", font_scale=2,
        rc={"lines.linewidth": 1., "pdf.fonttype": 42, 'text.usetex' : True}
    )
    sns.set_palette("tab10", 10, 1)
    colors = sns.color_palette(n_colors=10)
    markers = ["o", "s", "d", "*", "+", "x", "v", "^", "<", ">"]
    fig = plt.figure(figsize=(8, 6))
    ####
    items_types = ["intents", "actions"]
    common_dir = os.path.commonprefix(sdirs)
    ####
    d_cases_proba = {}
    data = {"Case": [], "KL": [], "Trial Id.": [], "Decision Time (Intent)": [], "Decision Time (Action)": [],
        "Decision Time (Total)": [], "Chosen Intent": [], "Chosen Action": []
    }
    ####
    columns_intent_diff = ["x_Red", "x_Green", "x_Blue", "Decision Time",
        "KL", "Trial Id.", "Case"
    ]
    columns_action_diff = ["x_Slot1", "x_Slot2", "Decision Time",
        "KL", "Trial Id.", "Case"
    ]
    ####
    diffusion_data_intent = None#pd.DataFrame(columns=columns_intent_diff)
    diffusion_data_action = None#pd.DataFrame(columns=columns_action_diff)
    ####
    for sdir in tqdm(sdirs):
        parts_ = sdir[len(common_dir):].split("/")[0].split("_")
        case = "_".join([parts_[0], parts_[1]])
        kl_type = parts_[-1]
        trial_id = sdir[len(common_dir):].split("/")[1]
        #####
        kl_type_ = kl_type[:3]
        l_val = kl_type[4:].replace("p", ".")
        kl_type = kl_type_.upper() + r" $\lambda={{{}}}$".format(l_val)
        kl_type = kl_type.replace("JKL", r"$\lambda$-KL")
        #####
        df_intent = pd.read_csv(sdir + "diffusion_data_intents.csv")
        df_action = pd.read_csv(sdir + "diffusion_data_actions.csv")
        #####
        df_ = pd.DataFrame(columns=columns_intent_diff)
        x = df_intent["mahal_dist"].apply(lambda x: np.fromstring(x.replace('\n','')
            .replace('[','').replace(']','').replace('  ',' '), sep=' ')
        )#.astype(float)
        x = np.stack(list(x))#[:500]
        df_["x_Red"] = x[:, 0]
        df_["x_Green"] = x[:, 1]
        df_["x_Blue"] = x[:, 2]
        df_["Decision Time"] = list(df_intent["RT"])#[:500]
        df_["KL"] = kl_type
        df_["Trial Id."] = trial_id
        df_["Case"] = case
        if diffusion_data_intent is None:
            diffusion_data_intent = df_
        else:
            diffusion_data_intent = pd.concat([diffusion_data_intent, df_], ignore_index=True)
        #####
        df_ = pd.DataFrame(columns=columns_action_diff)
        x = df_action["mahal_dist"].apply(lambda x: np.fromstring(x.replace('\n','')
            .replace('[','').replace(']','').replace('  ',' '), sep=' ')
        )
        x = np.stack(list(x))#[:500]
        df_["x_Slot1"] = x[:, 0]
        df_["x_Slot2"] = x[:, 1]
        df_["Decision Time"] = list(df_action["RT"])#[:500]
        df_["KL"] = kl_type
        df_["Trial Id."] = trial_id
        df_["Case"] = case
        if diffusion_data_action is None:
            diffusion_data_action = df_
        else:
            diffusion_data_action = pd.concat([diffusion_data_action, df_], ignore_index=True)
        #####
        rt_intent = list(df_intent["RT"])[-1]
        rt_action = list(df_action["RT"])[-1]
        #####
        df = pd.read_csv(sdir + "trial_decision.csv")
        chosen_intent = int(df["chosen_intent"].iloc[0])
        chosen_action = int(df["chosen_action"].iloc[0])
        #####
        data["Case"] += [case]
        data["KL"] += [kl_type]
        data["Trial Id."] += [trial_id]
        data["Decision Time (Intent)"] += [rt_intent]
        data["Decision Time (Action)"] += [rt_action]
        data["Decision Time (Total)"] += [rt_intent + rt_action]
        data["Chosen Intent"] += [chosen_intent]
        data["Chosen Action"] += [chosen_action]
        #####
        if d_cases_proba.get(case, None) is None:
            intent_proba = np.fromstring(df["intent_proba"].iloc[0].replace('\n','')
                .replace('[','').replace(']','').replace('  ',' '), sep=' ')
            action_proba = np.fromstring(df["action_proba"].iloc[0].replace('\n','')
                .replace('[','').replace(']','').replace('  ',' '), sep=' ')
            #####
            d_cases_proba[case] = {
                "intent_proba": intent_proba,
                "action_proba": action_proba,
            }
    #########################################
    #########################################
    print("")
    print("Plot histograms.")
    data = pd.DataFrame(data)
    #print(data)
    ####
    columns_to_plot_cont = ["Decision Time (Intent)", "Decision Time (Action)", "Decision Time (Total)"]
    columns_to_plot_discr = ["Chosen Intent", "Chosen Action"]
    ####
    for case in list(d_cases_proba.keys()):
        print("++ ", case)
        df = data.loc[data.Case.isin([case])]
        for col in columns_to_plot_cont:
            plt.clf()
            sns.displot(df, x=col, hue="KL", element="step")
            plt.tight_layout()
            plt.savefig(common_dir + col.replace(" ", "_").replace("(", "").replace(")", "") + "_" + case + ".pdf")
            plt.close()
        #####
        """
        plt.clf()
        sns.histplot(
            df, x="Decision Time (Action)", y="Decision Time (Intent)", hue="KL",
            bins=5, discrete=(True, True), log_scale=(False, False),
            alpha=0.5,
            #cbar=True, cbar_kws=dict(shrink=.75),
        )
        plt.tight_layout()
        plt.show()
        plt.close()"""
        #####
        print(d_cases_proba[case])
        for col in columns_to_plot_discr:
            #unique_values, counts = np.unique(list(df[col]), return_counts=True)
            unique_values = [0, 1, 2] if "Intent" in col else [0, 1]
            display_labels = ["Red", "Green", "Blue"] if "Intent" in col else ["Slot 1", "Slot 2"]
            #####
            plt.clf() # Clear current figure
            fig, ax = plt.subplots() # Create a figure and axes explicitly for easier management
            ##
            p = d_cases_proba[case]["intent_proba"] if "Intent" in col else d_cases_proba[case]["action_proba"]
            counts = (np.round(p, PROBA_ROUND_PRECISION) * PROBA_ROUND_MULT).astype(int)
            sns.barplot(x=unique_values, y=counts, alpha=0.3, color=colors[-1], ax=ax)
            #####
            countplot_artist = sns.countplot(x=col, hue="KL", data=df, alpha=0.6, ax=ax)#, alpha=0.7)
            #####
            # Set the tick locations explicitly before setting the labels
            ax.set_xticks(unique_values)
            ax.set_xticklabels(display_labels)
            #####
            # 3. Create a custom patch for the barplot
            barplot_label = r'Intent sel. $\mathcal{I}^{*}$' if "Intent" in col else r'Softmax pol. $\pi^{*}$'# The label you want for the barplot
            custom_patch = mpatches.Patch(color=colors[-1], alpha=0.3, label=barplot_label)
            # 4. Combine legend handles and labels
            # Get existing handles and labels from the countplot (from 'ax' object)
            handles, labels = ax.get_legend_handles_labels()
            # Append the custom handle to the list of handles
            handles.append(custom_patch)
            labels.append(barplot_label)
            # 5. Apply the combined legend to the axes
            ax.legend(handles=handles, labels=labels, title="KL") # Add a title if needed
            #####
            plt.tight_layout()
            plt.savefig(common_dir + col.replace(" ", "_") + "_" + case + ".pdf")
            plt.close()
        #####
    """
    df = data.loc[data.Case.isin([list(d_cases_proba.keys())[0]])]
    sns.displot(df, x="Decision Time (Action)", hue="KL", element="step")
    plt.show()
    """
    #########################################
    #########################################
    print("")
    print("Plot violin plots.")
    data['Cases'] = data['Case'].str.replace('LU_', '').str.replace('HU_', '')
    for col in columns_to_plot_cont:
        plt.clf()
        sns.violinplot(data, x="Cases", y=col, hue="KL")
        plt.grid(True, axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(common_dir + col.replace(" ", "_").replace("(", "").replace(")", "") + "_violin" + ".pdf")
        plt.close()
    #########################################
    #########################################
    print("")
    """
    print("Plot diffusion aggregates.")
    for case in list(d_cases_proba.keys()):
        print("++ ", case)
        plt.clf()
        fig, axes = plt.subplots(1, 2, figsize=(12,9))
        df_intent = diffusion_data_intent.loc[diffusion_data_intent.Case.isin([case])].melt(
            id_vars=["Decision Time", "KL", "Trial Id.", "Case"],
            value_vars=["x_Red", "x_Green", "x_Blue"],
            var_name='x',
            value_name='Distance',
        )
        #sns.lineplot(df_intent, x='Decision Time', y='x_Red', style='KL', color="#ea3323", ax=axes[0])
        #sns.lineplot(df_intent, x='Decision Time', y='x_Green', style='KL', color="#8afc63", ax=axes[0])
        #sns.lineplot(df_intent, x='Decision Time', y='x_Blue', style='KL', color="#4866ac", ax=axes[0])
        sns.lineplot(df_intent, x='Decision Time', y='Distance', hue='x', style='KL',
            palette=["#ea3323", "#8afc63", "#4866ac"], markers=markers[:3], ax=axes[0]
        )
        ######
        df_action = diffusion_data_action.loc[diffusion_data_action.Case.isin([case])].melt(
            id_vars=["Decision Time", "KL", "Trial Id.", "Case"],
            value_vars=["x_Slot1", "x_Slot2"],
            var_name='x',
            value_name='Distance',
        )
        #sns.lineplot(df_action, x='Decision Time', y='x_Slot1', style='KL', color="#ed702d", ax=axes[1])
        #sns.lineplot(df_action, x='Decision Time', y='x_Slot2', style='KL', color="#00b0a5", ax=axes[1])
        sns.lineplot(df_action, x='Decision Time', y='Distance', hue='x', style='KL',
            palette=["#ed702d", "#00b0a5"], markers=markers[:2], ax=axes[1]
        )
        ######
        # Add a horizontal line at y=1.
        axes[0].axhline(y=1., color='black', linestyle='--', label="Threshold")
        axes[1].axhline(y=1., color='black', linestyle='--', label="Threshold")
        ######
        plt.tight_layout()
        plt.savefig(common_dir + "diffusion_" + case + ".pdf")
        plt.close()
    """
    #########
    plt.close()

################################################################################
def main():
    make_dirs("./reaction_times/RGB_TOKEN/")
    pool = mp.Pool(processes=N_PROCESS)
    configs = [
        #(case, transition_probabilities, valuable_tokens, unavail_slot_idx, kl_type, lambd, trial_id, IS_RUN)
        #for (case, transition_probabilities, valuable_tokens, unavail_slot_idx), kl_type, trial_id in itertools.product(
        #    CASES, KL_TYPES, LAMBDA_VALUES, range(1, N_TRIALS+1)
        #)
        (case, transition_probabilities, valuable_tokens, unavail_slot_idx, "", lambd, trial_id, IS_RUN)
        for (case, transition_probabilities, valuable_tokens, unavail_slot_idx), lambd, trial_id in itertools.product(
            CASES, LAMBDA_VALUES, range(1, N_TRIALS+1)
        )
    ]
    sdirs = pool.starmap(run_sde, configs)
    pool.close()
    #####
    plot(sdirs)

################################################################################
if __name__ == "__main__":
    args = fetch_args()
    R = args.R
    main()
