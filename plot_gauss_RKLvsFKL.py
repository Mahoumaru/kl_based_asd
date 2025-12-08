import numpy as np; import torch as th
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.categorical import Categorical
from torch.distributions.independent import Independent
from torch.distributions.multivariate_normal import MultivariateNormal

import matplotlib as mpl
mpl.rcParams["text.usetex"] = True
mpl.rcParams.update({"font.size": 20})
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns

from scipy.stats import norm
from scipy.stats import multivariate_normal

def symsqrt(matrix):
    _, s, v = matrix.svd()
    above_cutoff = s > s.max() * s.size(-1) * th.finfo(s.dtype).eps
    s = s[..., above_cutoff]
    v = v[..., above_cutoff]
    return (v * s.sqrt().unsqueeze(-2)) @ v.transpose(-2, -1)


def get_vertices(n, r=1., w=None):
    a = r * np.sqrt(n / (n-1))
    I = np.eye(n)
    ones_ = np.ones(n)
    ######
    Q = np.eye(n)
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


def project_vertices_onto_hyperplane(V):
    n_plus_1 = V.shape[1]   # dimension is n+1
    n = n_plus_1 - 1        # the hyperplane is n-dimensional
    B = np.zeros((n_plus_1, n))
    for i in range(n):
        B[i, i]   = 1.0
        B[-1, i]  = -1.0
    #####
    Q, R = np.linalg.qr(B)    # Q has shape (n+1, n)
    coords = V @ Q
    #####
    return coords, Q


def get_derX_logprob(x, proba):
    y = x.clone()
    y.requires_grad_()
    log_prob = proba.log_prob(y)
    ###
    grad_output = th.ones_like(log_prob)
    ###
    log_prob.backward(gradient=grad_output)
    return y.grad, log_prob.detach()

R = 3
sc = 1.

p = np.array([0.5, 0.5])
p = np.array([0.3, 0.7])
#p = np.array([0.95, 0.05])
#p = np.array([0.05, 0.95])

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
mixture_sigmas = sc * th.FloatTensor(np.eye(d).reshape(1, d, d).repeat(n_choices, axis=0))
####
comp = MultivariateNormal(mixture_means, mixture_sigmas)
mix = Categorical(th.FloatTensor(p))
mixture_dist = MixtureSameFamily(mix, comp)
###########

mpl.rcParams.update({"font.size": 30})
lw = 3
kl_types = ["rkl", "fkl"]

#"""
fig, ax = plt.subplots(1,2, figsize=(15,5), sharey=True)

for i, kl_type in enumerate(kl_types):
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
    ########
    N = 1000
    ########
    p1 = norm(loc=mixture_means[0].item(), scale=mixture_sigmas[0].item())
    p2 = norm(loc=mixture_means[1].item(), scale=mixture_sigmas[1].item())
    q = norm(loc=mu.item(), scale=sigma.item())
    ########
    min_ppf = 0.001
    ########
    ax[i].tick_params(
        axis='both',          # changes apply to both x and y-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False) # labels along the bottom edge are off
    ########
    x = np.linspace(min(p1.ppf(min_ppf), p2.ppf(min_ppf)), max(p1.ppf(1. - min_ppf), p2.ppf(1. - min_ppf)), 100)
    mixt_pdf = p[0] * p1.pdf(x) + p[1] * p2.pdf(x)
    mask = (x < 0.)#*(x>-1)
    ax[i].plot(x[mask], mixt_pdf[mask], label=r'Option 1', lw=lw)
    ax[i].plot(x[~mask], mixt_pdf[~mask], label=r'Option 2', lw=lw)
    #ax[i].plot(x, p[0] * p1.pdf(x) + p[1] * p2.pdf(x), label=r'$p(x)$', lw=lw)
    ########
    x = np.linspace(q.ppf(min_ppf), q.ppf(1. - min_ppf), 100) if kl_type == "rkl" else x
    #ax[i].plot(x, q.pdf(x), label=r"$\pi^{*}(x)$", ls=(0, (5, 10)), c='g', lw=lw)
    #ax[i].plot(x, q.pdf(x), label=r"$q^{*}(x)$", ls=(0, (5, 10)), c='g', lw=lw)
    #ax[i].plot(x, q.pdf(x), label=r"Choice", ls="--", c='g', lw=lw)
    ########
    #ax[i].set_xlabel(r'$x$ (e.g. action/intent)')
    ax[i].set_xlabel(r'$x$')
    ax[i].spines['top'].set_linewidth(lw)
    ax[i].spines['bottom'].set_linewidth(lw)
    ax[i].spines['left'].set_linewidth(lw)
    ax[i].spines['right'].set_linewidth(lw)
    ########
    #legend = ax[i].legend()
    #frame = legend.get_frame()
    #frame.set_edgecolor('black')
    #frame.set_linewidth(lw)
    ########
    ax[i].grid(True, linestyle='--', alpha=0.6)

#plt.show()
plt.savefig("plot_kl_p{}.pdf".format(str(p[0]).replace(".", "")))
#"""

##############################
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
q = MultivariateNormal(mu, sigma)

####
mpl.rcParams.update({"font.size": 18})
fig, ax = plt.subplots(1,1)#, figsize=(15,5))
assert sc == 1
D = 1.

min_mu = th.min(mixture_means)
max_mu = th.max(mixture_means)
y = th.ones_like(mixture_means[0])

print(min_mu, max_mu)
sigma = mixture_sigmas[0]
x = th.FloatTensor(np.linspace(min_mu + sigma @ y, max_mu - sigma @ y, 100))
###
derX_logq, logq = get_derX_logprob(x, q)
derX_logp, logp = get_derX_logprob(x, mixture_dist)

lambda_vals = [0., 0.2, 0.4, 0.6, 0.8, 1.]
max_drift = -np.inf
min_drift = np.inf
for lambd in lambda_vals:
    ###
    ratio = (logp - logq).squeeze().detach().exp()
    ratio = 1. + lambd * (ratio - 1.)
    ##
    vt_exc = ratio * (derX_logp - derX_logq).squeeze()
    drift_exc = (vt_exc + D * derX_logq.squeeze()).numpy()
    ####
    M = max(drift_exc)
    if M > max_drift:
        max_drift = M
    ####
    M = min(drift_exc)
    if M < min_drift:
        min_drift = M
    ####
    ax.plot(x.squeeze().numpy(), drift_exc, label=rf'$\lambda={lambd}$')#, lw=lw)


ax.axvline(x=R, ymin=min_drift-0.1, ymax=max_drift+0.1, linewidth=1, ls="--", color='black')
ax.axvline(x=-R, ymin=min_drift-0.1, ymax=max_drift+0.1, linewidth=1, ls="--", color='black')
ax.set_xlim(-(R+1), (R+1))
ax.set_xticks(np.arange(-(R+1), (R+2), 1.))
ax.set_xlabel(r'Neural state $x$')
ax.set_ylabel(r'Drift $\mu(x)$')

plt.tight_layout()
plt.grid(True, linestyle='--', alpha=0.6)
# 1. Save the main figure WITHOUT the legend
# The 'bbox_extra_artists' argument ensures the legend is cut off from the main image
plt.savefig("plot_klddm_drift_p{}.pdf".format(str(p[0]).replace(".", "")))#,
#            bbox_extra_artists=(legend,),
#            bbox_inches='tight')

legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), framealpha=1, frameon=True)
# 2. Save the legend as a separate file
def save_legend_separately(legend, filename="legend.pdf"):
    """Saves a matplotlib legend object as a standalone image file."""
    # Create a new, empty figure and axis
    fig_legend = plt.figure(figsize=(3, len(lambda_vals)*0.5)) # Adjust size as needed
    ax_legend = fig_legend.add_subplot(111)
    # Place the legend in this new figure
    ax_legend.legend(handles=legend.legend_handles, labels=[text.get_text() for text in legend.get_texts()], loc='center')
    # Turn off the axis display for a clean look
    ax_legend.axis('off')
    # Save the new figure, bounding tightly around the legend content
    fig_legend.savefig(filename, bbox_inches='tight')
    plt.close(fig_legend) # Close the temporary figure

# Call the function to save the legend
save_legend_separately(legend, filename="plot_klddm_drift_p{}_legend.pdf".format(str(p[0]).replace(".", "")))

# Finally, display or close the original plot
# plt.show()
plt.close()
