import scipy.io as sio
import torch
import scipy.sparse as sp
import numpy as np
import random
import torch.nn.functional as F

def pdist(sample_1, sample_2, norm=2, eps=1e-5):
    """Compute the matrix of all squared pairwise distances.
    Arguments
    ---------
    sample_1 : torch.Tensor or Variable
        The first sample, should be of shape ``(n_1, d)``.
    sample_2 : torch.Tensor or Variable
        The second sample, should be of shape ``(n_2, d)``.
    norm : float
        The l_p norm to be used.
    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
        ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    norm = float(norm)
    if norm == 2.:
        norms_1 = torch.sum(sample_1**2, dim=1, keepdim=True)
        norms_2 = torch.sum(sample_2**2, dim=1, keepdim=True)
        norms = (norms_1.expand(n_1, n_2) +
                 norms_2.transpose(0, 1).expand(n_1, n_2))
        distances_squared = norms - 2 * sample_1.mm(sample_2.t())
        return torch.sqrt(eps + torch.abs(distances_squared))
    else:
        dim = sample_1.size(1)
        expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
        expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
        differences = torch.abs(expanded_1 - expanded_2) ** norm
        inner = torch.sum(differences, dim=2, keepdim=False)
        return (eps + inner) ** (1. / norm)

def wasserstein(x, y, device, p=0.5, lam=10, its=10, sq=False, backpropT=False, cuda=False):
    """return W dist between x and y"""
    '''distance matrix M'''
    nx = x.shape[0]
    ny = y.shape[0]

    #x = x.squeeze()
    #y = y.squeeze()

    #    pdist = torch.nn.PairwiseDistance(p=2)

    M = pdist(x, y)  # distance_matrix(x,y,p=2)

    '''estimate lambda and delta'''
    M_mean = torch.mean(M)
    M_drop = F.dropout(M, 10.0 / (nx * ny))
    delta = torch.max(M_drop).cpu().detach()
    eff_lam = (lam / M_mean).cpu().detach()

    '''compute new distance matrix'''
    Mt = M
    row = delta * torch.ones(M[0:1, :].shape)
    col = torch.cat([delta * torch.ones(M[:, 0:1].shape), torch.zeros((1, 1))], 0)
    if cuda:
        #row = row.cuda()
        #col = col.cuda()
        row = row.to(device)
        col = col.to(device)
    Mt = torch.cat([M, row], 0)
    Mt = torch.cat([Mt, col], 1)

    '''compute marginal'''
    a = torch.cat([p * torch.ones((nx, 1)) / nx, (1 - p) * torch.ones((1, 1))], 0)
    b = torch.cat([(1 - p) * torch.ones((ny, 1)) / ny, p * torch.ones((1, 1))], 0)

    '''compute kernel'''
    Mlam = eff_lam * Mt
    temp_term = torch.ones(1) * 1e-6
    if cuda:
        #temp_term = temp_term.cuda()
        #a = a.cuda()
        #b = b.cuda()
        temp_term = temp_term.to(device)
        a = a.to(device)
        b = b.to(device)
    K = torch.exp(-Mlam) + temp_term
    U = K * Mt
    ainvK = K / a

    u = a

    for i in range(its):
        u = 1.0 / (ainvK.matmul(b / torch.t(torch.t(u).matmul(K))))
        if cuda:
            #u = u.cuda()
            u = u.to(device)
    v = b / (torch.t(torch.t(u).matmul(K)))
    if cuda:
        #v = v.cuda()
        v = v.to(device)

    upper_t = u * (torch.t(v) * K).detach()

    E = upper_t * Mt
    D = 2 * torch.sum(E)

    if cuda:
        #D = D.cuda()
        D = D.to(device)

    return D, Mlam

def pdist2sq(x_t, x_cf):
    C = -2 * torch.matmul(x_t,torch.t(x_cf))
    n_t = torch.sum(x_t * x_t, 1, True)
    n_cf = torch.sum(x_cf * x_cf, 1, True)
    D = (C + torch.t(n_cf)) + n_t
    return D

def mmd2_rbf(Xt, Xc, p,sig):
    """ Computes the l2-RBF MMD for X given t """

    Kcc = torch.exp(-pdist2sq(Xc,Xc)/(sig)**2)
    Kct = torch.exp(-pdist2sq(Xc,Xt)/(sig)**2)
    Ktt = torch.exp(-pdist2sq(Xt,Xt)/(sig)**2)

    m = Xc.shape[0]
    n = Xt.shape[0]

    mmd = (1.0-p)**2/(m*(m-1.0))*(torch.sum(Kcc)-m)
    mmd = mmd + (p) ** 2/(n*(n-1.0))*(torch.sum(Ktt)-n)
    mmd = mmd - 2.0*p*(1.0-p)/(m*n)*torch.sum(Kct)
    mmd = 4.0*mmd

    return mmd

def mmd2_lin(Xt, Xc,p):
    ''' Linear MMD '''
    mean_control = torch.mean(Xc,0)
    mean_treated = torch.mean(Xt,0)

    mmd = torch.sum((2.0*p*mean_treated - 2.0*(1.0-p)*mean_control) ** 2)

    return mmd

def safe_sqrt(x, lbound=1e-10):
    ''' Numerically safe version of pytorch sqrt '''
    return torch.sqrt(torch.clamp(x, lbound, np.inf))
