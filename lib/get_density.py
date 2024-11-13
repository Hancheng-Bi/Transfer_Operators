import numpy as np
import scipy as sp
import ot
import torch
from lib.SinkhornNP import SolveOT





def getDistSqrTorus(x,y):
    dim = 1
    if (x.ndim == 2):
        dim = np.shape(x)[1]
    m = np.shape(x)[0]
    n = np.shape(y)[0]
    return sp.linalg.norm((x.reshape((m,1,dim))-y.reshape((1,n,dim)) + 0.5)%1 - 0.5,axis = 2)**2

# def cost(X,Y):
    
#     return getDistSqrTorus(X,Y)

def cost(X,Y,dev):
    return torch.tensor(getDistSqrTorus(X,Y),device = dev,dtype = torch.float64)

def dens_gauss_shift(X, Y, shift, std, shift_prob=0.5):
    """
    X, Y same shape numpy arrays to describe point clouds
    shift = shift of second diagonal
    std = standard deviation of both diagonals
    shift_prob = probability of going to the se
    """
    
    o = np.ceil(4 * std)
    
    a = (Y - X).reshape((*X.shape, 1))
    a = np.nan_to_num(a,nan = std*np.sqrt(2/np.pi) %1)
    z = np.arange(-o, o+1, dtype=float).reshape((*[1 for _ in X.shape], -1))
    
    d0 = 1 / ((2 * np.pi)**.5 * std) * np.sum(np.exp(-(a - z)**2 / (2 * std**2)), axis=-1)
    d1 = 1 / ((2 * np.pi)**.5 * std) * np.sum(np.exp(-(a - shift - z)**2 / (2 * std**2)), axis=-1)
    
    return shift_prob * d1 + (1 - shift_prob) * d0

def sample_Gau(gen,num ,std, shift,dim = 1, shift_prob = 0.5):
    x = gen.random(num)
    shift_ind = gen.choice([0,1], p=[1-shift_prob,shift_prob],size = num)
    gau = gen.normal(0, std, size = num)
    y = x + shift_ind*shift + gau
    y = y%1
    while (dim > 1):
        xx = gen.random(num)
        yy = gen.random(num)
        x = np.vstack((x, xx))
        y = np.vstack((y, yy))
        dim -= 1
    return x.T,y.T

# def EMML(EK_x,EK_y,M,EMML_itr,mus = None):
#     """
#     return probability density xi that minimise J_M^N(E_k(xi)) which is defined in Rmk17 and Def19 in the paper.
#     EK_x = transport plan k_{mu,tilde{mu}}. mu otimes mu as defined in Prop18 
#     EK_y = transport plan k_{nu,tilde{nu}}.nu otimes nu
#     M = number of point pairs in one batch
#     EMML_itr = iteration loops
#     mus = probability vector represent the subsampled mu_S, default is uniform.
#     """

#     N = int(EK_x.shape[0]/M)
#     S = int(EK_x.shape[1])
#     mn = M * N
#     if mus is None:
#         mus = np.full(S,1./S)
#     def Pti(x, i):
#         return (EK_y.T @ x)[:, np.newaxis] * np.sum(EK_x.T[:,i*M:(i+1)*M], axis=1)[np.newaxis, :] * (S**2 * N)

#     def Pii(rho, i):
#         return (EK_y[i*M:(i+1)*M,:] @ (rho @ np.sum(EK_x.T[:,i*M:(i+1)*M], axis=1))) * (S**2 * N)

#     def Ptii(x, i):
#         return (EK_y[i*M:(i+1)*M,:].T @ x)[:, np.newaxis] * np.sum(EK_x.T[:,i*M:(i+1)*M], axis=1)[np.newaxis, :] * (S**2 * N)



#     Pcs = [Pti(np.ones(N * M), i) for i in range(N)]
#     Pcs = np.sum(Pcs,axis=0)
#     rho = np.full((S, S), S**-2)
#     rho *= S * mus[:,np.newaxis]
#     for _ in range(EMML_itr):
#         d = np.zeros((S, S))
#         for i in range(N):
#             d += Ptii(1. / Pii(rho, i), i)
#         rho *= d / mn
#         rho /= Pcs
#         rho /= np.sum(rho, axis=1)[:,np.newaxis]
#         rho *= S * mus[:,np.newaxis] 
        
#     return rho/S

def EMML(EK_x,EK_y,M,EMML_itr,dev,mus = None):
    """
    return probability density xi that minimise J_M^N(E_k(xi)) which is defined in Rmk17 and Def19 in the paper.
    EK_x = transport plan k_{mu,tilde{mu}}. mu otimes mu as defined in Prop18 
    EK_y = transport plan k_{nu,tilde{nu}}.nu otimes nu
    M = number of point pairs in one batch
    EMML_itr = iteration loops
    mus = probability vector represent the subsampled mu_S, default is uniform.
    """

    N = int(EK_x.shape[0]/M)
    S = int(EK_x.shape[1])
    mn = M * N

    if mus is None:
        mus = torch.full([S],1./S,device = dev, dtype = torch.float64)
    def Pti(x, i):
        return (EK_y.T @ x)[:, None] * torch.sum(EK_x.T[:,i*M:(i+1)*M], axis=1)[None, :] * (S**2 * N)

#     def Pii(rho, i):
#         return (EK_y[i*M:(i+1)*M,:] @ (rho @ torch.sum(EK_x.T[:,i*M:(i+1)*M], axis=1))) * (S**2 * N)

#     def Ptii(x, i):
#         return (EK_y[i*M:(i+1)*M,:].T @ x)[:, ]] * torch.sum(EK_x.T[:,i*M:(i+1)*M], axis=1)[torch.newaxis, :] * (S**2 * N)

    def Pii_parallel(rho):
        # (EK_y[i*M:(i+1)*M,:] @ (rho @ torch.sum(EK_x.T[:,i*M:(i+1)*M], axis=1))) * (S**2 * N)
        s1,s2 = EK_y.shape
        EK_y_slices = EK_y.view(N, M, S) # (N, M, S)
        rho_mult = rho @ EK_x.T.reshape(S, N, M).sum(-1)# (s, N)
        rho_mult = rho_mult.T.reshape(N, S, 1)
        return (EK_y_slices @ rho_mult) * (S**2 * N) # (N, M, 1)

    def Ptii_parallel(x):
        # apply Ptii to the slices of x
        EK_y_slices = EK_y.view(N, M, S).permute(0,2,1) # (N, S, M)
        EK_y_x = EK_y_slices @ x # (N,S,1)
        EK_x_sum = EK_x.T.view(S,N,M).sum(-1) #(S, N)
        return (EK_y_x * EK_x_sum.T.reshape(N, 1, S)).sum(0) * (S**2 * N)






    # Pcs = [Pti(torch.ones(N * M,device = dev,dtype = torch.float64), i).cpu().numpy() for i in range(N)]
    Pcs = torch.concat([Pti(torch.ones(N * M,device = dev,dtype = torch.float64), i) for i in range(N)])
    Pcs = torch.sum(Pcs,axis=0)
    # Pcs = torch.tensor(Pcs,device = dev,dtype = torch.float64)
    rho = torch.full((S, S), (1.*S)**-2,device = dev, dtype = torch.float64)
    rho *= S * mus[:,None]
    for _ in range(EMML_itr):
    #     d = torch.zeros((S, S),device = dev, dtype = torch.float64)
    #     for i in range(N):
    #         d += Ptii(1. / Pii(rho, i), i)
        d = Ptii_parallel(1./Pii_parallel(rho))
        rho *= d / mn
        rho /= Pcs
        rho /= torch.sum(rho, axis=1)[:,None]
        rho *= S * mus[:,None] 
        
    return rho/S



def get_Dens(gen,M:int,N:int,S:int,std:float,jump:float,jump_prob:float,ve:float,subsample:bool,dev,EMML_itr:int = 10000,E:int = 200):
    """
    Simulate data and return estimated density on a linspace meshgrid
    M = number of pointpairs in one Batch
    N = number of Batches
    S = number of subsampled points, S < M * N
    std = standard deviation for distance between two species
    jump = shift distance
    jump_prob = shift probability
    ve = sinkhorn regulariser
    subsample = if subsample or not
    dev = device used(cpu/gpu)
    EMML_itr = EMML iterations
    E = resolution of meshgrid
    """
    mn = M*N
    x,y = sample_Gau(gen,num = mn ,std = std, shift = jump, shift_prob = jump_prob)
    #Subsample points
    xx = torch.tensor(x,device = dev,dtype = torch.float64)
    yy = torch.tensor(y,device = dev,dtype = torch.float64)
    S1 = gen.integers(0, mn, size = S)
    S2 = gen.integers(0, mn, size = S)
    if (subsample):
        sx = x[S1]
        sy = y[S2]
    else:
        sx = x
        sy = y
        S = mn
    LX = ot.sinkhorn(torch.ones(mn, device = dev, dtype = torch.float64)/mn,
                torch.ones(S,device = dev,dtype = torch.float64)/S, cost(x,sx,dev),ve
                ,log = True,numItermax = 1000000,method = 'sinkhorn_log')
    LY = ot.sinkhorn(torch.ones(mn, device = dev, dtype = torch.float64)/mn,
                    torch.ones(S,device = dev,dtype = torch.float64)/S, cost(y,sy,dev),ve
                    ,log = True,numItermax = 1000000,method = 'sinkhorn_log')
    EK_x = LX[0]
    EK_y = LY[0]

    rho = EMML(EK_x,EK_y,M,EMML_itr,dev = dev)

    x_e = y_e = np.linspace(0,1,E,endpoint=False)
    xpot = 1/(torch.sum(torch.exp(-cost(x_e,sx,dev)/ve + LX[1]["log_v"]),axis=1))
    ypot = 1/(torch.sum(torch.exp(-cost(y_e,sy,dev)/ve + LY[1]["log_v"]),axis=1))
    F_X = xpot[:,None]*torch.exp(-cost(x_e,sx,dev)/ve + LX[1]["log_v"])*S
    F_Y = ypot[:,None]*torch.exp(-cost(y_e,sy,dev)/ve + LY[1]["log_v"])*S
    return F_Y@rho@F_X.T


# def get_Dens(gen,M:int,N:int,S:int,std:float,jump:float,jump_prob:float,ve:float,subsample:bool,EMML_itr:int = 10000,E:int = 100):
#     """
#     Simulate data and return estimated density on a linspace meshgrid
#     M = number of pointpairs in one Batch
#     N = number of Batches
#     S = number of subsampled points, S < M * N
#     std = standard deviation for distance between two species
#     jump = shift distance
#     jump_prob = shift probability
#     ve = sinkhorn regulariser
#     subsample = if subsample or not
#     EMML_itr = EMML iterations
#     E = resolution of meshgrid
#     """
#     mn = M*N
#     x,y = sample_Gau(gen,num = mn ,std = std, shift = jump, shift_prob = jump_prob)
#     #Subsample points
#     S1 = gen.integers(0, mn, size = S)
#     S2 = gen.integers(0, mn, size = S)
#     if (subsample):
#         sx = x[S1]
#         sy = y[S2]
#     else:
#         sx = x
#         sy = y
#         S = mn
#     Res_X = SolveOT(np.ones(mn)/mn,np.ones(S)/S,cost(x,sx),1e-9,ve,1,returnSolver = True)
#     EK_x = Res_X[1].toarray() #Transport plan from mu to subsampled x points
#     Res_Y = SolveOT(np.ones(mn)/mn,np.ones(S)/S,cost(y,sy),1e-9,ve,1,returnSolver = True)
#     EK_y = Res_Y[1].toarray() #Transport plan from nu to subsampled y points

#     #EMML minimisation of cost function
#     rho = EMML(EK_x,EK_y,M,EMML_itr)

#     #Do kernel extension for illustration, x_e and y_e should be close to the true marginal (in this case is uniform so we take linspace) 
#     x_e = y_e = np.linspace(0,1,E,endpoint = False)
#     xpot = 1/(np.sum(np.exp((-cost(x_e,sx) + Res_X[2].beta)/ve),axis=1))
#     ypot = 1/(np.sum(np.exp((-cost(y_e,sy) + Res_Y[2].beta)/ve),axis=1))
#     F_X = xpot[:,np.newaxis]*np.exp((-cost(x_e,sx) + Res_X[2].beta)/ve)*S
#     F_Y = ypot[:,np.newaxis]*np.exp((-cost(y_e,sy) + Res_Y[2].beta)/ve)*S
    
#     return F_Y@rho@F_X.T



def get_L2_estimator(gen,M:int,N:int,S:int,std:float,jump:float,jump_prob:float,ve:float,subsample:bool,dev,EMML_itr:int = 10000,E:int = 100):
    """
    Return discretised L2 norm between simulated density estimator and the true one
    M = number of pointpairs in one Batch
    N = number of Batches
    S = number of subsampled points, S < M * N
    std = standard deviation for distance between two species
    jump = shift distance
    jump_prob = shift probability
    ve = sinkhorn regulariser
    subsample = if subsample or not
    EMML_itr = EMML iterations
    E = resolution of discretisation
    """
    Mat = get_Dens(gen,M,N,S,std,jump,jump_prob,ve,subsample,dev,EMML_itr,E)
    x_e = y_e = np.linspace(0,1,E,endpoint = False)
    True_M = dens_gauss_shift(*np.meshgrid(x_e,y_e),jump, std, jump_prob)
    return np.linalg.norm(True_M - Mat.cpu().numpy())/E

def get_L2_uniform(std:float,jump:float,jump_prob:float,E:int = 100):
    """
    Return discretised L2 norm between the true density and Uniform density
    std = standard deviation for distance between two species
    jump = shift distance
    jump_prob = shift probability
    E = resolution of discretisation
    """
    x_e = y_e = np.linspace(0,1,E,endpoint = False)
    True_M = dens_gauss_shift(*np.meshgrid(x_e,y_e),jump, std, jump_prob)
    return np.linalg.norm(True_M - np.ones(E**2).reshape((E,E)))/E


def get_L2_true(ve : float,std:float,jump:float,jump_prob:float,E:int = 100):
    """
    [ONLY FOR TORUS + GAUSSIAN + SQUARED-DISTANCE-COST CASE]
    Return discretised L2 norm between the true density and asymptotic estimator as N to infinity
    ve = sinkhorn regulariser
    std = standard deviation for distance between two species
    jump = shift distance
    jump_prob = shift probability
    E = resolution of discretisation
    """
    x_e = y_e = np.linspace(0,1,E,endpoint = False)
    True_M = dens_gauss_shift(*np.meshgrid(x_e,y_e),jump, std, jump_prob)
    True_est_M = dens_gauss_shift(*np.meshgrid(x_e,y_e),jump, np.sqrt(std**2 + ve), jump_prob)
    return np.linalg.norm(True_M - True_est_M)/E