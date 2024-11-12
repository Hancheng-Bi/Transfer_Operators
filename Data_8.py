import numpy as np
import pandas as pd
import time, sys, os, itertools

# path prefix for stored samples
sample_path = "samples_torus_jump"

def pt(x, y, t, p_jump, d_jump):
    if p_jump == 0:
        r = np.ceil(8. * t) + 1
        xy = x - y
        z = np.arange(-r, r+1)
        z = z.reshape((*[1 for _ in xy.shape], -1))
        xy = xy.reshape((*xy.shape, 1))
        a = (xy + z)**2
        return 1. / ((2 * np.pi)**.5 * t) * np.sum(np.exp(-a / (2 * t**2)), axis=-1)
    elif p_jump == 1:
        return pt((x + d_jump) % 1, y, t, 0, 0)
    else:
        return (1 - p_jump) * pt(x, y, t, 0, 0) + p_jump * pt(x, y, t, 1, d_jump)

def dpt(x, y, t, p_jump, d_jump):
    if p_jump == 0:
        r = np.ceil(8. * t) + 1
        xy = x - y
        z = np.arange(-r, r+1)
        z = z.reshape((*[1 for _ in xy.shape], -1))
        xy = xy.reshape((*xy.shape, 1))
        a = (xy + z)**2
        return 1. / ((2 * np.pi)**.5 * t**4) * np.sum((a - t**2) * np.exp(-a / (2 * t**2)), axis=-1)
    elif p_jump == 1:
        return dpt((x + d_jump) % 1, y, t, 0, 0)
    else:
        return (1 - p_jump) * dpt(x, y, t, 0, 0) + p_jump * dpt(x, y, t, 1, d_jump)

def sample(gen, M, t, p_jump, d_jump):
    X = gen.random(M)
    Y = (X + gen.normal(0, t, size=M) + d_jump * (gen.random(M) < p_jump).astype(float)) % 1
    return X.reshape((M, 1)), Y.reshape((1, M))

def I_sample(gen, M, tgen, teval, p_jump, d_jump):
    X, Y = sample(gen, M, tgen, p_jump, d_jump)
    p = pt(X, Y, teval, p_jump, d_jump)
    dp = dpt(X, Y, teval, p_jump, d_jump)
    F = np.sum(dp, axis=1) / np.sum(p, axis=1)
    return np.sum(F.reshape((1,-1)) * F.reshape((-1, 1)))

def I_limit(gen, t, p_jump, d_jump, sres=10000, ires=200):
    y1 = gen.random((sres,1))
    y2 = gen.random((sres,1))
    x1 = np.linspace(0, 1, ires, endpoint=False).reshape((1,ires))
    a = dpt(x1, y2, t, p_jump, d_jump)
    b = pt(x1, y1, t, p_jump, d_jump)
    ss = np.sum(a**2 * b,axis=-1) / ires - (np.sum(a * b,axis=-1) / ires)**2
    return np.mean(ss), np.std(ss) / sres**.5

def f(X, Y, t, p_jump, d_jump):
    return np.sum(np.log(np.mean(pt(X, Y, t, p_jump, d_jump), axis=1)))

def f_multi(Xs, Ys, t, p_jump, d_jump):
    return np.sum([f(X, Y, t, p_jump, d_jump) for X, Y in zip(Xs, Ys)])

def sample_multi(gen, N, M, t, p_jump, d_jump):
    Xs, Ys = [], []
    for _ in range(N):
        X, Y = sample(gen, M, t, p_jump, d_jump)
        Xs += [X]
        Ys += [Y]
    return Xs, Ys

def tsearch_max(fun, tmin, tmax, err=1e-4):
    tmid = 0.333 * (tmax - tmin) + tmin
    fmid = fun(tmid)
    its = 0
    while tmax - tmin > err:
        its += 1
        if tmax - tmid > tmid - tmin:
            tmid2 = 0.5 * (tmax + tmid)
            fmid2 = fun(tmid2)
            if fmid > fmid2:
                tmax = tmid2
            else:
                tmin = tmid
                tmid = tmid2
                fmid = fmid2
        else:
            tmid2 = 0.5 * (tmid + tmin)
            fmid2 = fun(tmid2)
            if fmid > fmid2:
                tmin = tmid2
            else:
                tmax = tmid
                tmid = tmid2
                fmid = fmid2
    return tmid, fmid, its
        
# def get_sample(gen, M, t, cnt):
#     r = np.empty(cnt)
#     for i in range(cnt):
#         r[i] = I_sample(gen, M, t, t)
    
#     return {"sum": np.sum(r),
#             "sqsum": np.sum(r**2)}

def get_sample(gen, N, M, t, p_jump, d_jump):
    Xs, Ys = sample_multi(gen, N, M, t, p_jump, d_jump)
    tmax, fmax, its = tsearch_max(lambda t : f_multi(Xs, Ys, t, p_jump, d_jump), 0, 1, 1e-4)
    if tmax >= 1 - 1e-4:
        return {"tmax" : float("inf")}
    return {"tmax" : tmax}

# resulting data frame
res = {"time" : [],
       "entropy" : []}

# path to sample file
fname = os.path.join(sample_path, "sample_{}.csv".format(np.random.SeedSequence().entropy))

if len(sys.argv) < 2:
    print("usage: {} number_of_samples (param=value )*".format(sys.argv[0]))
    exit(1)

# parse parameters
params = {}
for pv in sys.argv[2:]:
    pv = pv.split("=")
    params[pv[0]] = eval(pv[1])
    if not hasattr(params[pv[0]], '__iter__'):
        params[pv[0]] = [params[pv[0]]]
    res[pv[0]] = []

print("params: {}".format(params))

snum = int(sys.argv[1])

# safety save time interval
save_time = time.time()
save_interval = 60

# gen samples
t0_tot = time.time()
for _ in range(snum):
    
    all_params = itertools.product(*[params[k] for k in params.keys()])
    for lp in all_params:
        loc_params = {k : lp[i] for i, k in enumerate(params.keys())}
    
        # prepare new random number generator
        seed = np.random.SeedSequence()
        gen = np.random.Generator(np.random.PCG64(seed))
        #gen = np.random.Generator(np.random.MT19937(seed))

        # generate sample
        t0 = time.time()
        s = get_sample(gen, **loc_params)
        t1 = time.time()
        if type(s) != dict:
            s = {"sample": s}

        # add result to dict
        res["entropy"] += [seed.entropy]
        if len(res["time"]) == 0:
            for p in s.keys():
                res[p] = []
        for p in s.keys():
            res[p] += [s[p]]
        res["time"] += [t1 - t0]
        for p in loc_params.keys():
            res[p] += [loc_params[p]]

        # save if enough time has passed
        t = time.time()
        if t - save_time > save_interval:
            pd.DataFrame(res).to_csv(fname)
            save_time = t
            print("doing intermediate save to {}".format(fname))

t1_tot = time.time()
print("generated {} sets of samples in {}s".format(snum, t1_tot - t0_tot))

#print(res)

print("saving to {}".format(fname))
pd.DataFrame(res).to_csv(fname)

