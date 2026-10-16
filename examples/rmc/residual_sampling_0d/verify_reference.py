import numpy as np
from rmc0d import *
S = Setup(30, 1, "flat", 400)
print("fine nodes", S.Ef.size)
F = reference_collision_density(S.Ef, S.q)
phi = S.moments(F / S.St)[:, 0] / np.diff(S.edges)
S2 = Setup(30, 1, "flat", 1600)
F2 = reference_collision_density(S2.Ef, S2.q)
phi2 = S2.moments(F2 / S2.St)[:, 0] / np.diff(S2.edges)
print("ref grid convergence max rel", np.max(np.abs(phi/phi2-1)))
N=2_000_000; rng=np.random.default_rng(0)
E0 = S.edges[-2] + rng.random(N)*(S.edges[-1]-S.edges[-2])
t = transport(E0, np.full(N,1.0/N), S.edges, 1, 3)
smc = t[:,0]/np.diff(S.edges)
print("SMC vs ref rel:", np.round(smc/phi-1, 4))
# consistency: T of exact-ish psi vs F - q
