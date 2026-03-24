import numpy as np
from plant import Plant
import pdb
import json
import os

nsim = 100
k = 0.1
T = 2
Ts = 0.1

P = Plant(k,T,Ts)
u_vec = []
y_vec = []
for k in range(0,nsim):
    u = np.random.normal(0.0,0.2)
    #u = 1
    P.update(u)
    u_vec.append(u)
    y_vec.append(P.output())

dat = {"u":u_vec,"y":y_vec}
f = open("rand_sim.json","w")
f.write(json.dumps(dat))
f.close()
    