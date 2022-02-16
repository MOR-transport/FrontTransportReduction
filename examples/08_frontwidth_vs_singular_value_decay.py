import globs
import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
import tikzplotlib
import sys
from plot_utils import save_fig

imagepath="../imgs/"


x = np.linspace(0,1,1000)
dx = x[1]-x[0]
Nt = 100
dt = 1/(Nt)
t = np.linspace(0,1-dt,Nt-1)
c = 1
Xgrid,Tgrid = np.meshgrid(x,t)
front = lambda x,l: 0.5+0.5*np.tanh(x/l)

l = 1
decay_const = []
width_list = []
for n in range(20):
    data = front(Xgrid-c*Tgrid,l)

    [U,S,VT] = np.linalg.svd(data,full_matrices=False)
    rank = np.sum(S > 1e-12)
    #plt.semilogy(S/S[0],'-*', label=r"$l=10^{-"+str(n)+"}L$", color=[0,0,0])
    p = np.polyfit(np.arange(rank), np.log(S[:rank]/S[0]), 1)#,  w=np.sqrt((S[:rank]/S[0])))
    decay_const.append(np.abs(p[0]))
    print(p)
    width_list.append(l)
    l = l/2
    if l<5*dx:
        break

fig, ax = plt.subplots(1)
ax.loglog(width_list,decay_const,'*', label=r"$\beta$")

ax.set_xlabel(r"$l_f/L$")
ax.set_ylabel(r"decay rate $\beta$")

save_fig(imagepath+"decay_rate.tex")
