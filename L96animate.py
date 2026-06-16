"""Lorenz 1996 model animation (with zonally varying damping)
Lorenz E., 1996. Predictability: a problem partly solved. In
Predictability. Proc 1995. ECMWF Seminar, 1-18."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from L96 import L96

np.random.seed(42)                               # fix random seed for reproducibility

nspinup = 1000                                   # time steps to spin up truth run
nmax = 10000                                     # number of ob times.

N = 80
F = 8
deltaF = 1./8.
Fcorr = np.exp(-1)**(1./3.) 					 # efolding over n timesteps, n=3
#model = L96(N=80,F=F,deltaF=deltaF,Fcorr=Fcorr) # model instance for truth run
model = L96(N=N,F=F) 							 # model instance for truth run

for nt in range(nspinup):                        # spinup truth run
    model.fcst()

uu = []
tt = []
x = np.arange(N)

#----------------------------------------------------------------------------------
# 1. plot the values of 80 variables for eash timestep using matplotlib.animation
#----------------------------------------------------------------------------------
# references:
# - https://matplotlib.org/api/_as_gen/matplotlib.animation.FuncAnimation.html
# - https://pinkwink.kr/860 
fig, ax = plt.subplots()
line, = ax.plot(x, model.x.squeeze())           # squeeze(): remove one-dimensional entries from the shape of an array
ax.set_xlim(0,N-1)
#ax.set_ylim(3,3)
#Init only required for blitting to give a clean slate.
def init():
    global line
    line.set_ydata(np.ma.array(x, mask=True))   # generate data arrays with masks (When an element of the mask is True, the corresponding element of the associated array is said to be masked (invalid))
    return line,

def updatefig(n):
    global tt,uu
    model.fcst()
    u = model.x.squeeze()
    line.set_ydata(u)
    print("{} {} {}".format(n,u.min(),u.max()))
    uu.append(u)
    tt.append(n*model.dt)
    return line,

#Writer = animation.writers['ffmpeg']
#writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
ani = animation.FuncAnimation(fig, func=updatefig, frames=np.arange(0,nmax+1), init_func=init,
                              interval=25, blit=True, repeat=False)
#ani.save('KS.mp4',writer=writer)
plt.show()

#----------------------------------------------------------------------------------
# 2. plot time-longitude snapshot (hovmuller diagram) of L96 model
#----------------------------------------------------------------------------------
plt.figure()
# make contour plot of solution, plot spectrum.
uu = np.array(uu)
tt = np.array(tt)
print("{} {}".format(tt.min(), tt.max()))
print("{}".format(uu.shape))
nplt = 500
print("{} {}".format(uu[:nplt].min(), uu[:nplt].max()))
#plt.contourf(x,tt[:nplt],uu[:nplt],np.linspace(-18,18,41),cmap=plt.cm.bwr,extend='both')
plt.contourf(x,tt[:nplt],uu[:nplt],np.linspace(-18,18,41),cmap=plt.cm.bwr)
plt.xlabel('$\mathrm{x}$')
plt.ylabel('t')
plt.colorbar()
plt.title('time-longitude snapshot (hovmuller diagram) of L96 model')
plt.savefig('hovmuller.png')

#----------------------------------------------------------------------------------
# 3. plot climatological covariance matrix for 96 model
#----------------------------------------------------------------------------------
plt.figure()
ncount = len(uu)
uup = uu - uu.mean(axis=0)
print("{}".format(uup.shape))
cov = np.dot(uup.T,uup)/(ncount-1)
print("{} {} {} {}".format('cov',cov.min(), cov.max(), cov.shape))
plt.pcolormesh(x,x,cov,cmap=plt.cm.bwr,vmin=-30,vmax=30)
plt.title('climatological covariance matrix for L96 model')
plt.xlabel('$\mathrm{x}$')
plt.ylabel('$\mathrm{x}$')
plt.colorbar()
plt.savefig('covmat.png')

plt.show()
