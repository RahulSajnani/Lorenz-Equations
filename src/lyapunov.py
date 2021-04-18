import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(".")
from lorenz import Lorenz_equations

lorenz1 = Lorenz_equations(prandtl_number = 10, rayleigh_number = 25, beta = 8/3, delta_t = 1e-2)
trajectory1 = lorenz1.getLorenzTrajectory([1,3,5])
ep = 1e-9
lorenz2 = Lorenz_equations(prandtl_number = 10, rayleigh_number = 25, beta = 8/3, delta_t = 1e-2)
trajectory2 = lorenz2.getLorenzTrajectory([1,3,5+ep])
#d = trajectory1[:,0]-1
d = (abs(trajectory1[:,0,None]-trajectory2[:,0,None]))**2 + (abs(trajectory1[:,1,None]-trajectory2[:,1,None]))**2 + (abs(trajectory1[:,2,None]-trajectory2[:,2,None]))**2
d = np.sqrt(d)
t = np.arange(0,d.shape[0],1)
plt.semilogy(t * 1e-2 ,d) 
plt.title('Log of magnitude of separation of nearby Lorenz trajectories', fontsize = 12)
plt.xlabel('time (sec)', fontsize = 10) 
plt.ylabel("Distance in log scale")
plt.grid(True)
plt.savefig('lyapunov.png')
plt.show()