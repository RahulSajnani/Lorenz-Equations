import numpy as np
import matplotlib.pyplot as plt

n=-0.7
m=-1.2
g=[]
t = np.linspace(-5, 5, 100)

for i in range(t.shape[0]):
    g.append(n*t[i]+0.5*(m-n)*(np.abs(t[i]+1)-np.abs(t[i]-1)))

minor_ticks = np.arange(0, t[-1]//1, 1)
g=np.array(g)

fig, axs = plt.subplots(1, 1, constrained_layout=True)

axs.plot(t, g, "b")
axs.set_title("Change in resistance vs. Current across the Chua diode")
axs.set_xlabel("x")
axs.set_ylabel("g(x)")
axs.set_xticks(minor_ticks, minor = True) 
axs.grid(which='minor', alpha=0.2)
axs.grid(True)
plt.show()