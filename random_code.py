import matplotlib.pyplot as plt
import numpy as np

def schrodinger_sol_analytical(x, t, a, k0):
    return 1/(2*np.pi)*((1/(2*a*np.pi))**(1/4))*np.exp(k0/4*a)*(1/np.sqrt((1/2*a)+1j*t))*(np.exp((-(x-((1j*k0)/(2*a)))**2)/((1/a)+2*1j*t)))


x = np.arange(-10, 10, 0.1)

#print(schrodinger_sol_analytical(x, 0, 2, 2))
plt.figure(figsize = [8,6])
ax1 = plt.subplot(1,1,1)
ax1.plot(x, np.abs(schrodinger_sol_analytical(x, 0, 1, 1))**2, label = "Analytical solution")
ax1.grid()
ax1.set_ylim(0, 0.06)
ax1.legend()
ax1.plot()