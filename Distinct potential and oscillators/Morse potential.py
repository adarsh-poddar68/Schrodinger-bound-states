#Morse Potential
import numpy as np
import matplotlib.pyplot as plt

r0 = 0.131349
rn = 1
n = 1000

x = np.linspace(r0, rn, n)
h = x[1] - x[0]

hbar = 197.3
m = 940e6
f = (hbar**2) / (2 * m * h**2)

e = 3.795
d = 0.755501
a = 1.44

rprime = (x - r0) / x
V = d * (np.exp(-2 * a * rprime) - np.exp(-a * rprime))

D = np.zeros((n, n))
np.fill_diagonal(D, 2 * f + V)
np.fill_diagonal(D[1:], -f)
np.fill_diagonal(D[:, 1:], -f)

E, psi = np.linalg.eigh(D)

for i in range(3):
    plt.plot(x, psi[:, i] ** 2, label=f"Energy level {i+1}: E={E[i]:.3e}")

plt.grid()
plt.xlabel('x')
plt.ylabel(r'$\Psi^2$')
plt.legend()
plt.show() 
