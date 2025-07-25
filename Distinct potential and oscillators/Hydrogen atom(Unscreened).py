#solving the schrodinger equation for a hydrogen atom(realistic case) unscreened coulamb potential
import numpy as np
import matplotlib.pyplot as plt

# Parameters
r0 = 0.01     # Initial position
rn = 10       # Final position
n = 1000      # Number of steps

x = np.linspace(r0, rn, n - 1)
delx = x[1] - x[0]

hbar = 1.973
m = 0.511e6  # Electron mass in eV/c^2
k = (hbar**2) / (2 * m * delx**2)

e = 1.6e-19
v = 3.795

V = np.zeros((n - 1, n - 1))
for i in range(n - 1):
    V[i, i] = (e**2 * v) / x[i]

# Matrix construction
diagonals = np.zeros((n - 1, n - 1))
np.fill_diagonal(diagonals, 2 / delx**2)
np.fill_diagonal(diagonals[1:], -1 / delx**2)
np.fill_diagonal(diagonals[:, 1:], -1 / delx**2)

# Hamiltonian matrix
D = diagonals + (2 * m * V / hbar**2)

# Solve eigenvalue problem
E_ev, psi = np.linalg.eigh(D)
E_ev = (E_ev * hbar**2) / (2 * m)

# Plot the first few eigenfunctions
plt.figure(figsize=(15, 6))
for i in range(3):  # Plot first 3 energy levels
    plt.plot(x, psi[:, i]**2, label=f'Energy level {i+1}: E_ev = {E_ev[i]:.3e}')
plt.axhline(y=0, color='black')
plt.xlabel('position')
plt.ylabel(r'$\Psi^2$', size=15)
plt.title('Probability density for Different Energy Levels')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(15, 6))
for i in range(3):  # Plot first 3 energy levels
    plt.plot(x, psi[:, i], label=f'Energy level {i+1}: E_ev = {E_ev[i]:.3e}')
plt.axhline(y=0, color='black')
plt.xlabel('position')
plt.ylabel(r'$\Psi$', size=15)
plt.title('Wavefunction for Different Energy Levels')
plt.legend()
plt.grid()
plt.show()
