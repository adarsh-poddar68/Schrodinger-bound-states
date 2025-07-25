# Anharmonic oscillator
import numpy as np
import matplotlib.pyplot as plt

# Parameters
r0 = -0.7  # initial position
rn = 0.7   # final position
n = 1000   # Number of steps

x = np.linspace(r0, rn, n - 1)
delx = x[1] - x[0]

# Constants
hbar = 197.3  # MeV·fm
m = 940       # MeV/c^2
k = 100       # MeV/fm^2
b = 10        # MeV/fm^3

# Potential
V = np.zeros((n - 1, n - 1))
for i in range(n - 1):
    V[i, i] = (0.5 * k * x[i] ** 2) + (b * x[i] ** 3 / 3)

# Matrix construction
diagonals = np.zeros((n - 1, n - 1))
np.fill_diagonal(diagonals, 2 / delx**2)
np.fill_diagonal(diagonals[1:], -1 / delx**2)
np.fill_diagonal(diagonals[:, 1:], -1 / delx**2)

# Hamiltonian matrix
H = diagonals + (2 * m * V / hbar**2)
print("Hamiltonian matrix:\n", H[:4, :4])

# Solve eigenvalue problem
E_ev, psi = np.linalg.eigh(H)
E_ev = (E_ev * hbar**2) / (2 * m)

# Plot the first few eigenfunctions
plt.figure(figsize=(15, 6))
for i in range(3):  # Plot first 3 energy levels
    plt.plot(x, psi[:, i], label=f"Energy level {i+1}: E_ev = {E_ev[i]:.3f}")
plt.axhline(y=0, color="black")
plt.xlabel("x")
plt.ylabel("Psi")
plt.title("Wave Functions for Different Energy Levels")
plt.legend()
plt.grid()
plt.show()
