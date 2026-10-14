import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()
plt.rcParams.update({'font.size': 14})

# Number of particles and bonds
N = 17
Nb = N - 1

zeta = 1.0
kBT = 1.0
l0 = 1.0

# From <l_i^2> = 3 kBT / K and l0^2 = <l_i^2>
K = 3.0

# Characteristic diffusion time
tau = zeta * l0 ** 2 / kBT

# Mode relaxation time
kp = np.pi * np.arange(1, N, 1) / N
taup = zeta / (4 * K * np.sin(kp / 2) ** 2)
print(taup)

# Integration timestep
dt = 0.01 * tau

# Number of integration steps
M = 100000
printevery = 100

# Initial configuration
r = np.zeros((N, 3))

# Fully stretched chain along x
r[:, 0] = np.arange(N) * l0

r_initial = r

# Store potential energy, mean bond length and normal modes
energies = []
mean_bond_lengths = []
normal_modes = []
projections = []

# Set up normal modes as a matrix
A: np.ndarray = np.zeros((Nb, Nb))

for q in range(Nb):
    for j in range(Nb):
        A[q, j] = np.sqrt(2 / N) * np.sin(np.pi * (j+1) * (q+1) / N)


def spring_forces():
    """Calculate Hookean spring forces for all 17 particles."""
    N = len(r)
    F = np.zeros_like(r)

    # First particle
    F[0] = K * (r[1] - r[0])

    # Internal particles
    F[1:-1] = K * (r[:-2] + r[2:] - 2 * r[1:-1])

    # Last particle
    F[-1] = K * (r[-2] - r[-1])

    return F


def ACF(begin, end, index=0, color='r'):
    """Plots the autocorrelation function of the mode with the given index."""
    Q = np.array(normal_modes)
    q = Q[:, index]
    acf = np.zeros(end - begin)

    iterations = np.arange(begin, end)

    for n in range(end - begin):
        i = iterations[n]
        acf[n] = np.mean(q[:len(q)-i] * q[i:]) / np.mean(q**2)

    t = iterations * printevery * dt

    plt.scatter(t, acf, c=color, marker='s', label=f'Simulated modes p={index}')
    plt.plot(t, np.exp(-t / taup[index]), c=color, label=f'Fitted modes p={index}')


for step in range(M):
    # Potential spring forces
    F = spring_forces()

    # Random Langevin forces
    sigma_R = np.sqrt(2 * zeta * kBT / dt)
    R = rng.normal(0, sigma_R, size=(N, 3))
    # finite difference update
    r += (dt / zeta) * (F + R)

    if step % printevery == 0:  # Computing the properties every step is costly and results in too many data points
        # Calculate properties
        bonds = r[1:] - r[:-1]
        bond_squared = np.sum(bonds ** 2, axis=1)

        U = 0.5 * K * np.sum(bond_squared)
        mean_bond_length = np.mean(np.sqrt(bond_squared))

        # 4a.
        u = bonds[:, 0]
        q = A @ u

        energies.append(U)
        mean_bond_lengths.append(mean_bond_length)
        normal_modes.append(q)
        projections.append(u)

r_final = r

final_bonds = r_final[1:] - r_final[:-1]
final_bond_lengths = np.linalg.norm(final_bonds, axis=1)

print("Mean final bond length:", np.mean(final_bond_lengths))
print("Maximum final bond length:", np.max(final_bond_lengths))

# Result for question 4a.
print("q^2", np.sum(normal_modes[-1] ** 2))
print("u^2", np.sum(projections[-1] ** 2))

print("\nFinal particle positions:")
print(r_final)

# Results for question 4b.
ACF(0, 15, index=0, color='r')
ACF(0, 15, index=1, color='g')
ACF(0, 15, index=2, color='b')
plt.xlabel('Time [-]')
plt.ylabel('ACF of normal modes [-]')
plt.legend()
plt.tight_layout()
plt.show()
