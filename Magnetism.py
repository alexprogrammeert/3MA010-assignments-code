import numpy as np
import matplotlib.pyplot as plt

N: int = 64

DIRS: np.ndarray = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
S: np.ndarray = np.ones((N, N))
J: float = 1.
H: float = 1.
mu: float = 1.
kB: float = 1.
T: float = 1.


def GetDeltaEnergy(i: int, j: int) -> float:
    """Computes the change in energy of altering a single spin state in S at location (x, y)."""
    SC: float = 0.

    for k in DIRS:
        di: int = i + k[0]
        dj: int = j + k[1]

        # Periodic boundaries
        if di >= N:
            di -= N
        if dj >= N:
            dj -= N

            SC += S[i, j] * S[di, dj]

    return 2 * J * SC + 2 * mu * H * S[i, j]


def GetTotalEnergy() -> float:
    """Computes the total energy of a grid of spin states S according to the ising model."""
    energy: float = 0.

    for i in range(N):
        for j in range(N):
            SC: float = 0.

            for k in DIRS:
                di: int = i + k[0]
                dj: int = j + k[1]

                # Periodic boundaries
                if di >= N:
                    di -= N
                if dj >= N:
                    dj -= N

                SC += S[i, j] * S[di, dj]

            energy += -0.5 * J * SC - mu * H * S[i, j]

    return energy


def GetMagnetisation() -> float:
    """Computes the total magnetisation."""
    return mu * S.sum()


def FlipState(i, j) -> None:
    """Flips the spin state at a randomized site."""
    S[i, j] = -1 * S[i, j]


def PlotState() -> None:
    """Plots the spin states of the magnetic substance."""
    plt.imshow(S)
    plt.show()


def Main(tMax: int) -> None:
    """Main loop for the metropolis algorithm."""
    for t in range(tMax):
        x, y = np.random.randint(0, N), np.random.randint(0, N)
        dE: float = GetDeltaEnergy(x, y)

        a: float = np.random.rand()
        w: float = min(1, np.exp(-dE / (kB * T)))

        if a <= w:
            FlipState(x, y)
        else:
            continue


Main(5_000)
PlotState()

# Exercise C.
J = 0.
Q: int = 8  # number of measurements
hArray: np.ndarray = np.linspace(0.1, 3.0, Q)
mArray: np.ndarray = np.zeros(Q)

for i in range(Q):
    H: float = hArray[i]

    Main(10_000)
    mArray[i] = GetMagnetisation()

mFit: np.ndarray = mu * N * N * np.tanh(mu * hArray / (kB * T))

plt.rcParams.update({'font.size': 14})
plt.scatter(hArray, mArray, c='k', marker='s', label='simulated')
plt.plot(hArray, mFit, c='r', label='fitted')
plt.xlabel('Magnetic field strength [a.u.]')
plt.ylabel('Magnetisation [a.u.]')
plt.legend()
plt.tight_layout()
plt.show()
