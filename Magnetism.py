import numpy as np
import matplotlib.pyplot as plt

N: int = 32
tMax: int = 50_000

DIRS: np.ndarray = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
S: np.ndarray = np.ones((N, N))
J: float = 5.
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

    return -0.5 * SC - mu * H * S[i, j]


def GetTotalEnergy() -> float:
    """Computes the total energy of a grid of spin states S according to the ising model."""
    energy: float = 0.

    for i in range(N):
        for j in range(N):
            energy += GetDeltaEnergy(i, j)

    return energy


def FlipState(i, j) -> None:
    """Flips the spin state at a randomized site."""
    S[i, j] = -1 * S[i, j]


def PlotState() -> None:
    """Plots the spin states of the magnetic substance."""
    plt.imshow(S)
    plt.show()


E: float = GetTotalEnergy()
print(f"start energy: {E}")

for t in range(tMax):
    x, y = np.random.randint(0, N), np.random.randint(0, N)
    FlipState(x, y)
    dE: float = GetDeltaEnergy(x, y)

    a: float = np.random.rand()
    w: float = min(1, np.exp(-dE / (kB * T)))

    if a <= w:
        continue
    else:
        FlipState(x, y)
        print(t)

E: float = GetTotalEnergy()
print(f"final energy: {E}")

PlotState()
