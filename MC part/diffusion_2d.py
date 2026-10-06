import numpy as np
import matplotlib.pyplot as plt

r: float = 1.


def GetTrajectory(N: int, A: float = 1.0) -> np.ndarray:
    """Retrieves a list of vectors indicating the points traveled in the bacterias' trajectory."""
    trajectory: np.ndarray = np.zeros((N, 2))
    trajectory[0] = np.zeros(2)

    for n in range(1, N):
        phi = GetAngleReject(A)
        trajectory[n] = GetVector(phi) + trajectory[n-1]

    return trajectory


def GetVector(phi: float) -> np.ndarray:
    """Uses the angle phi to compute the vector of the trajectory step."""
    return np.array([r * np.cos(phi), r * np.sin(phi)])


def PDF(phi: float, A: float) -> float:
    """The probability density distribution of the bacterias' forward motion."""
    return 1. / (2. * np.pi) * (1. + A * np.cos(phi))


def GetAngleReject(A: float) -> float:
    """Rejection sampling of the PDF to acquire the angle of the motion step."""
    while True:
        phi: float = 2. * np.pi * np.random.rand()
        y: float = A * np.random.rand()

        if y <= PDF(phi, A):
            break

    return phi


def ExpectedDist(A: float, N: int | np.ndarray) -> float | np.ndarray:
    """Computes the expected distance the bacteria travels along the x-direction."""
    return 0.5 * N * A * r


d: int = 10
kArray: np.ndarray = np.logspace(2, 4, num=d, dtype=int)
aArray: np.ndarray = np.array([1.0, 0.5, 0.2])
xArray: np.ndarray = np.zeros((d, 3))
fitArray: np.ndarray = np.array([kArray[0], kArray[-1]])

# Vary both A and N
for i in range(3):
    for j in range(d):
        a: int = aArray[i]
        k: int = kArray[j]

        traj: np.ndarray = GetTrajectory(k, a)
        xArray[j, i] = traj[-1, 0]

plt.rcParams.update({'font.size': 14})

plt.scatter(kArray, xArray[:, 0], c="r", marker="s", label=f"simulated; A={aArray[0]}")
plt.plot(fitArray, ExpectedDist(aArray[0], fitArray), c="r", label=f"fitted; A={aArray[0]}")

plt.scatter(kArray, xArray[:, 1], c="g", marker="s", label=f"simulated; A={aArray[1]}")
plt.plot(fitArray, ExpectedDist(aArray[1], fitArray), c="g", label=f"fitted; A={aArray[1]}")

plt.scatter(kArray, xArray[:, 2], c="b", marker="s", label=f"simulated; A={aArray[1]}")
plt.plot(fitArray, ExpectedDist(aArray[2], fitArray), c="b", label=f"fitted; A={aArray[2]}")

plt.xlabel("Steps [-]")
plt.ylabel("Distance [a.u.]")
plt.legend()
plt.tight_layout()
plt.show()

