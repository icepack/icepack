
import os
import numpy as np
import matplotlib.pyplot as plt

from icepack import ucd

def A(T):
    year_in_sec = 365.25 * 24 * 3600
    ideal_gas = 8.3144621e-3

    transition_temperature = 263.215
    A0_cold = 3.985e-13 * year_in_sec * 1.0e18
    A0_warm = 1.916e3 * year_in_sec * 1.0e18
    Q_cold = 60
    Q_warm = 139

    A0 = A0_cold
    Q = Q_cold

    if T > transition_temperature:
        A0 = A0_warm
        Q = Q_warm

    return A0 * np.exp(-Q / (ideal_gas * T))


def B(T):
    return A(T)**(-1.0/3)


def plot_field(x, y, q, qmin, qmax, fig, subplot, title = ""):
    ax = fig.add_subplot(subplot)
    ax.set_aspect('equal')

    ctr = ax.tricontourf(x/1000.0, y/1000.0, q,
                         np.linspace(qmin, qmax, 45), cmap = 'viridis',
                         shading = 'faceted', extend = 'both')

    ax.set_xlabel('x (km)')
    ax.set_ylabel('y (km)')
    ax.set_title(title)

    cbar = fig.colorbar(ctr, fraction = 0.046, pad = 0.04)

    return


if __name__ == "__main__":
    runs = ["basic", "noisy", "regularized", "boxy-sg", "boxy-tv"]

    for name in runs:
        x, y, cell, theta_true = ucd.read(name + "/theta_true.ucd")
        _, _, _, vx_true, vy_true = ucd.read(name + "/u_true.ucd")
        bmin, bmax = B(np.max(theta_true)), B(np.min(theta_true))
        v_true = np.sqrt(vx_true**2 + vy_true**2)
        vmin, vmax = np.min(v_true), np.max(v_true)

        _, _, _, theta = ucd.read(name + "/theta.ucd")

        n = len(x)
        b, b_true = np.zeros(n), np.zeros(n)
        for i in range(n):
            b[i] = B(theta[i])
            b_true[i] = B(theta_true[i])

        _, _, _, vx, vy = ucd.read(name + "/u.ucd")
        v = np.sqrt(vx**2 + vy**2)

        # Plot the exact rheology and velocity
        fig = plt.figure()

        plot_field(x, y, b_true, bmin, bmax, fig, 121,
                   title = "Rheology (MPa $\cdot$ year${}^{1/3}$)")
        plot_field(x, y, v_true, vmin, vmax, fig, 122,
                   title = "Speed (m/a)")

        plt.tight_layout()
        plt.savefig(name + "/result_exact.png",
                    bbox_inches = "tight")

        # Plot the inferred rheology and velocity
        fig = plt.figure()

        plot_field(x, y, b, bmin, bmax, fig, 121,
                   title = "Rheology (MPa $\cdot$ year${}^{1/3}$)")
        plot_field(x, y, v, vmin, vmax, fig, 122,
                   title = "Speed (m/a)")

        plt.tight_layout()
        plt.savefig(name + "/result_inferred.png",
                    bbox_inches = "tight")
