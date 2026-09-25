"""
Plot displacement vs time for the Katsuragi-Durian granular drag model,
given known/chosen parameters (no fitting -- pure forward simulation).

Model (z measured downward, positive into the medium):
    m * z'' = -m*g + m*(z')**2 / d1 + k*|z| - f0
"""

import numpy as np
import matplotlib
matplotlib.use("QtAgg")
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ---- parameters (edit these) ----
m = 0.05        # projectile mass [kg]
g = 9.81        # gravity [m/s^2]
v0 = 3.0        # impact velocity at z=0 [m/s]
d1 = 8.       # inertial drag lengthscale [m]
k = 10.3        # frictional drag constant [N/m]
f0 = 0.0        # optional constant residual/yield force [N]

t_max = 1.0     # how long to simulate [s]
n_points = 500  # time resolution


def rhs(t, y):
    z, v = y
    dzdt = v
    dvdt = g - (v**2) / d1 - (k / m) * np.abs(z) - f0 / m
    return [dzdt, dvdt]


def stopped(t, y):
    return y[1]  # triggers when v = 0


stopped.terminal = True
stopped.direction = -1  # only trigger when v is decreasing through zero

t_eval = np.linspace(0, t_max, n_points)
sol = solve_ivp(
    rhs, (0, t_max), y0=[0.0, v0],
    t_eval=t_eval, method="RK45", rtol=1e-8, atol=1e-10,
    events=stopped,
)

z = sol.y[0]
v = sol.y[1]

fig, axes = plt.subplots(2, 1, figsize=(6, 7), sharex=True)

axes[0].plot(sol.t, z)
axes[0].set_ylabel('depth z [m]')
axes[0].set_title('Katsuragi-Durian model: displacement vs time')
axes[0].grid(alpha=0.3)

axes[1].plot(sol.t, v, color='tab:orange')
axes[1].set_ylabel('velocity v [m/s]')
axes[1].set_xlabel('time [s]')
axes[1].grid(alpha=0.3)
axes[1].set_xlim(0, t_max)

plt.tight_layout()
plt.show()
print(f"Final depth reached: {z[-1]:.4f} m")
