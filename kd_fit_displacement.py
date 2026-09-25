"""
Fit the Katsuragi-Durian granular drag model to displacement-vs-time data.

Model (z measured downward, positive into the medium):
    m * z'' = -m*g + m*(z')**2 / d1 + k*|z| - f0

State vector y = [z, v] where v = z'.

Free parameters fit to data: d1, k, f0 (f0 optional, set to 0 to disable).
d1, k are the Katsuragi-Durian material parameters (length and spring-constant
units respectively). f0 is an optional constant residual/yield force that lets
the model asymptote to a nonzero drag rather than fully stopping.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

# ---- known/measured constants (edit these) ----
m = 0.045          # projectile mass [kg]
g = 9.81          # gravity [m/s^2]
v0 = 3.0          # impact velocity at z=0 [m/s] (from your free-fall estimate)

# ---- measured data exported from master_notes.ipynb ----
data_path = Path(__file__).resolve().parent / "Data" / "kd_fit_displacement_data.csv"
data = pd.read_csv(data_path)
t_data = data["time_s"].to_numpy(dtype=float)
z_data = data["displacement_m"].to_numpy(dtype=float)


def rhs(t, y, d1, k, f0):
    z, v = y
    dzdt = v
    dvdt = -g + (v**2) / d1 + (k / m) * np.abs(z) - f0 / m
    return [dzdt, dvdt]


def simulate(params, t_eval):
    d1, k, f0 = params
    sol = solve_ivp(
        rhs, (t_eval[0], t_eval[-1]), y0=[0.0, v0],
        t_eval=t_eval, args=(d1, k, f0),
        method="RK45", rtol=1e-8, atol=1e-10,
    )
    if not sol.success or len(sol.y[0]) != len(t_eval):
        # integration blew up / failed for this parameter guess -- return a
        # large penalty vector so least_squares steers away from this region
        return np.full_like(t_eval, 1e6)
    return sol.y[0]  # z(t)


def residuals(params, t_data, z_data):
    z_model = simulate(params, t_data)
    return z_model - z_data


# initial guesses: [d1, k, f0] -- tune these based on rough magnitude of your data
# NOTE: pick p0 so the simulated trajectory doesn't blow up (v^2/d1 term can runaway
# if d1 is too small relative to k/m); check with a forward simulate() call first.
p0 = [0.0002, 0.0001, 0.0]

result = least_squares(
    residuals, p0, args=(t_data, z_data),
    bounds=([1e-6, 0, 0], [np.inf, np.inf, np.inf]),  # physical: d1>0, k>=0, f0>=0
)

d1_fit, k_fit, f0_fit = result.x
print(f"Fitted d1 = {d1_fit:.4g} m")
print(f"Fitted k  = {k_fit:.4g} N/m")
print(f"Fitted f0 = {f0_fit:.4g} N")

z_fit = simulate(result.x, t_data)

plt.figure(figsize=(6, 4))
plt.plot(t_data, z_data, 'o', label='data', ms=4)
plt.plot(t_data, z_fit, '-', label='Katsuragi-Durian fit')
plt.xlabel('time [s]')
plt.ylabel('depth z [m]')
plt.legend()
plt.tight_layout()
output_path = Path(__file__).resolve().parent / "kd_fit_result.png"
plt.savefig(output_path, dpi=150)
print(f"Saved plot to {output_path}")
