import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
})

# Parameters from Pang et al. (Table III)
alpha = 4.5
G_N = 5e3       # s^-1
kappa = 1.2136e11 # s^-1
T_S = 0.8059e-9  # s
T_P = 1.1628e-12 # s
P_0 = 3.5183e5
nu = 2 * np.pi * 251e6 # rad/s

def pang_derivatives(t, y, E_x):
    # y[0] = Re(E), y[1] = Im(E), y[2] = Delta N
    E = y[0] + 1j * y[1]
    dN = y[2]
    
    # Eq (1)
    dE_dt = 0.5 * (1 + 1j * alpha) * G_N * dN * E + kappa * E_x * np.exp(1j * nu * t)
    
    # Eq (2)
    P = np.abs(E)**2
    ddN_dt = -(1/T_S + G_N * P_0) * dN - (1/T_P + G_N * dN) * (P - P_0)
    
    return [dE_dt.real, dE_dt.imag, ddN_dt]

E_x_vals = [0.3, 1.2, 1.7]
labels = ['(a) E_x = 0.3 (Nonlinear/Chaotic)', '(b) E_x = 1.2 (Periodic)', '(c) E_x = 1.7 (Stable Locking)']

t_span = (0, 100e-9)
t_eval = np.linspace(0, 100e-9, 10000)

plt.figure(figsize=(10, 8))

for i, E_x in enumerate(E_x_vals):
    y0 = [np.sqrt(P_0), 0.0, 0.0] # E = sqrt(P_0), dN = 0
    print(f"Simulating E_x = {E_x}...")
    sol = solve_ivp(pang_derivatives, t_span, y0, args=(E_x,), t_eval=t_eval, method='BDF', rtol=1e-5, atol=1e-8)
    
    E_complex = sol.y[0] + 1j * sol.y[1]
    amplitude = np.abs(E_complex)
    
    plt.subplot(3, 1, i+1)
    # Plot last 50 ns to show steady state/attractor
    mask = sol.t >= 50e-9
    plt.plot(sol.t[mask]*1e9, amplitude[mask], color='tab:blue')
    plt.title(labels[i])
    plt.ylabel('|E(t)|')
    if i == 2:
        plt.xlabel('Time (ns)')

plt.tight_layout()
plt.savefig('figures/pang_verification.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.close()
print("Verification complete. Plot saved to figures/pang_verification.pdf")
