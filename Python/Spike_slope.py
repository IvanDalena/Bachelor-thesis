import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Stile da articolo
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 12,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
})


# ============================================================
# Costanti fisiche e parametri del problema
# ============================================================

r_sun = 8330.0               # pc
rho_sun = 0.4                # GeV / cm^3

# Buco nero centrale e spike
G_pc = 4.30091e-3            # pc (km/s)^2 / M_sun
c_kms = 299792.458           # km/s

M_BH = 4.15e6                # M_sun
v0 = 105.0                   # km/s

R_h = G_pc * M_BH / v0**2            # pc
R_sp = 0.2 * R_h                     # pc
R_S = 2.0 * G_pc * M_BH / c_kms**2   # pc

# Densità di saturazione: rho_core = m_DM / (<sigma v> t_BH)
m_DM = 1.0e3                 # GeV
sigmav = 2.2e-26             # cm^3/s
t_BH = 1.0e10 * 3.1557e7     # s
rho_core = m_DM / (sigmav * t_BH)    # GeV/cm^3

r_c = 1000.0                 # pc, core del profilo isotermo


# ============================================================
# Profili di densità
# ============================================================

def rho_halo(r, gamma):
    """
    Alone a legge di potenza, normalizzato a rho_sun in r_sun.
    r in pc, densità in GeV/cm^3.
    """

    return rho_sun * (r / r_sun)**(-gamma)


def rho_halo_iso(r):
    """
    Alone isotermo con core r_c, normalizzato a rho_sun in r_sun.
    """

    return rho_sun * (1.0 + (r_sun / r_c)**2) / (1.0 + (r / r_c)**2)


def density_profile(r, gamma):
    """
    Profilo completo alone + spike:

    rho = 0                             per r < 4 R_S
    rho = spike saturata                per 4 R_S <= r < R_sp
    rho = alone                         per r >= R_sp

    La spike include il fattore di cattura g = (1 - 4 R_S / r)^3
    e la saturazione a rho_core.

    gamma = None indica il profilo isotermo, per cui gamma_sp = 3/2.
    """

    r = np.asarray(r, dtype=float)
    x = np.maximum(r, 1e-300)

    if gamma is None:
        halo = rho_halo_iso(r)
        rho_R = float(rho_halo_iso(R_sp))
        gamma_sp = 1.5
    else:
        halo = rho_halo(x, gamma)
        rho_R = float(rho_halo(R_sp, gamma))
        gamma_sp = (9.0 - 2.0 * gamma) / (4.0 - gamma)

    g = np.clip(1.0 - 4.0 * R_S / x, 0.0, None)**3

    rho_prime = rho_R * g * (x / R_sp)**(-gamma_sp)
    spike = rho_prime * rho_core / (rho_prime + rho_core)

    out = np.where(r < R_sp, spike, halo)

    return np.where(r < 4.0 * R_S, 0.0, out)


# ============================================================
# Figura
# ============================================================

r = np.logspace(-7.0, 5.0, 4000)   # pc

curves = [
    (1.3,  r"$\gamma = 1.3$",    "#0072B2"),
    (1.0,  r"$\gamma = 1$",      "#D55E00"),
    (0.1,  r"$\gamma = 0.1$",    "#009E73"),
    (0.01, r"$\gamma = 0.01$",   "#CC79A7"),
    (None, "isothermal (cored)", "#E69F00"),
]

fig, ax = plt.subplots(figsize=(8.0, 5.5))

for gamma, label, color in curves:
    rho = density_profile(r, gamma)
    ax.loglog(r, np.where(rho > 0.0, rho, np.nan),
              label=label, color=color, lw=2.0)

# Densità di saturazione e raggio della spike
ax.axhline(rho_core, color="0.55", ls=":", lw=1.0)
ax.text(2.5e-7, 3.2e11, r"$\rho_{\mathrm{core}}$", color="0.35", fontsize=11)

ax.axvline(R_sp, color="0.55", ls="--", lw=1.0)
ax.text(1.25 * R_sp, 2.2e-3, r"$R_{\mathrm{sp}}$", color="0.35", fontsize=11)

ax.axvline(4.0 * R_S, color="0.55", ls="--", lw=1.0)
ax.text(1.3 * 4.0 * R_S, 2.2e-3, r"$4R_S$", color="0.35", fontsize=11)

ax.set_xlim(1e-7, 1e5)
ax.set_ylim(1e-3, 1e13)
ax.set_xlabel(r"$r\ \,[\mathrm{pc}]$", fontsize=12)
ax.set_ylabel(r"$\rho\ \,[\mathrm{GeV\,cm^{-3}}]$", fontsize=12)

ax.legend(frameon=False, fontsize=11, loc="upper right")

fig.tight_layout()

output_path = (
    Path(__file__).resolve().parent.parent / "Latex" / "images" / "Spike_slope.png"
)
fig.savefig(output_path, dpi=300)

print(f"Figura salvata in {output_path}")
print(f"rho_core = {rho_core:.3e} GeV/cm^3")
