import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.integrate import quad
from scipy.optimize import root

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
# Costanti fisiche e condizioni del fit (come J-factor.py)
# ============================================================

r_sun = 8.33                 # kpc
rho_sun = 0.4                # GeV / cm^3

R_mass = 60.0                # kpc
M_target = 4.7e11            # M_sun

kpc_to_cm = 3.0856775814913673e21
M_sun_g = 1.98847e33
GeV_g = 1.78266192e-24

mass_conversion = kpc_to_cm**3 * GeV_g / M_sun_g


# ============================================================
# Profili di densita' adimensionali (forme funzionali PPPC)
#
# rho(r) = rho_s * shape(r / r_s)
# ============================================================

def density_shape(profile, x):
    """
    Parte adimensionale del profilo, x = r / r_s.
    """

    x = np.maximum(x, 1e-300)

    if profile == "nfw":
        return 1.0 / (x * (1.0 + x)**2)

    elif profile == "moore":
        return x**(-1.16) * (1.0 + x)**(-1.84)

    elif profile == "einasto":
        return np.exp(-(2.0 / 0.17) * (x**0.17 - 1.0))

    elif profile == "einastoB":
        return np.exp(-(2.0 / 0.11) * (x**0.11 - 1.0))

    elif profile == "iso":
        return 1.0 / (1.0 + x**2)

    elif profile == "burkert":
        return 1.0 / ((1.0 + x) * (1.0 + x**2))

    else:
        raise ValueError("Profilo non riconosciuto.")


def density(profile, r, rho_s, rs):
    return rho_s * density_shape(profile, r / rs)


# ============================================================
# Fit di rho_s e r_s con le due condizioni:
# rho(r_sun) = rho_sun e M(<60 kpc) = M_target
# ============================================================

def mass_enclosed(profile, R, rho_s, rs):

    def integrand(r):
        return 4.0 * np.pi * r**2 * float(density(profile, r, rho_s, rs))

    integral, error = quad(integrand, 0.0, R, epsrel=1e-8, limit=300)

    return integral * mass_conversion


def solve_rhos_rs(profile):

    def equations(log_variables):
        log_rhos, log_rs = log_variables

        rho_s = 10.0**log_rhos
        rs = 10.0**log_rs

        eq_density = np.log10(density(profile, r_sun, rho_s, rs) / rho_sun)
        eq_mass = np.log10(mass_enclosed(profile, R_mass, rho_s, rs) / M_target)

        return [eq_density, eq_mass]

    # Per l'isotermo il sistema non ha soluzione: con
    # rho(r_sun) = 0.4 GeV/cm^3 la coda 1/r^2 implica
    # M(<60 kpc) >= 5.4e11 M_sun per qualunque r_s, sempre
    # sopra M_target. Si usa quindi il raggio di scala di
    # PPPC4DMID (r_s = 4.38 kpc), normalizzato alla densita'
    # locale.
    if profile == "iso":
        rs = 4.38
        return rho_sun / density_shape("iso", r_sun / rs), rs

    solution = root(equations, [np.log10(0.2), np.log10(20.0)])

    if not solution.success:
        raise RuntimeError(f"{profile}: {solution.message}")

    return 10.0**solution.x[0], 10.0**solution.x[1]


# ============================================================
# J-factor adimensionale lungo la linea di vista:
#
# J(theta) = int ds / r_sun * [rho(r(s,theta)) / rho_sun]^2
#
# Per theta <= 90 deg si usa la sostituzione r = b cosh(u)
# (come in J-factor-spike.py), che campiona in modo
# logaritmico attorno al punto di minima distanza b e gestisce
# le leggi di potenza fino ad angoli piccolissimi.
# Per theta > 90 deg il minimo di r e' in s = 0 e l'integrale
# in s e' regolare.
# ============================================================

def J_los(profile, psi, rho_s, rs, s_max=300.0):

    b = r_sun * np.sin(psi)
    s0 = r_sun * np.cos(psi)

    if psi <= 0.5 * np.pi:

        r_far = np.sqrt(b**2 + (s_max - s0)**2)

        def branch(r_max):
            u_max = np.arccosh(r_max / b)

            def integrand(u):
                r = b * np.cosh(u)
                return (
                    density(profile, r, rho_s, rs) / rho_sun
                )**2 * b * np.cosh(u)

            integral, error = quad(
                integrand, 0.0, u_max,
                epsrel=2e-5, epsabs=0.0, limit=300
            )
            return integral

        return (branch(r_sun) + branch(r_far)) / r_sun

    def integrand(s):
        r = np.sqrt(r_sun**2 + s**2 - 2.0 * r_sun * s * np.cos(psi))
        return (density(profile, r, rho_s, rs) / rho_sun)**2 / r_sun

    integral, error = quad(
        integrand, 0.0, s_max,
        epsrel=2e-5, epsabs=0.0, limit=300
    )

    return integral


# ============================================================
# Figura: pannello principale (0-180 deg) e inset log-log
# ============================================================

profiles = [
    ("moore",    "Moore",    "#2CA02C"),
    ("nfw",      "NFW",      "#1F4FD8"),
    ("einastoB", "EinastoB", "#F4948C"),
    ("einasto",  "Einasto",  "#7B2D8B"),
    ("iso",      "Iso",      "#D62728"),
    ("burkert",  "Burkert",  "#8C5A2B"),
]

theta_main = np.concatenate([
    np.logspace(np.log10(0.1), np.log10(10.0), 25),
    np.linspace(11.0, 180.0, 35),
])
theta_inset = np.logspace(-8.0, 1.0, 40)

fig, ax = plt.subplots(figsize=(7.6, 6.8))
ax_in = fig.add_axes([0.44, 0.52, 0.44, 0.36])

print()
print("Parametri dal fit (rho_sun = 0.4 GeV/cm^3, M60 = 4.7e11 M_sun)")
print("==============================================================")

for profile, label, color in profiles:

    rho_s, rs = solve_rhos_rs(profile)
    print(f"{label:9s}: rho_s = {rho_s:.4g} GeV/cm^3   r_s = {rs:.4g} kpc")

    J_main = [J_los(profile, np.deg2rad(t), rho_s, rs) for t in theta_main]
    J_in = [J_los(profile, np.deg2rad(t), rho_s, rs) for t in theta_inset]

    ax.plot(theta_main, J_main, color=color, lw=2.0, label=label)
    ax_in.plot(theta_inset, J_in, color=color, lw=1.6)

ax.set_yscale("log")
ax.set_xlim(0.0, 180.0)
ax.set_ylim(0.3, 1e3)
ax.set_xticks([0, 30, 60, 90, 120, 150, 180])
ax.set_xlabel(r"$\theta\ \,[\mathrm{degrees}]$", fontsize=12)
ax.set_ylabel(r"$J(\theta)$", fontsize=12)
ax.legend(frameon=False, fontsize=11, loc="lower left",
          bbox_to_anchor=(0.64, 0.13))

ax_in.set_xscale("log")
ax_in.set_yscale("log")
ax_in.set_xlim(1e-8, 10.0)
ax_in.set_ylim(1.0, 1e15)
ax_in.tick_params(labelsize=9)

fig.tight_layout()

output_path = (
    Path(__file__).resolve().parent.parent / "Latex" / "images" / "J-factor-mine.png"
)
fig.savefig(output_path, dpi=300)

print(f"\nFigura salvata in {output_path}")
