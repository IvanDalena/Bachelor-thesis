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


# Costanti fisiche e parametri del problema

r_sun = 8330.0               # pc
rho_sun = 0.4                # GeV / cm^3

# Raggio di scala NFW dal fit di J-factor-spike.py
rs = 14.0164e3               # pc

# Buco nero centrale e spike
G_pc = 4.30091e-3            # pc (km/s)^2 / M_sun
c_kms = 299792.458           # km/s

M_BH = 4.15e6                # M_sun
v0 = 105.0                   # km/s

R_h = G_pc * M_BH / v0**2            # pc
R_sp = 0.2 * R_h                     # pc
R_S = 2.0 * G_pc * M_BH / c_kms**2   # pc
r_cut = 4.0 * R_S                    # pc, densità nulla sotto
r_b = 0.01                           # pc, raggio di rottura less-stellar-heating

# Profilo cored: NFW fuori dal core, pendenza gamma_c dentro
gamma_c = 0.4
r_c = 1000.0                 # pc

# Densità di saturazione: rho_core = m_DM / (<sigma v> t_BH)
m_DM = 1.0e3                 # GeV
sigmav = 2.2e-26             # cm^3/s
t_BH = 1.0e10 * 3.1557e7     # s
rho_core = m_DM / (sigmav * t_BH)    # GeV/cm^3


# Profili di densità (spike non saturate, <sigma v> -> 0)

def rho_nfw(r):
    """
    Profilo NFW in GeV/cm^3, r in pc, normalizzato a rho_sun in r_sun.
    """

    rho_s = rho_sun * (r_sun / rs) * (1.0 + r_sun / rs)**2
    x = np.maximum(r, 1e-300) / rs

    return rho_s / (x * (1.0 + x)**2)


def rho_halo(halo, r):
    """
    Densità dell'alone: 'nfw' oppure 'cored' (NFW per r >= r_c,
    pendenza gamma_c per r < r_c).
    """

    if halo == "nfw":
        return rho_nfw(r)

    r = np.asarray(r, dtype=float)
    x = np.maximum(r, 1e-300)

    return np.where(
        r < r_c,
        float(rho_nfw(r_c)) * (x / r_c)**(-gamma_c),
        rho_nfw(r)
    )


def gamma_sp_gs(halo):
    """
    Pendenza di Gondolo-Silk: gamma_sp = (9 - 2 gamma) / (4 - gamma).
    """

    gamma = 1.0 if halo == "nfw" else gamma_c

    return (9.0 - 2.0 * gamma) / (4.0 - gamma)


def density_profile(profile, halo, r):
    """
    Profilo completo alone + spike non saturata:

    rho = 0                 per r < 4 R_S
    rho = spike             per 4 R_S <= r < R_sp
    rho = alone             per r >= R_sp

    profile : str
        'gs'  : Gondolo-Silk
        'lsh' : less-stellar-heating, pendenza 3/2 per r > r_b
                e pendenza GS per r < r_b
    """

    r = np.asarray(r, dtype=float)
    x = np.maximum(r, 1e-300)

    rho_Rsp = float(rho_halo(halo, R_sp))
    gs = gamma_sp_gs(halo)

    if profile == "gs":
        spike = rho_Rsp * (x / R_sp)**(-gs)

    elif profile == "lsh":
        spike = np.where(
            r < r_b,
            rho_Rsp * (r_b / R_sp)**(-1.5) * (x / r_b)**(-gs),
            rho_Rsp * (x / R_sp)**(-1.5)
        )

    else:
        raise ValueError("Profilo non riconosciuto.")

    out = np.where(r < R_sp, spike, rho_halo(halo, r))

    return np.where(r < r_cut, 0.0, out)


# Due figure separate: alone NFW e cored

r = np.logspace(-7.0, 3.0, 4000)   # pc

panels = [
    ("nfw", "NFW halo", "Spike_profiles_NFW.png"),
    ("cored", r"cored halo ($\gamma_c = 0.4$, $r_c = 1$ kpc)",
     "Spike_profiles_cored.png"),
]

images_dir = Path(__file__).resolve().parent.parent / "Latex" / "images"

for halo, title, filename in panels:

    fig, ax = plt.subplots(figsize=(6.4, 5.2))

    ax.loglog(r, rho_halo(halo, r), label="no spike", color="0.45", lw=1.6)

    for profile, label, color in [
        ("gs",  "GS spike",  "#0072B2"),
        ("lsh", "LSH spike", "#D55E00"),
    ]:
        rho = density_profile(profile, halo, r)
        ax.loglog(r, np.where(rho > 0.0, rho, np.nan),
                  label=label, color=color, lw=2.0)

    # Densità di saturazione e raggi caratteristici
    ax.axhline(rho_core, color="0.55", ls=":", lw=1.0)
    ax.text(2.5e-7, 3.6e11, r"$\rho_{\mathrm{core}}$", color="0.35", fontsize=10)

    ax.axvline(R_sp, color="0.55", ls="--", lw=1.0)
    ax.text(1.3 * R_sp, 3e-2, r"$R_{\mathrm{sp}}$", color="0.35", fontsize=10)

    ax.axvline(r_b, color="0.55", ls="-.", lw=0.8)
    ax.text(1.3 * r_b, 3e-2, r"$r_b$", color="0.35", fontsize=10)

    ax.axvline(r_cut, color="0.55", ls="--", lw=1.0)
    ax.text(1.3 * r_cut, 3e-2, r"$4R_S$", color="0.35", fontsize=10)

    ax.set_xlim(1e-7, 1e3)
    ax.set_ylim(1e-2, 1e18)
    ax.set_xlabel(r"$r\ \,[\mathrm{pc}]$", fontsize=12)
    ax.set_ylabel(r"$\rho\ \,[\mathrm{GeV\,cm^{-3}}]$", fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.legend(frameon=False, fontsize=11, loc="upper right")

    fig.tight_layout()

    output_path = images_dir / filename
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"Figura salvata in {output_path}")


# Figura unica: NFW (linea continua) e cored (tratteggiata) sovrapposti.
# Stesso colore per lo stesso caso (no spike / GS / LSH) e stile di linea
# diverso per i due aloni, cosi' il confronto tra i due e' immediato.

from matplotlib.lines import Line2D

fig, ax = plt.subplots(figsize=(6.8, 5.6))

halo_styles = [
    ("nfw", "-"),
    ("cored", "--"),
]

for halo, ls in halo_styles:

    ax.loglog(r, rho_halo(halo, r), color="0.45", lw=1.8, ls=ls)

    for profile, color in [("gs", "#0072B2"), ("lsh", "#D55E00")]:
        rho = density_profile(profile, halo, r)
        ax.loglog(r, np.where(rho > 0.0, rho, np.nan),
                  color=color, lw=2.0, ls=ls)

# Densita' di saturazione e raggi caratteristici (uguali per i due aloni).
# Rese puntinate cosi' si distinguono dalle curve tratteggiate del cored.
ax.axhline(rho_core, color="0.55", ls=":", lw=1.0)
ax.text(2.5e-7, 3.6e11, r"$\rho_{\mathrm{core}}$", color="0.35", fontsize=10)

for xline, xlabel in [(R_sp, r"$R_{\mathrm{sp}}$"),
                      (r_b, r"$r_b$"),
                      (r_cut, r"$4R_S$")]:
    ax.axvline(xline, color="0.7", ls=":", lw=0.9)
    ax.text(1.3 * xline, 3e-2, xlabel, color="0.35", fontsize=10)

ax.set_xlim(1e-7, 1e3)
ax.set_ylim(1e-2, 1e18)
ax.set_xlabel(r"$r\ \,[\mathrm{pc}]$", fontsize=12)
ax.set_ylabel(r"$\rho\ \,[\mathrm{GeV\,cm^{-3}}]$", fontsize=12)

# Doppia legenda: il colore identifica il caso, lo stile di linea l'alone.
color_handles = [
    Line2D([0], [0], color="0.45", lw=1.8, label="no spike"),
    Line2D([0], [0], color="#0072B2", lw=2.0, label="GS spike"),
    Line2D([0], [0], color="#D55E00", lw=2.0, label="LSH spike"),
]
style_handles = [
    Line2D([0], [0], color="black", lw=1.8, ls="-", label="NFW halo"),
    Line2D([0], [0], color="black", lw=1.8, ls="--",
           label=r"cored halo ($\gamma_c = 0.4$, $r_c = 1$ kpc)"),
]

leg_color = ax.legend(handles=color_handles, frameon=False, fontsize=10,
                      loc="upper right", bbox_to_anchor=(1.0, 1.0))
ax.add_artist(leg_color)
ax.legend(handles=style_handles, frameon=False, fontsize=10,
          loc="upper right", bbox_to_anchor=(1.0, 0.80))

fig.tight_layout()

output_path = images_dir / "Spike_profiles.png"
fig.savefig(output_path, dpi=300)
plt.close(fig)

print(f"Figura unita salvata in {output_path}")

print(f"rho_core = {rho_core:.3e} GeV/cm^3")
