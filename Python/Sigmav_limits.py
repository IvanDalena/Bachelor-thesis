import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.integrate import quad
from numpy.polynomial.legendre import leggauss

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


# Costanti fisiche

kpc_to_cm = 3.0856775814913673e21
G_pc = 4.30091e-3            # pc (km/s)^2 / M_sun
c_kms = 299792.458           # km/s
v0 = 105.0                   # km/s, dispersione di velocita'

# Sezione d'urto che riproduce la relic abundance per DM
# autoconiugata con massa sopra ~10 GeV (Steigman et al. 2012)
sigmav_relic = 2.2e-26       # cm^3/s

# Densita' di saturazione della spike (benchmark m_DM = 1 TeV)
m_DM = 1.0e3                 # GeV
t_BH = 1.0e10 * 3.1557e7     # s
rho_core = m_DM / (sigmav_relic * t_BH)   # GeV/cm^3

# Raggio di rottura less-stellar-heating
r_b = 0.01e-3                # kpc

# Core del profilo cored (capitolo 2)
gamma_c = 0.4
r_c = 1.0                    # kpc


# Set di parametri NFW
#
# Ogni analisi usa una propria normalizzazione dell'alone:
#
# 'mine' : questa tesi, rho_s e r_s dal fit di J-factor.py
#          (rho_sun = 0.4 GeV/cm^3 a r_sun = 8.33 kpc,
#           M(<60 kpc) = 4.7e11 M_sun)
# 'hess' : H.E.S.S. Inner Galaxy Survey (2207.10471, Tab. 2)
# 'sala' : Balaji et al. (2303.12107)
#
# rho_s in GeV/cm^3, r_s e r_sun in kpc, M_BH in M_sun.
# Dove il paper quota rho_sun invece di rho_s, rho_s e' fissato
# da rho_nfw(r_sun) = rho_sun.

def rho_s_from_local(rho_sun, r_sun, rs):
    x = r_sun / rs
    return rho_sun * x * (1.0 + x)**2


PARAMS = {
    "mine": {
        "r_sun": 8.33,
        "rs": 14.0164,
        "rho_s": rho_s_from_local(0.4, 8.33, 14.0164),
        "M_BH": 4.15e6,
    },
    "hess": {
        "r_sun": 8.5,
        "rs": 21.0,
        "rho_s": 0.307,
        "M_BH": None,
    },
    "sala": {
        "r_sun": 8.2,
        "rs": 18.6,
        "rho_s": rho_s_from_local(0.383, 8.2, 18.6),
        "M_BH": 4.3e6,
    },
}


def spike_radii(M_BH):
    """
    Restituisce (R_sp, r_cut) in kpc dal M_BH in M_sun.
    """

    R_h = G_pc * M_BH / v0**2            # pc
    R_sp = 0.2 * R_h * 1e-3              # kpc
    r_cut = 4.0 * 2.0 * G_pc * M_BH / c_kms**2 * 1e-3   # kpc

    return R_sp, r_cut


# Profili di densita' (come in J-factor-spike.py)

def rho_nfw(r, p):
    x = np.maximum(r, 1e-300) / p["rs"]
    return p["rho_s"] / (x * (1.0 + x)**2)


def rho_halo(halo, r, p):
    """
    'nfw'        : profilo NFW
    'cored'      : NFW per r >= r_c, pendenza gamma_c per r < r_c
    'cored_flat' : NFW per r >= r_c, costante per r < r_c
                   (eq. del capitolo 1, caso conservativo gamma_c = 0)
    """

    if halo == "nfw":
        return rho_nfw(r, p)

    r = np.asarray(r, dtype=float)
    x = np.maximum(r, 1e-300)

    if halo == "cored":
        return np.where(
            r < r_c,
            float(rho_nfw(r_c, p)) * (x / r_c)**(-gamma_c),
            rho_nfw(r, p)
        )

    elif halo == "cored_flat":
        return np.where(
            r < r_c,
            float(rho_nfw(r_c, p)),
            rho_nfw(r, p)
        )

    else:
        raise ValueError("Alone non riconosciuto.")


def gamma_sp_gs(halo):
    gamma = 1.0 if halo == "nfw" else gamma_c
    return (9.0 - 2.0 * gamma) / (4.0 - gamma)


def rho_spike(profile, halo, r, p):
    """
    Spike non saturata, continua con l'alone a R_sp.
    """

    R_sp, _ = spike_radii(p["M_BH"])
    rho_Rsp = float(rho_halo(halo, R_sp, p))
    gs = gamma_sp_gs(halo)

    if profile == "gs":
        return rho_Rsp * (r / R_sp)**(-gs)

    elif profile == "lsh":
        return np.where(
            r < r_b,
            rho_Rsp * (r_b / R_sp)**(-1.5) * (r / r_b)**(-gs),
            rho_Rsp * (r / R_sp)**(-1.5)
        )

    else:
        raise ValueError("Profilo non riconosciuto.")


# J-factor lungo la linea di vista (sostituzione r = b cosh u,
# come in J-factor-spike.py)

def J_los(halo, psi, p, s_max=300.0):
    """
    Integrale di rho^2 lungo la linea di vista, in GeV^2/cm^6 * kpc.
    """

    r_sun = p["r_sun"]

    b = r_sun * np.sin(psi)
    s0 = r_sun * np.cos(psi)

    r_near = np.sqrt(b**2 + s0**2)
    r_far = np.sqrt(b**2 + (s_max - s0)**2)

    def branch(r_max):

        u_max = np.arccosh(r_max / b)

        def integrand(u):
            r = b * np.cosh(u)
            return float(rho_halo(halo, r, p))**2 * b * np.cosh(u)

        points = None
        if halo in ("cored", "cored_flat") and b < r_c < r_max:
            points = [np.arccosh(r_c / b)]

        integral, error = quad(
            integrand,
            0.0,
            u_max,
            points=points,
            epsrel=2e-5,
            epsabs=0.0,
            limit=300
        )

        return integral

    return branch(r_near) + branch(r_far)


# J-factor sul pixel 0.1 x 0.1 deg intorno a Sgr A*, come in
# Balaji et al. (2303.12107): il pixel centrale dei dati H.E.S.S.
# del Galactic Centre (Fig. 3 di 1603.07730). Coordinate
# galattiche l, b in [-side/2, +side/2].

def J_pixel(halo, p, side_deg=0.1, n=64):

    half = np.deg2rad(side_deg) / 2.0

    x, w = leggauss(n)

    nodes = half * x
    weights = half * w

    angular_integral = 0.0

    for b, wb in zip(nodes, weights):
        for l, wl in zip(nodes, weights):
            psi = np.arccos(np.cos(b) * np.cos(l))
            angular_integral += wb * wl * np.cos(b) * J_los(halo, psi, p)

    return angular_integral * kpc_to_cm


def J_spike(profile, halo, p, saturated=True):
    """
    Contributo puntiforme della spike, in GeV^2/cm^5,
    con saturazione a rho_core e sottrazione dell'alone
    per evitare il doppio conteggio.
    """

    R_sp, r_cut = spike_radii(p["M_BH"])

    def integrand(lnr):
        r = np.exp(lnr)

        rho = rho_spike(profile, halo, r, p)

        if saturated:
            rho = rho * rho_core / (rho + rho_core)

        return 4.0 * np.pi * r**3 * (
            rho**2 - float(rho_halo(halo, r, p))**2
        )

    points = [np.log(r_b)] if profile == "lsh" else None

    integral, error = quad(
        integrand,
        np.log(r_cut),
        np.log(R_sp),
        points=points,
        epsrel=1e-8,
        limit=300
    )

    kpc3_to_cm3 = kpc_to_cm**3

    return integral * kpc3_to_cm3 / (p["r_sun"] * kpc_to_cm)**2


# J-factor sulla regione dell'Inner Galaxy Survey:
# 0.5 deg < theta < 3 deg, |b| > 0.3 deg

def J_igs(halo, p, n_b=40, n_l=40):

    theta_in = np.deg2rad(0.5)
    theta_out = np.deg2rad(3.0)
    b_min = np.deg2rad(0.3)

    xb, wb = leggauss(n_b)
    xl, wl = leggauss(n_l)

    # b da b_min a theta_out (b > 0, poi simmetria x2)
    b_nodes = 0.5 * (theta_out - b_min) * xb + 0.5 * (theta_out + b_min)
    b_weights = 0.5 * (theta_out - b_min) * wb

    angular_integral = 0.0

    for b, wb_i in zip(b_nodes, b_weights):

        # theta < theta_out: |l| < l_out
        l_out = np.arccos(np.clip(np.cos(theta_out) / np.cos(b), -1.0, 1.0))

        # theta > theta_in: |l| > l_in (solo se b < theta_in)
        if b < theta_in:
            l_in = np.arccos(np.clip(np.cos(theta_in) / np.cos(b), -1.0, 1.0))
        else:
            l_in = 0.0

        if l_out <= l_in:
            continue

        # l da l_in a l_out (l > 0, poi simmetria x2)
        l_nodes = 0.5 * (l_out - l_in) * xl + 0.5 * (l_out + l_in)
        l_weights = 0.5 * (l_out - l_in) * wl

        integral_l = 0.0

        for l, wl_i in zip(l_nodes, l_weights):
            psi = np.arccos(np.cos(b) * np.cos(l))
            integral_l += wl_i * J_los(halo, psi, p)

        # dOmega = cos(b) db dl, simmetrie b -> -b e l -> -l
        angular_integral += wb_i * np.cos(b) * integral_l

    angular_integral *= 4.0

    return angular_integral * kpc_to_cm


# Curve digitalizzate (massa in TeV)
#
# hess_igs_nfw_ww.csv   : limite NFW, H.E.S.S. IGS (2207.10471, Fig. 2)
# sgrA_spike_gs_ww.csv  : limite NFW+GS, Sgr A* (2303.12107, Fig. 3,
#                         pannello in alto a sinistra, curva blu continua)
# sgrA_spike_lsh_ww.csv : limite NFW+LSH, Sgr A* (2303.12107, Fig. 3,
#                         pannello in alto a sinistra, curva viola continua)
#
# Le due curve di spike sono quelle a rho_sun = 0.383 GeV/cm^3 (continue,
# blu per GS e viola per LSH); le tratteggiate dello stesso pannello sono a
# rho_sun = 0.55 e non vanno usate qui. Gli assi del pannello vanno da
# 3.2e2 a 1e5 GeV e da 1e-32 a 1e-22 cm^3/s: le curve concordano entro il
# 2% con i dati vettoriali del file NFW+spikeWW.pdf del sorgente arXiv.

data_dir = Path(__file__).resolve().parent / "data"


def load_curve(name):
    """
    Legge un CSV di WebPlotDigitizer (separatore ';', virgola
    decimale) e lo ordina per massa crescente.
    """

    rows = []
    with open(data_dir / name) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            a, b = line.split(";")
            rows.append((
                float(a.replace(",", ".")),
                float(b.replace(",", "."))
            ))

    m, sv = np.array(rows).T

    order = np.argsort(m)

    return m[order], sv[order]


# Fattori di rescaling
#
# Il limite scala come 1/J, quindi il limite riferito al nostro
# NFW si ottiene da quello del paper come
#
# sigmav_lim^mine = sigmav_lim^paper * (J^paper / J^mine)
#
# con i J calcolati sulla regione di osservazione di ciascuna
# analisi.

print()
print("J-factor sulla regione IGS (0.5-3 deg, |b| > 0.3 deg)")
print("=====================================================")

J_igs_hess = J_igs("nfw", PARAMS["hess"])
J_igs_mine = J_igs("nfw", PARAMS["mine"])
J_igs_cored = J_igs("cored_flat", PARAMS["mine"])

print(f"NFW H.E.S.S.       : log10 J = {np.log10(J_igs_hess):.3f}")
print(f"NFW tesi           : log10 J = {np.log10(J_igs_mine):.3f}")
print(f"cored flat (1 kpc) : log10 J = {np.log10(J_igs_cored):.3f}")

k_nfw = J_igs_hess / J_igs_mine
k_cored = J_igs_mine / J_igs_cored

print(f"k_nfw   = J_hess / J_mine  = {k_nfw:.3f}")
print(f"k_cored = J_nfw  / J_cored = {k_cored:.3f}")

print()
print("J-factor sul pixel 0.1 x 0.1 deg intorno a Sgr A*")
print("===================================================")

J_sgra = {}
for tag in ("sala", "mine"):
    p = PARAMS[tag]
    J_h = J_pixel("nfw", p)
    J_sgra[tag] = {
        "gs": J_h + J_spike("gs", "nfw", p),
        "lsh": J_h + J_spike("lsh", "nfw", p),
    }
    print(f"{tag:5s}: log10 J(NFW+GS)  = {np.log10(J_sgra[tag]['gs']):.3f}   "
          f"log10 J(NFW+LSH) = {np.log10(J_sgra[tag]['lsh']):.3f}")

k_gs = J_sgra["sala"]["gs"] / J_sgra["mine"]["gs"]
k_lsh = J_sgra["sala"]["lsh"] / J_sgra["mine"]["lsh"]

print(f"k_gs  = {k_gs:.3f}")
print(f"k_lsh = {k_lsh:.3f}")


# Curve riscalate

m_nfw, sv_nfw = load_curve("hess_igs_nfw_ww.csv")
m_gs, sv_gs = load_curve("sgrA_spike_gs_ww.csv")
m_lsh, sv_lsh = load_curve("sgrA_spike_lsh_ww.csv")

sv_nfw = sv_nfw * k_nfw
sv_cored = sv_nfw * k_cored
sv_gs = sv_gs * k_gs
sv_lsh = sv_lsh * k_lsh

print()
print(f"Minimo del limite NFW riscalato: "
      f"{sv_nfw.min():.2e} cm^3/s a m = {m_nfw[np.argmin(sv_nfw)]:.2f} TeV")


# Figura capitolo 1: NFW + cored + relic

images_dir = Path(__file__).resolve().parent.parent / "Latex" / "images"

fig, ax = plt.subplots(figsize=(8.0, 5.5))

ax.loglog(m_nfw, sv_nfw, color="#0072B2", lw=2.0, label="NFW")
ax.loglog(m_nfw, sv_cored, color="#009E73", lw=2.0,
          label=r"cored NFW ($r_c = 1$ kpc)")

ax.axhline(sigmav_relic, color="0.35", ls="--", lw=1.4,
           label=r"$\langle\sigma v\rangle_{\mathrm{relic}}$")

ax.set_xlim(0.2, 70.0)
ax.set_ylim(1e-26, 1e-22)
ax.set_xlabel(r"$m_{\mathrm{DM}}\ \,[\mathrm{TeV}]$", fontsize=12)
ax.set_ylabel(r"$\langle\sigma v\rangle\ \,[\mathrm{cm^3\,s^{-1}}]$",
              fontsize=12)
ax.legend(frameon=False, fontsize=11, loc="upper left")

fig.tight_layout()
fig.savefig(images_dir / "Sigmav_limits.png", dpi=300)
print(f"Figura salvata in {images_dir / 'Sigmav_limits.png'}")


# Figura capitolo 2: NFW + GS + LSH + relic

fig, ax = plt.subplots(figsize=(8.0, 5.5))

ax.loglog(m_nfw, sv_nfw, color="0.45", lw=2.0, label="NFW (no spike)")
ax.loglog(m_gs, sv_gs, color="#0072B2", lw=2.0, label="NFW + GS spike")
ax.loglog(m_lsh, sv_lsh, color="#D55E00", lw=2.0, label="NFW + LSH spike")

ax.axhline(sigmav_relic, color="0.35", ls="--", lw=1.4,
           label=r"$\langle\sigma v\rangle_{\mathrm{relic}}$")

ax.set_xlim(0.2, 1e2)
ax.set_ylim(1e-32, 1e-22)
ax.set_xlabel(r"$m_{\mathrm{DM}}\ \,[\mathrm{TeV}]$", fontsize=12)
ax.set_ylabel(r"$\langle\sigma v\rangle\ \,[\mathrm{cm^3\,s^{-1}}]$",
              fontsize=12)
ax.legend(frameon=False, fontsize=11, loc="lower left")

fig.tight_layout()
fig.savefig(images_dir / "Sigmav_limits_spike.png", dpi=300)
print(f"Figura salvata in {images_dir / 'Sigmav_limits_spike.png'}")
