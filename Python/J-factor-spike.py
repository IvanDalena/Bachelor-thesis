import numpy as np
from scipy.integrate import quad
from scipy.optimize import root
from numpy.polynomial.legendre import leggauss

# Lato del pixel quadrato di Balaji et al. (2303.12107): il pixel
# centrale 0.1 x 0.1 deg dei dati H.E.S.S. del Galactic Centre
# (Fig. 3 di 1603.07730), centrato su Sgr A*.
pixel_side_deg = 0.1


# Costanti fisiche e parametri del problema

r_sun = 8.33                 # kpc
rho_sun = 0.4                # GeV / cm^3

R_mass = 60.0                # kpc
M_target = 4.7e11            # M_sun

kpc_to_cm = 3.0856775814913673e21
M_sun_g = 1.98847e33
GeV_g = 1.78266192e-24

GeV_to_Msun = GeV_g / M_sun_g
kpc3_to_cm3 = kpc_to_cm**3

mass_conversion = kpc3_to_cm3 * GeV_to_Msun

# Buco nero centrale e spike
G_pc = 4.30091e-3            # pc (km/s)^2 / M_sun
c_kms = 299792.458           # km/s

M_BH = 4.15e6                # M_sun
v0 = 105.0                   # km/s

R_h = G_pc * M_BH / v0**2            # pc
R_sp = 0.2 * R_h * 1e-3              # kpc
R_S = 2.0 * G_pc * M_BH / c_kms**2   # pc
r_cut = 4.0 * R_S * 1e-3             # kpc, densità nulla per r < 4 R_S
r_b = 0.01e-3                        # kpc, raggio di rottura less-stellar-heating

# Profilo cored: NFW fuori dal core, legge di potenza dolce dentro.
# Sotto r_c il potenziale è dominato dai barioni, quindi si usano
# gli stessi rho_s e r_s della NFW.
gamma_c = 0.4                # pendenza interna del core
r_c = 1.0                    # kpc, raggio del core

# Densità di saturazione: rho_core = m_DM / (<sigma v> t_BH)
m_DM = 1.0e3                 # GeV
sigmav = 2.2e-26             # cm^3/s
t_BH = 1.0e10 * 3.1557e7     # s
rho_core = m_DM / (sigmav * t_BH)    # GeV/cm^3

# Regione di osservazione: pixel 0.1 x 0.1 deg intorno a Sgr A*,
# come in Balaji et al. (2303.12107).
pixel_half = np.deg2rad(pixel_side_deg) / 2.0


# Profili dell'alone e fit di rho_s e r_s

def rho_nfw(r, rho_s, rs):
    """
    Profilo NFW in GeV/cm^3, con r in kpc.
    """

    x = np.maximum(r, 1e-300) / rs

    return rho_s / (x * (1.0 + x)**2)


def rho_halo(halo, r, rho_s, rs):
    """
    Densità dell'alone in GeV/cm^3.

    halo : str
        'nfw'   : profilo NFW
        'cored' : NFW per r >= r_c, pendenza gamma_c per r < r_c
    """

    if halo == "nfw":
        return rho_nfw(r, rho_s, rs)

    elif halo == "cored":
        r = np.asarray(r, dtype=float)
        x = np.maximum(r, 1e-300)

        return np.where(
            r < r_c,
            float(rho_nfw(r_c, rho_s, rs)) * (x / r_c)**(-gamma_c),
            rho_nfw(r, rho_s, rs)
        )

    else:
        raise ValueError("Alone non riconosciuto.")


def mass_enclosed(R, rho_s, rs):
    """
    Calcola M(<R) in unità di masse solari, per il profilo NFW.

    M(<R) = 4 pi int_0^R rho(r) r^2 dr
    """

    def integrand(r):
        return 4.0 * np.pi * r**2 * float(rho_nfw(r, rho_s, rs))

    integral, error = quad(
        integrand,
        0.0,
        R,
        epsrel=1e-8,
        limit=300
    )

    return integral * mass_conversion


def solve_rhos_rs():
    """
    Trova rho_s e r_s imponendo:

    rho(r_sun) = rho_sun
    M(<60 kpc) = M_target
    """

    def equations(log_variables):
        log_rhos, log_rs = log_variables

        rho_s = 10.0**log_rhos
        rs = 10.0**log_rs

        eq_density = np.log10(
            rho_nfw(r_sun, rho_s, rs) / rho_sun
        )

        eq_mass = np.log10(
            mass_enclosed(R_mass, rho_s, rs) / M_target
        )

        return [eq_density, eq_mass]

    initial_guess = [np.log10(0.2), np.log10(20.0)]

    solution = root(equations, initial_guess)

    if not solution.success:
        raise RuntimeError(solution.message)

    rho_s = 10.0**solution.x[0]
    rs = 10.0**solution.x[1]

    return rho_s, rs


# Profili della spike
#
# La pendenza di Gondolo-Silk dipende da quella dell'alone
# vicino al BH: gamma = 1 per la NFW, gamma = gamma_c per il
# profilo cored.

def gamma_sp_gs(halo):
    """
    Pendenza di Gondolo-Silk: gamma_sp = (9 - 2 gamma) / (4 - gamma).
    """

    gamma = 1.0 if halo == "nfw" else gamma_c

    return (9.0 - 2.0 * gamma) / (4.0 - gamma)


def rho_spike(profile, halo, r, rho_s, rs):
    """
    Spike non saturata, normalizzata per continuità con l'alone a R_sp.

    profile : str
        'gs'  : Gondolo-Silk
        'lsh' : less-stellar-heating, pendenza 3/2 per r > r_b
                e pendenza GS per r < r_b
    """

    rho_Rsp = float(rho_halo(halo, R_sp, rho_s, rs))
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


# J-factor dell'alone lungo la linea di vista

def J_los_dimensionless(halo, psi, rho_s, rs, s_max=300.0):
    """
    Calcola il J-factor adimensionale lungo la linea di vista:

    J_ann(psi) = int ds / r_sun * [rho(r(s,psi))/rho_sun]^2

    Con b = r_sun sin(psi) e s0 = r_sun cos(psi) si ha
    r^2 = b^2 + (s - s0)^2, quindi l'integrale si spezza nei due
    rami s < s0 e s > s0, ciascuno riscritto come integrale in r:

    ds = r dr / sqrt(r^2 - b^2).

    La sostituzione r = b cosh(u) elimina la singolarità
    integrabile in r = b e campiona i raggi in modo
    logaritmico, adatto alle leggi di potenza.
    """

    b = r_sun * np.sin(psi)
    s0 = r_sun * np.cos(psi)

    # Raggio massimo dei due rami: verso l'osservatore (s = 0,
    # dove r = r_sun) e verso s_max.
    r_near = np.sqrt(b**2 + s0**2)
    r_far = np.sqrt(b**2 + (s_max - s0)**2)

    def branch(r_max):

        u_max = np.arccosh(r_max / b)

        def integrand(u):
            r = b * np.cosh(u)

            # ds = b cosh(u) du
            return (
                rho_halo(halo, r, rho_s, rs) / rho_sun
            )**2 * b * np.cosh(u)

        # Punto di rottura al raggio del core, se attraversato
        points = None
        if halo == "cored" and b < r_c < r_max:
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

    return (branch(r_near) + branch(r_far)) / r_sun


def J_halo(halo, rho_s, rs, n=64):
    """
    J-factor fisico del solo alone, integrato sul pixel quadrato
    0.1 x 0.1 deg centrato su Sgr A* (coordinate galattiche
    l, b in [-pixel_half, +pixel_half]):

    J = rho_sun^2 * r_sun[cm] * int dl db cos(b) J(psi),
    con psi = arccos(cos b cos l) distanza angolare dal centro.
    """

    x, w = leggauss(n)

    nodes = pixel_half * x
    weights = pixel_half * w

    angular_integral = 0.0

    for b, wb in zip(nodes, weights):
        for l, wl in zip(nodes, weights):
            psi = np.arccos(np.cos(b) * np.cos(l))
            angular_integral += wb * wl * np.cos(b) * J_los_dimensionless(
                halo, psi, rho_s, rs
            )

    return rho_sun**2 * r_sun * kpc_to_cm * angular_integral


# Contributo della spike
#
# La spike (R_sp ~ 0.3 pc) è molto più piccola della regione
# osservata, quindi la trattiamo come una sorgente puntiforme:
#
# J_sp = (1 / r_sun[cm]^2) int_{4R_S}^{R_sp} 4 pi r^2
#        [rho_spike^2 - rho_halo^2] dr
#
# La sottrazione di rho_halo^2 evita il doppio conteggio
# dell'alone entro R_sp.

def J_spike(profile, halo, rho_s, rs, saturated=False):
    """
    Contributo della spike al J-factor, in GeV^2/cm^5.
    Integrazione in log(r) per gestire l'ampio range dinamico.

    saturated : bool
        False -> limite non annichilante <sigma v> -> 0
        True  -> spike saturata a rho_core:
                 rho = rho' rho_core / (rho' + rho_core)
    """

    def integrand(lnr):
        r = np.exp(lnr)

        rho = rho_spike(profile, halo, r, rho_s, rs)

        if saturated:
            rho = rho * rho_core / (rho + rho_core)

        return 4.0 * np.pi * r**3 * (
            rho**2 - float(rho_halo(halo, r, rho_s, rs))**2
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

    # kpc^3 * GeV^2/cm^6 -> GeV^2/cm^5
    return integral * kpc3_to_cm3 / (r_sun * kpc_to_cm)**2


# Esecuzione

rho_s, rs = solve_rhos_rs()

print()
print("Parametri")
print("=========")
print(f"rho_s    = {rho_s:.6g} GeV/cm^3")
print(f"r_s      = {rs:.6g} kpc")
print(f"R_h      = {R_h:.3f} pc")
print(f"R_sp     = {R_sp * 1e3:.3f} pc")
print(f"4 R_S    = {r_cut * 1e3:.3e} pc")
print(f"rho_core = {rho_core:.3e} GeV/cm^3 "
      f"(m_DM = {m_DM:.0f} GeV, sigmav = {sigmav:.1e} cm^3/s)")
print(f"core     : gamma_c = {gamma_c}, r_c = {r_c} kpc, "
      f"gamma_sp(GS) = {gamma_sp_gs('cored'):.3f}")

print()
print("J-factors sul pixel 0.1 x 0.1 deg [GeV^2/cm^5]")
print("===============================================")

for halo, name in [("nfw", "NFW"), ("cored", "Core")]:

    J_h = J_halo(halo, rho_s, rs)

    J_gs = J_h + J_spike("gs", halo, rho_s, rs)
    J_lsh = J_h + J_spike("lsh", halo, rho_s, rs)

    J_gs_sat = J_h + J_spike("gs", halo, rho_s, rs, saturated=True)
    J_lsh_sat = J_h + J_spike("lsh", halo, rho_s, rs, saturated=True)

    print()
    print(f"Alone {name}:")
    print(f"  solo alone                    "
          f"J = {J_h:.3e}   log10 J = {np.log10(J_h):.2f}")
    print(f"  + GS   (<sigma v> -> 0)       "
          f"J = {J_gs:.3e}   log10 J = {np.log10(J_gs):.2f}")
    print(f"  + LSH  (<sigma v> -> 0)       "
          f"J = {J_lsh:.3e}   log10 J = {np.log10(J_lsh):.2f}")
    print(f"  + GS   (saturata)             "
          f"J = {J_gs_sat:.3e}   log10 J = {np.log10(J_gs_sat):.2f}")
    print(f"  + LSH  (saturata)             "
          f"J = {J_lsh_sat:.3e}   log10 J = {np.log10(J_lsh_sat):.2f}")
