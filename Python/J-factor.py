import numpy as np
from scipy.integrate import quad
from scipy.optimize import root
from numpy.polynomial.legendre import leggauss


# Costanti fisiche e parametri del problema

r_sun = 8.33                 # kpc
rho_sun = 0.4                # GeV / cm^3

R_mass = 60.0                # kpc
M_target = 4.7e11            # M_sun

alpha_einasto = 0.17

kpc_to_cm = 3.0856775814913673e21
M_sun_g = 1.98847e33
GeV_g = 1.78266192e-24

# Conversione:
# rho [GeV/cm^3] * volume [kpc^3] -> massa [M_sun]
GeV_to_Msun = GeV_g / M_sun_g
kpc3_to_cm3 = kpc_to_cm**3

mass_conversion = kpc3_to_cm3 * GeV_to_Msun


# Profili di densità adimensionali
#
# rho(r) = rho_s * shape(r, r_s)

def density_shape(profile, r, rs, rc=None):
    """
    Restituisce la parte adimensionale del profilo di densità.

    profile : str
        'einasto', 'burkert', oppure 'nfw_cored'

    r : float or array
        Raggio galattocentrico in kpc.

    rs : float
        Raggio di scala in kpc.

    rc : float, optional
        Core radius in kpc, solo per NFW cored.
    """

    r = np.asarray(r)
    x = r / rs

    if profile == "einasto":
        return np.exp(
            -(2.0 / alpha_einasto) * (x**alpha_einasto - 1.0)
        )

    elif profile == "burkert":
        return 1.0 / ((1.0 + x) * (1.0 + x**2))

    elif profile == "nfw_cored":
        if rc is None:
            raise ValueError("Per il profilo NFW cored serve rc.")

        return np.where(
            r <= rc,
            (rs / rc) / (1.0 + rc / rs)**2,
            (rs / np.maximum(r, 1e-300)) / (1.0 + r / rs)**2
        )

    else:
        raise ValueError("Profilo non riconosciuto.")


def density(profile, r, rho_s, rs, rc=None):
    """
    Densità fisica in GeV/cm^3.
    """

    return rho_s * density_shape(profile, r, rs, rc)


# Massa entro R

def mass_enclosed(profile, R, rho_s, rs, rc=None):
    """
    Calcola M(<R) in unità di masse solari.

    M(<R) = 4 pi int_0^R rho(r) r^2 dr
    """

    points = None
    if profile == "nfw_cored" and rc is not None and 0.0 < rc < R:
        points = [rc]

    def integrand(r):
        return 4.0 * np.pi * r**2 * float(density(profile, r, rho_s, rs, rc))

    integral, error = quad(
        integrand,
        0.0,
        R,
        points=points,
        epsrel=1e-8,
        limit=300
    )

    return integral * mass_conversion


# Fit di rho_s e r_s

def solve_rhos_rs(profile, rc=None):
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
            density(profile, r_sun, rho_s, rs, rc) / rho_sun
        )

        eq_mass = np.log10(
            mass_enclosed(profile, R_mass, rho_s, rs, rc) / M_target
        )

        return [eq_density, eq_mass]

    initial_guess = [np.log10(0.2), np.log10(20.0)]

    solution = root(equations, initial_guess)

    if not solution.success:
        raise RuntimeError(solution.message)

    rho_s = 10.0**solution.x[0]
    rs = 10.0**solution.x[1]

    return rho_s, rs


# Geometria della linea di vista

def galactocentric_radius(s, psi):
    """
    Raggio galattocentrico:

    r(s, psi)^2 = r_sun^2 + s^2 - 2 r_sun s cos(psi)

    dove:
    s   = distanza lungo la linea di vista, in kpc
    psi = angolo rispetto alla direzione del Galactic Center
    """

    return np.sqrt(
        r_sun**2 + s**2 - 2.0 * r_sun * s * np.cos(psi)
    )


# Integrale lungo la linea di vista

def J_los_dimensionless(profile, psi, rho_s, rs, rc=None, s_max=300.0):
    """
    Calcola il J-factor adimensionale lungo la linea di vista:

    J_ann(psi) = int ds / r_sun * [rho(r(s,psi))/rho_sun]^2

    L'integrale ds è fatto in kpc, quindi il fattore 1/r_sun
    rende J_ann adimensionale.
    """

    cospsi = np.cos(psi)

    def integrand(s):
        r = np.sqrt(
            r_sun**2 + s**2 - 2.0 * r_sun * s * cospsi
        )

        return (
            density(profile, r, rho_s, rs, rc) / rho_sun
        )**2 / r_sun

    # Punto di massimo contributo lungo la l.o.s.
    s_closest = r_sun * cospsi

    points = [
        p for p in [s_closest - 2.0, s_closest, s_closest + 2.0]
        if 0.0 < p < s_max
    ]

    integral, error = quad(
        integrand,
        0.0,
        s_max,
        points=points,
        epsrel=2e-5,
        epsabs=0.0,
        limit=300
    )

    return integral


# Integrazione sulla regione angolare
#
# Omega_obs = Omega(theta < 1 deg, |b| > 0.3 deg)

def integrated_J_factor(profile, rc=None, n_b=36, n_l=36):
    """
    Calcola il J-factor fisico integrato sulla regione:

    theta < 1 deg,
    |b| > 0.3 deg.

    Restituisce:

    log10(J [GeV^2/cm^5]), rho_s, r_s
    """

    rho_s, rs = solve_rhos_rs(profile, rc)

    theta_max = np.deg2rad(1.0)
    b_min = np.deg2rad(0.3)

    # Gauss-Legendre per b positivo.
    # La regione è simmetrica in b, quindi moltiplico per 2.
    xb, wb = leggauss(n_b)

    b_nodes = 0.5 * (theta_max - b_min) * xb + 0.5 * (theta_max + b_min)
    b_weights = 0.5 * (theta_max - b_min) * wb

    angular_integral = 0.0

    for b, wb_i in zip(b_nodes, b_weights):

        # Vincolo theta < theta_max.
        # Per coordinate galattiche:
        #
        # cos(theta) = cos(b) cos(l)
        #
        # quindi, fissato b:
        #
        # |l| < arccos[cos(theta_max)/cos(b)]
        #
        l_max = np.arccos(np.cos(theta_max) / np.cos(b))

        xl, wl = leggauss(n_l)

        l_nodes = l_max * xl
        l_weights = l_max * wl

        integral_l = 0.0

        for l, wl_i in zip(l_nodes, l_weights):

            psi = np.arccos(np.cos(b) * np.cos(l))

            J_los = J_los_dimensionless(
                profile,
                psi,
                rho_s,
                rs,
                rc
            )

            integral_l += wl_i * J_los

        # dOmega = cos(b) db dl
        angular_integral += wb_i * np.cos(b) * integral_l

    # Simmetria b -> -b
    angular_integral *= 2.0

    # Conversione al J fisico:
    #
    # J_phys = rho_sun^2 * r_sun[cm] * integral_dimensionless
    #
    J_phys = rho_sun**2 * r_sun * kpc_to_cm * angular_integral

    return np.log10(J_phys), rho_s, rs


# Esecuzione

profiles = [
    ("Einasto", "einasto", None),
    ("Burkert", "burkert", None),
    ("NFW_30pc", "nfw_cored", 0.03),
    ("NFW_300pc", "nfw_cored", 0.3),
    ("NFW_3kpc", "nfw_cored", 3.0),
]


print()
print("Risultati")
print("=========")
print()

for name, profile, rc in profiles:

    logJ, rho_s_fit, rs_fit = integrated_J_factor(
        profile,
        rc=rc,
        n_b=36,
        n_l=36
    )

    print(f"{name:12s}")
    print(f"  rho_s = {rho_s_fit:.6g} GeV/cm^3")
    print(f"  r_s   = {rs_fit:.6g} kpc")
    print(f"  log10 J = {logJ:.4f}")
    print()
