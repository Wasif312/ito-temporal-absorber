"""hot_electron.py -- plasma frequency and damping of ITO against electron temperature.

Nonparabolic conduction band in the Kane form

    E (1 + C E) = hbar^2 k^2 / (2 m0*),

with C = 0.4191 eV^-1 and m0* = 0.3964 me (Liu et al., as read in Un et al.)
and a Fermi energy E_F at 300 K of 0.88-1 eV (Un et al., Baxter et al.; full
references in the README). At electron temperature Te the chemical
potential mu keeps the electron density fixed,

    N = integral g(E) f(E; mu, Te) dE,    g(E) dE = k^2 dk / pi^2,

and the Drude weight follows from the band velocity v = (1/hbar) dE/dk,

    wp^2(Te) = (e^2 / 3 eps0) integral g(E) v(E)^2 (-df/dE) dE,

which reduces to N e^2 / (eps0 m0* (1 + 2 C E_F)) at Te = 0. Hot electrons
occupy states of larger mass, so wp falls as Te rises. The damping follows
gamma(Te) = gamma0 (1 + Te / T0) with T0 = 15 000 K (Baxter et al.),
normalised here to its value at 300 K. The pump in route 1 is specified by the
largest drop of wp; this module returns the electron temperature and the
damping scale that belong to a given drop.
"""
import numpy as np

KB = 8.617333262e-5            # eV/K
C_NP = 0.4191                  # eV^-1
T0_GAMMA = 15000.0             # K
TE_ROOM = 300.0                # K


def _moments(mu, Te, E_F, n=40001):
    """Density and Drude weight (common factors dropped) for chemical
    potential mu (eV) and temperature Te (K)."""
    kT = KB * Te
    E = np.linspace(0.0, max(E_F, mu) + 60 * kT, n)
    k = np.sqrt(E + C_NP * E ** 2)                   # in units with 2 m0* / hbar^2 = 1
    x = np.clip((E - mu) / kT, -700, 700)
    f = 1.0 / (np.exp(x) + 1.0)
    mdf = f * (1 - f) / kT                           # -df/dE
    g = k * (1 + 2 * C_NP * E)                       # g(E) up to a constant: k^2 dk/dE
    gv2 = k ** 3 / (1 + 2 * C_NP * E)                # g v^2 up to the same kind of constant
    return np.trapezoid(g * f, E), np.trapezoid(gv2 * mdf, E)


def wp2_ratio(Te, E_F=0.9):
    """wp^2(Te) / wp^2(300 K) at fixed electron density."""
    def at(T):
        k_F = np.sqrt(E_F + C_NP * E_F ** 2)
        N0 = 2 * k_F ** 3 / 3                         # integral of g from 0 to E_F at T = 0
        lo, hi = E_F - 40.0, E_F + 0.5
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            N, _ = _moments(mid, T, E_F)
            if N > N0:
                hi = mid
            else:
                lo = mid
        return _moments(0.5 * (lo + hi), T, E_F)[1]
    return at(Te) / at(TE_ROOM)


def drop_of_te(Te, E_F=0.9):
    """Fractional drop of wp, 1 - wp(Te)/wp(300 K)."""
    return 1.0 - np.sqrt(wp2_ratio(Te, E_F))


def gamma_scale_of_te(Te):
    return (1 + Te / T0_GAMMA) / (1 + TE_ROOM / T0_GAMMA)


class DropMap:
    """Tabulated link between the drop of wp, the electron temperature and the
    damping scale, for one Fermi energy."""
    def __init__(self, E_F=0.9, te_max=40000.0, n=161):
        self.E_F = E_F
        self.te = np.linspace(TE_ROOM, te_max, n)
        self.drop = np.array([drop_of_te(T, E_F) for T in self.te])
        if np.any(np.diff(self.drop) <= 0):
            raise ValueError("drop must rise with temperature")

    def te_of_drop(self, drop):
        return np.interp(np.asarray(drop, dtype=float), self.drop, self.te)

    def gamma_scale(self, drop):
        return gamma_scale_of_te(self.te_of_drop(drop))


def load_map(E_F=0.9):
    """DropMap for E_F from results/hot_electron.json when it exists (the
    table takes about a minute to compute), else computed here."""
    import json
    import os
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "hot_electron.json")
    if os.path.exists(p):
        d = json.load(open(p)).get(str(E_F))
        if d is not None:
            m = DropMap.__new__(DropMap)
            m.E_F, m.te, m.drop = E_F, np.array(d["te"]), np.array(d["drop"])
            return m
    return DropMap(E_F)


if __name__ == "__main__":
    import json
    import os
    OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    # check: the T = 0 limit of the Drude weight is k_F^3 / (1 + 2 C E_F)
    E_F = 0.9
    k_F = np.sqrt(E_F + C_NP * E_F ** 2)
    _, w_lowT = _moments(E_F, 5.0, E_F)
    print(f"low-temperature Drude weight {w_lowT:.6f} against k_F^3/(1 + 2 C E_F) = "
          f"{k_F ** 3 / (1 + 2 * C_NP * E_F):.6f}")
    out = {}
    for E_F in (0.88, 0.9, 1.0):
        m = DropMap(E_F)
        rows = []
        for d in (0.05, 0.10, 0.15, 0.19, 0.20, 0.23, 0.25, 0.30, 0.35, 0.40):
            Te = float(m.te_of_drop(d))
            rows.append(dict(drop=d, Te=Te, gamma_scale=float(gamma_scale_of_te(Te))))
        out[str(E_F)] = dict(te=m.te.tolist(), drop=m.drop.tolist(), table=rows)
        print(f"E_F = {E_F} eV: " + "; ".join(f"drop {r['drop']*100:.0f} % at Te {r['Te']:.0f} K, gamma x"
                                              f"{r['gamma_scale']:.2f}" for r in rows))
    with open(os.path.join(OUT, "hot_electron.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("saved results/hot_electron.json")
