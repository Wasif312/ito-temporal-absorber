"""angle_scan.py -- route 3 with the angle free for both films.

For the probe of the main estimate (centre 1370 nm, 150 nm chirp), the static
film may choose any fixed drop of omega_p in 0-10 % and the switched film any
pump delay and probe duration, at each angle inside the glass. Angles above
41.8 deg need prism coupling.
"""
import numpy as np
import chirp_scope as cs
from ito_model import eps_ito, eps_gold, tmm


def main():
    lam_c, span = 1370e-9, 150e-9
    rows = []
    for deg in (30.0, 37.17, 41.0, 45.0, 50.0, 55.0, 57.0, 60.0, 65.0):
        th = np.deg2rad(deg)
        cs.Rmap[:] = np.array([[tmm([2.25, eps_ito(wl, "Tirole2022", wp_scale=1 - x), eps_gold(wl), 1.0],
                                    cs.D, wl, th, "p")[0] for wl in cs.lam] for x in cs.dl])
        a_s = max(cs.absorbed(lam_c, span, 300e-15, np.full_like(cs.t, x)) for x in cs.dl)
        a_v = max(cs.absorbed(lam_c, span, T, cs.drop_profile(tp, cs.DMAX))
                  for T in (150e-15, 300e-15, 450e-15, 600e-15)
                  for tp in np.arange(-900e-15, 301e-15, 25e-15))
        rows.append((deg, a_s, a_v))
        print(f"angle in glass {deg:5.2f} deg: best static {a_s:.3f}, best switched {a_v:.3f}, gain {a_v - a_s:+.3f}")
    s = max(rows, key=lambda r: r[1])
    v = max(rows, key=lambda r: r[2])
    print(f"best static over angles {s[1]:.3f} at {s[0]} deg; best switched {v[2]:.3f} at {v[0]} deg; "
          f"gain {v[2]-s[1]:+.3f}")


if __name__ == "__main__":
    main()
