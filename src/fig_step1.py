"""fig_step1.py -- the ITO model and the time-varying mirror of Tirole et al.

(a) Real permittivity of the three published ITO Drude sets.
(b) Reflectance of the ITO/Au time-varying mirror of Tirole et al. (2022)
    before and after the pump, from the transfer-matrix model.
(c) Reflectance at 1300 nm against the pump-induced drop of the plasma
    frequency, with the two measured values.
(d) Adiabatic estimate of the reflectance in time for the response kernel
    of the same paper.
"""
import os
import numpy as np
import figstyle as fs
from ito_model import (ITO_SETS, eps_ito, eps_gold, tmm, enz_wavelength,
                       delta_wp_from_pump)


def main():
    plt = fs.setup()
    OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(OUT, exist_ok=True)

    lam = np.linspace(1000e-9, 1700e-9, 701)
    TH = np.arcsin(np.sin(np.deg2rad(65)) / 1.5)          # 65 deg in air -> glass
    D = [40e-9, 100e-9]
    DELTA_PUMPED = -0.10


    def R_stack(wl, scale=1.0):
        return tmm([2.25, eps_ito(wl, "Tirole2022", wp_scale=scale), eps_gold(wl), 1.0],
                   D, wl, TH, "p")[0]


    fig, axs = plt.subplots(2, 2, figsize=(fs.FULL, 4.6), constrained_layout=True)

    # (a) permittivity of the three sets
    ax = axs[0, 0]
    styles = {"Alam2016": (fs.BLUE, "-", "Alam 2016"),
              "Bohn2021": (fs.TEAL, "--", "Bohn 2021"),
              "Tirole2022": (fs.ORANGE, ":", "Tirole 2022")}
    for k, (col, ls, lab) in styles.items():
        e = eps_ito(lam, k)
        ax.plot(lam * 1e9, e.real, color=col, ls=ls, label=lab)
        p = ITO_SETS[k]
        le = enz_wavelength(p["eps_inf"], p["wp"], p["gamma"]) * 1e9
        ax.plot([le], [0], "o", ms=fs.MS, mfc="white", mec=col, mew=1.0)
    ax.axhline(0, color=fs.AXIS, lw=fs.LW_THIN)
    ax.set_xlim(1000, 1700)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Re ε of ITO")
    ax.legend(loc="lower left")
    fs.grid(ax)
    fs.title(ax, "a", "Published Drude sets")

    # (b) reflectance spectra, unpumped and pumped
    ax = axs[0, 1]
    R0 = np.array([R_stack(wl) for wl in lam])
    R1 = np.array([R_stack(wl, 1 + DELTA_PUMPED) for wl in lam])
    ax.plot(lam * 1e9, R0, color=fs.BLUE, label="unpumped")
    ax.plot(lam * 1e9, R1, color=fs.ORANGE, ls="--", label="pumped, −10 %")
    ax.plot([1300, 1300], [0.07, 0.66], ls="none", marker="s", ms=fs.MS + 0.6,
            mfc="none", mec=fs.BLACK, mew=0.9)
    fs.label_end(ax, 1300, 0.66, "measured", dx=3, dy=9, ha="left", va="bottom")
    fs.label_end(ax, 1300, 0.07, "measured", dx=-6, ha="right")
    ax.set_xlim(1000, 1700)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Reflectance")
    ax.legend(loc="lower right", handlelength=1.8)
    fs.grid(ax)
    fs.title(ax, "b", "ITO/Au mirror, p-polarised")

    # (c) reflectance at 1300 nm against the plasma-frequency drop
    ax = axs[1, 0]
    dd = np.linspace(0, 0.30, 301)
    Rd = np.array([R_stack(1300e-9, 1 - x) for x in dd])
    ax.plot(dd * 100, Rd, color=fs.BLUE)
    for Rm in (0.07, 0.66):
        ax.axhline(Rm, color=fs.GREY, lw=fs.LW_THIN, ls=":")
    x66 = np.interp(0.66, Rd, dd) * 100
    ax.plot([0, x66], [0.07, 0.66], ls="none", marker="s", ms=fs.MS + 0.6, mfc="none",
            mec=fs.BLACK, mew=0.9, clip_on=False, zorder=5)
    fs.label_end(ax, x66, 0.66, f"measured 0.66 at {x66:.1f} %", dx=6, dy=-8, va="top")
    ax.text(9, 0.10, "measured 0.07", fontsize=8, color=fs.DARKGREY, va="bottom")
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Drop of the plasma frequency, −Δω$_p$/ω$_p$ (%)")
    ax.set_ylabel("Reflectance at 1300 nm")
    fs.grid(ax)
    fs.title(ax, "c", "Calibrating the modulation depth")

    # (d) adiabatic reflectance in time
    ax = axs[1, 1]
    t = np.arange(-300e-15, 900e-15, 0.5e-15)
    pump = np.exp(-4 * np.log(2) * (t / 220e-15) ** 2)          # 220 fs FWHM pump
    delta = delta_wp_from_pump(t, pump * 100.0)
    delta = -np.abs(delta) / np.abs(delta).max() * abs(DELTA_PUMPED)
    Rt = np.array([R_stack(1300e-9, 1 + x) for x in delta])
    ax.plot(t * 1e15, Rt, color=fs.BLUE)
    ax.axvline(0, color=fs.GREY, lw=fs.LW_THIN, ls=":")
    fs.label_end(ax, 0, 0.92, "pump peak", dx=4, va="top", color=fs.GREY)
    ax.set_xlim(-300, 900)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time (fs)")
    ax.set_ylabel("Reflectance at 1300 nm")
    fs.grid(ax)
    fs.title(ax, "d", "Adiabatic response, τ$_{rise}$ = 30 fs, τ$_{dec}$ = 210 fs")

    fs.save(fig, "fig_step1_ito_model", OUT)
    print("saved", OUT)
    print(f"modulation depth that reproduces R = 0.66: {x66:.2f} %")
    print(f"peak adiabatic R(t) = {Rt.max():.3f} at {t[Rt.argmax()]*1e15:.0f} fs after the pump maximum")


if __name__ == "__main__":
    main()
