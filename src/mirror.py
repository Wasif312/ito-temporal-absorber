"""mirror.py -- the ITO/Au time-varying mirror of Tirole et al. (2022) in the FDTD solver.

glass (eps = 2.25) | 40 nm ITO | 100 nm Au | air, p-polarised probe at 65 deg in
air, i.e. 37.17 deg inside the glass, at a fixed in-plane wavevector set by the
1300 nm carrier.
"""
import numpy as np
from fdtd1d import Medium, Stack, simulate, C0
from ito_model import ITO_SETS, AU_OLMON, kernel_tirole2022

ITO = ITO_SETS["Tirole2022"]
SPECIES = [(ITO["wp"], ITO["gamma"]), (AU_OLMON["wp"], AU_OLMON["gamma"])]
GLASS, AIR = Medium(2.25), Medium(1.0)
M_ITO, M_AU = Medium(ITO["eps_inf"], 0), Medium(AU_OLMON["eps_inf"], 1)
LAM0 = 1300e-9
W0 = 2 * np.pi * C0 / LAM0
KX = 2 * np.pi / LAM0 * np.sin(np.deg2rad(65))      # = 1.5 k0 sin(37.17 deg)
COS_G = np.sqrt(1 - (KX / (1.5 * 2 * np.pi / LAM0)) ** 2)
V_Z = C0 / 1.5 * COS_G                               # z group velocity in glass


def stack(d_ito=40e-9, d_au=100e-9):
    return Stack(GLASS, [(M_ITO, d_ito), (M_AU, d_au)], AIR, SPECIES)


def gaussian_probe(T, tc, w0=W0, chirp=0.0):
    """Current-sheet waveform: Gaussian intensity envelope of FWHM T centred at
    tc, carrier w0, optional linear chirp (rad/s^2)."""
    def f(t):
        x = t - tc
        return np.exp(-2 * np.log(2) * (x / T) ** 2) * np.sin(w0 * x + 0.5 * chirp * x ** 2)
    return f


def arrival_time(tc, z_src=-2.0e-6):
    """Time at which the probe envelope peak reaches the ITO film (z = 0)."""
    return tc + abs(z_src) / V_Z


def delta_profile(t_pump, depth, tau_rise=30e-15, tau_dec=210e-15, T_pump=220e-15,
                  dt=0.25e-15):
    """Drop of the plasma frequency, delta(t) <= 0, with its largest magnitude
    equal to depth: Gaussian pump intensity (FWHM T_pump, peak at t_pump)
    convolved with the response kernel of Tirole et al. (2022). Returns a
    callable of time."""
    tt = np.arange(t_pump - 4 * T_pump, t_pump + 4 * T_pump + 12 * tau_dec, dt)
    pump = np.exp(-4 * np.log(2) * ((tt - t_pump) / T_pump) ** 2)
    ker = kernel_tirole2022(np.arange(0, 12 * tau_dec, dt), tau_rise, tau_dec)
    conv = np.convolve(pump, ker)[: len(tt)] * dt
    d = -depth * conv / conv.max()

    def f(t):
        return np.interp(t, tt, d, left=0.0, right=d[-1])
    return f


def run_pair(probe_fn, t_end, wp_scale=None, momentum=False, ref=None, **kw):
    """Stack run and (if not given) the matching reference run."""
    st = stack()
    run = simulate(st, KX, probe_fn, t_end, wp_scale=wp_scale, momentum=momentum, **kw)
    if ref is None:
        ref = simulate(st, KX, probe_fn, t_end, reference=True, **kw)
    return run, ref
