#!/usr/bin/env python3

import numpy as np
import scipy.stats as ss
import string
import itertools
import scipy.interpolate as inter

heat_names = list(
    map(
        lambda x: "".join(x),
        itertools.product([""] + list(string.ascii_lowercase), repeat=2),
    )
)[1:]

if __name__ == "__main__":
    nheats = 20
    nper = 15

    srange = np.linspace(100.0, 200.0, nper)[::-1]
    Trange = np.linspace(500, 750.0, nper) + 273.15
    frac = 0.5

    ydata = np.linspace(125.0, 175.0, nper)[::-1] * 1.9

    yfn = inter.interp1d(Trange, ydata)

    poly_lower = [-1.0e3, 2.0e4]
    C_lower = 16.0

    poly_lower_mcr = -np.array([-3.0e3, 2.0e4 * 1])
    C_lower_mcr = -10

    poly_upper = [-2.0e3, 3.0e4]
    C_upper = 20.0

    poly_upper_mcr = -np.array([-6.0e3, 4.0e4 * 1])
    C_upper_mcr = -28.0

    C_scale = 0.75

    noise = 30.0

    C_dist_upper = ss.norm(C_upper, C_scale)
    C_dist_lower = ss.norm(C_lower, C_scale)

    C_dist_mcr_upper = ss.norm(C_upper_mcr, C_scale)
    C_dist_mcr_lower = ss.norm(C_lower_mcr, C_scale)

    heats = []
    data = np.zeros((nheats * nper, 4))
    for i in range(nheats):
        heats.extend([heat_names[i]] * nper)
        # Generate real Cs
        C_upper = C_dist_upper.rvs()
        C_lower = C_dist_lower.rvs()

        C_upper_mcr = C_dist_mcr_upper.rvs()
        C_lower_mcr = C_dist_mcr_lower.rvs()

        # Split and get LMPS
        real_stress = ss.norm(srange, noise).rvs()

        lmp_upper = np.polyval(poly_upper, np.log10(real_stress))
        lmp_lower = np.polyval(poly_lower, np.log10(real_stress))

        lmp_upper_mcr = np.polyval(poly_upper_mcr, np.log10(real_stress))
        lmp_lower_mcr = np.polyval(poly_lower_mcr, np.log10(real_stress))

        upper = real_stress >= yfn(Trange) * frac
        lower = np.logical_not(upper)

        t = np.zeros_like(real_stress)
        t[upper] = 10.0 ** (lmp_upper[upper] / Trange[upper] - C_upper)
        t[lower] = 10.0 ** (lmp_lower[lower] / Trange[lower] - C_lower)

        rate = np.zeros_like(real_stress)
        rate[upper] = 10.0 ** (lmp_upper_mcr[upper] / Trange[upper] - C_upper_mcr)
        rate[lower] = 10.0 ** (lmp_lower_mcr[lower] / Trange[lower] - C_lower_mcr)

        data[i * nper : (i + 1) * nper, 0] = Trange - 273.15
        data[i * nper : (i + 1) * nper, 1] = real_stress
        data[i * nper : (i + 1) * nper, 2] = t
        data[i * nper : (i + 1) * nper, 3] = rate

    with open("data.csv", "w") as f:
        f.write("Heat/Lot ID,Temp (C),Stress (MPa),Life (h),Creep rate (%/hr)\n")
        for h, d in zip(heats, data):
            f.write(h + "," + ",".join(map(str, d)) + "\n")
