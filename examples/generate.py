#!/usr/bin/env python3

import numpy as np
import scipy.stats as ss
import string
import itertools

heat_names = list(map(lambda x: "".join(x), itertools.product([""]+
    list(string.ascii_lowercase), repeat = 2)))[1:]

if __name__ == "__main__":
    nheats = 20
    nper = 15
    
    srange = np.linspace(100.0,200.0,nper)[::-1]
    Trange = np.linspace(500,750.0,nper) + 273.15
    real_poly = [-900.0, -2300.0, 27000.0]
    C_mean = 16.0
    C_scale = 0.75

    noise = 30.0

    dist = ss.norm(C_mean, C_scale)

    heats = []
    data = np.zeros((nheats*nper,3))
    for i in range(nheats):
        heats.extend([heat_names[i]]*nper)
        # Generate real C
        C = dist.rvs()

        real_stress = ss.norm(srange, noise).rvs()

        lmp = np.polyval(real_poly, np.log10(real_stress))
        t = 10.0**(lmp / Trange - C)
        
        data[i*nper:(i+1)*nper,0] = Trange - 273.15
        data[i*nper:(i+1)*nper,1] = real_stress
        data[i*nper:(i+1)*nper,2] = t

    with open("data.csv", 'w') as f:
        f.write("Heat/Lot ID,Temp (C),Stress (MPa),Life (h)\n")
        for h, d in zip(heats, data):
            f.write(h+","+",".join(map(str, d))+"\n")
