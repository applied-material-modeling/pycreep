#!/usr/bin/env python3

import sys
sys.path.append('../../')

import os.path

from pycreep import ttp, data

def write_report(fname, res):
    with open(fname, 'w') as f:
        f.write("Regression results:\n")
        f.write("Coef.\t\tValue\n")
        for i,p in enumerate(res["polyavg"][::-1]):
            f.write("a%i\t\t%.10e\n" % (i,p))
        f.write("Overall C\t%.10e\n" % res["C_avg"])
        f.write("\n")
        f.write("Statistics:\n")
        f.write("R2\t\t%.10e\n" % res["R2_heat"])
        f.write("SEE\t\t%.10e\n" % res["SEE_heat"])
        f.write("\n")
        f.write("Heat summary:\n")
        f.write("%28s\tCount\tLot C\t\t\tLot RMS error\n" % "Heat")
        for heat in sorted(res["C_heat"].keys()):
            f.write("%28s\t%i\t%.10e\t%.10e\n" % 
                    (heat, res["heat_count"][heat], res["C_heat"][heat],
                        res["heat_rms"][heat]))

if __name__ == "__main__":
    data_sources = {
            "TD-DA-13-12-20-316H-rupture-Customary.csv": 
            (["Life (h)", "Temp (R)", "Stress (ksi)", "Heat/Lot ID",
                "R", "ksi", "h", "R", "ksi", "h"], 2),
            "TD-DA-13-12-20-316H-rupture-SI.csv": 
            (["Life (h)", "Temp (K)", "Stress (MPa)", "Heat/Lot ID",
                "K", "MPa", "h", "K", "MPa", "h"], 2),
            "TD-DA-13-12-20-316H-t1%-Customary.csv": 
            (["Time to 1% (h)", "Temp (R)", "Stress (ksi)", "Heat/Lot ID",
                "R", "ksi", "h", "R", "ksi", "h"], 2),
            "TD-DA-13-12-20-316H-t1%-SI.csv": 
            (["Time to 1% (h)", "Temp (K)", "Stress (MPa)", "Heat/Lot ID",
                "K", "MPa", "h", "K", "MPa", "h"], 2),
            }

    
    for fname, (args, order) in data_sources.items():
        df = data.load_data_from_file(fname)
        analysis = ttp.LMPAnalysis(df, *args)
        res = analysis.polynomial_analysis(order, lot_centering = True)

        outfile, ext = os.path.splitext(fname)
        outfile += "-report.txt"
        write_report(outfile, res) 
