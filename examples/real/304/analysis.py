#!/usr/bin/env python3

import sys
sys.path.append('../../../')

import os.path

from pycreep import ttp, data

if __name__ == "__main__":
    data_sources = {
            "TD-DA-13-12-20-304H-rupture-Customary.csv": 
            (["Life (h)", "Temp (R)", "Stress (ksi)", "Heat/Lot ID",
                "R", "ksi", "h", "R", "ksi", "h"], 2),
            "TD-DA-13-12-20-304H-rupture-SI.csv": 
            (["Life (h)", "Temp (K)", "Stress (MPa)", "Heat/Lot ID",
                "K", "MPa", "h", "K", "MPa", "h"], 2),
            "TD-DA-13-12-20-304H-t1%-Customary.csv": 
            (["Time to 1% (h)", "Temp (R)", "Stress (ksi)", "Heat/Lot ID",
                "R", "ksi", "h", "R", "ksi", "h"], 1),
            "TD-DA-13-12-20-304H-t1%-SI.csv": 
            (["Time to 1% (h)", "Temp (K)", "Stress (MPa)", "Heat/Lot ID",
                "K", "MPa", "h", "K", "MPa", "h"], 1),
            }

    
    for fname, (args, order) in data_sources.items():
        df = data.load_data_from_file(fname)
        analysis = ttp.LotCenteredAnalysis(ttp.LarsonMillerParameter(),
                order, df, *args).analyze()
                
        outfile, ext = os.path.splitext(fname)
        outfile += "-report.xlsx"
        analysis.excel_report(outfile)
