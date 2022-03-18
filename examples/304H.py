#!/usr/bin/env python3

import sys
sys.path.append('../')

from pycreep import data, ttp

if __name__ == "__main__":
    df = data.load_data_from_file("304H-rupture.csv")

    analysis = ttp.LMPAnalysis(df)
    
    res_uncentered = analysis.polynomial_analysis(2)
    res_centered = analysis.polynomial_analysis(2, lot_centering = True)

    print(res_centered)
