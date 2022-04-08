#!/usr/bin/env python3

import sys
sys.path.append('../..')

from pycreep import data, ttp

if __name__ == "__main__":
    df = data.load_data_from_file("304H-rupture.csv")

    param = ttp.LarsonMillerParameter()
    order = 2

    uncentered = ttp.UncenteredAnalysis(param, order, df).analyze()
    centered = ttp.LotCenteredAnalysis(param, order, df).analyze()
    
    print(centered.report())
