#!/usr/bin/env python3

import sys

sys.path.append("..")

from pycreep import data, ttp

if __name__ == "__main__":
    df = data.load_data_from_file("quadratic.csv")

    param = ttp.LarsonMillerParameter()
    order = 2

    uncentered = ttp.UncenteredAnalysis(param, order, df).analyze()

    uncentered.excel_report("uncentered.xlsx")
