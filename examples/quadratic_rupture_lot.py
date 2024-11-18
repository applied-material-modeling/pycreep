#!/usr/bin/env python3

import sys

sys.path.append("..")

from pycreep import data, ttp

if __name__ == "__main__":
    df = data.load_data_from_file("quadratic.csv")

    param = ttp.LarsonMillerParameter()
    order = 2

    centered = ttp.LotCenteredAnalysis(param, order, df).analyze()

    centered.excel_report("centered.xlsx")
