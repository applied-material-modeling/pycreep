#!/usr/bin/env python3

import sys

sys.path.append("..")

from pycreep import data, wilshire, time_independent

if __name__ == "__main__":
    # Tensile strength as a function of temperature: strength = -0.4 * T + 550 with T in Celsius and strength in MPa
    tensile_strength = time_independent.UserPolynomialTimeIndependentCorrelation(
        [-0.4, 550]
    )

    # Load creep data
    df = data.load_data_from_file("quadratic.csv")

    # Create Wilshire correlation
    model = wilshire.WilshireAnalysis(tensile_strength, df).analyze()

    # Write to file
    model.excel_report("wilshire.xlsx")
