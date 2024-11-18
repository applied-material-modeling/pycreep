"""Basic functions for dealing with units, using the pint library
"""

import pint

ureg = pint.UnitRegistry()


def convert(data, in_units, out_units):
    """
    Convert some data from in_units to out_units

    Args:
        data:       input numerical data
        in_units:   input units, i.e. the units of data
        out_units:  desired units

    Returns:
        data converted to out_units
    """
    return (data * ureg(in_units)).to(ureg(out_units)).magnitude
