import math
import numpy as np

"""
A set of voltage functions that can be used as the "voltageFunction" parameter in a electrochemistry simulation
These enable the applied voltage to be changed during the simulation
Each of these functions must take time and state as the first arguments, even if they are not used
"""

def cosVoltage(time, state, period, amplitude, offset):
    """Determine the voltage to apply based on a cosine with the specified period and amplitude"""

    cosValue = time * 2 * math.pi / period

    return amplitude * math.cos(cosValue) + offset


def constantVoltage(time, state, value):
    """Output a constant voltage regardless of time and state"""
    return value