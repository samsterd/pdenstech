import math
import numpy as np
import scipy.signal

"""
A set of voltage functions that can be used as the "voltageFunction" parameter in a electrochemistry simulation
These enable the applied voltage to be changed during the simulation
Each of these functions must take time and state as the first arguments, even if they are not used
"""

def cosVoltage(time, state, period, amplitude, offset):
    """Determine the voltage to apply based on a cosine with the specified period and amplitude
    """

    cosValue = time * 2 * math.pi / period

    return amplitude * math.cos(cosValue) + offset

def sinVoltage(time, state, period, amplitude, offset, phase = 0):
    """Determine the voltage to apply based on a sine with the specified period, amplitude, offset, and phase shift
    """

    sinValue = (time - phase) * 2 * math.pi / period

    return amplitude * math.sin(sinValue) + offset

def constantVoltage(time, state, value):
    """Output a constant voltage regardless of time and state"""

    return value

def equilBeforeSin(time, state, equilTime, equilVoltage, period, amplitude, offset):
    """
    Apply a constant voltage for set time and voltage, then apply a sine voltage

    Args:
        time (float): current time of the simulation
        state (FieldCollection): current state of the concentration fields
        equilTime (float): time from t=0 to t=equilTime to apply a constant voltage
        equilVoltage (float): voltage to apply during equilTime
        period (float): period of the cosine voltage
        amplitude (float): voltage amplitude of cosine
        offset (float): constant added to cosine
    Returns:
        voltage (float)

    The phase of the sine is shifted such that it starts with a value of offset at the end of equilTime
    """
    if time < equilTime:
        return equilVoltage
    else:
        return sinVoltage(time, state, period, amplitude, offset, equilTime)

def squareWave(time, state, period, amplitude, offset, duty, phase = 0):
    """
    Apply a square wave with a changeable duty cycle

    Ars:
    :param time: simulation time
    :param state: FieldCollection of concentration fields
    :param period:
    :param amplitude:
    :param offset:
    :param duty:
    :return: voltage
    """
    squareArg = 2 * (time - phase) * math.pi / period

    return amplitude * scipy.signal.square(squareArg, duty) + offset

def equilBeforeSquare(time, state, equilTime, period, amplitude, offset, duty):
    """
    Equilibrate for a set time at at the duty-weighted average of the square wave, then apply a square wave
    :param time:
    :param state:
    :param equilTime:
    :param equilVoltage:
    :param period:
    :param amplitude:
    :param offset:
    :param duty:
    :return:
    """
    if time < equilTime:
        highVoltage = ((amplitude + offset) * duty)
        lowVoltage = ((offset - amplitude) * (1 - duty))
        return (highVoltage + lowVoltage) / 2
    else:
        return squareWave(time, state, period, amplitude, offset, duty, equilTime)
