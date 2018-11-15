#!/usr/bin/env python
import warnings

from gsw import rho, SP_from_C, SA_from_SP, CT_from_t


def calculate_practical_salinity(conductivity, temperature, pressure):
    """Calculates practical salinity given glider conductivity, temperature,
    and pressure using Gibbs gsw SP_from_C function.

    Parameters:
        conductivity (S/m), temperature (C), and pressure (dbar).

    Returns:
        salinity (psu PSS-78).
    """

    correct_sizes = (
        conductivity.size == temperature.size == pressure.size
    )
    if correct_sizes is False:
        raise ValueError('Arguments must all be the same length')

    # Convert S/m to mS/cm
    mS_conductivity = conductivity * 10

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SP_from_C(
            mS_conductivity,
            temperature,
            pressure
        )


def calculate_density(temperature, pressure, salinity, latitude, longitude):
    """Calculates density given glider practical salinity, pressure, latitude,
    and longitude using Gibbs gsw SA_from_SP and rho functions.

    Parameters:
        temperature (C), pressure (dbar), salinity (psu PSS-78),
        latitude (decimal degrees), longitude (decimal degrees)

    Returns:
        density (kg/m**3),
    """

    correct_sizes = (
        temperature.size == pressure.size == salinity.size == latitude.size == longitude.size
    )
    if correct_sizes is False:
        raise ValueError('Arguments must all be the same length')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        absolute_salinity = SA_from_SP(
            salinity,
            pressure,
            longitude,
            latitude
        )

        conservative_temperature = CT_from_t(
            absolute_salinity,
            temperature,
            pressure
        )

        density = rho(
            absolute_salinity,
            conservative_temperature,
            pressure
        )

        return density
