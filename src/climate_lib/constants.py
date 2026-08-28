"""
This module contains constants for common use included in the library.

Attributes:
    g_earth (float): Earth's gravitational acceleration (m/s^2)
    R_earth (float): Earth's radius (m)
    P0pa (float): Standard pressure at sea level (Pa)
    conversion_factors (dict): Dictionary of unit conversion factors
        - "hPa_to_Pa": Conversion factor from hectopascals to pascals
        - "Pa_to_hPa": Conversion factor from pascals to hectopascals
        - "K_to_C": Conversion factor from Kelvin to Celsius
        - "C_to_K": Conversion factor from Celsius to Kelvin
        - "m_to_km": Conversion factor from meters to kilometers
        - "km_to_m": Conversion factor from kilometers to meters
        - "s_to_hr": Conversion factor from seconds to hours
        - "hr_to_s": Conversion factor from hours to seconds
"""

# Earth's gravitational acceleration (m/s^2)
g_earth = 9.80665

# Earth's radius (m)
R_earth = 6371000.0

# Standard pressure at sea level (Pa)
P0pa = 100000.0

# Unit conversion factors
conversion_factors = {
    "hPa_to_Pa": 100.0,
    "Pa_to_hPa": 0.01,
    "K_to_C": -273.15,
    "C_to_K": 273.15,
    "m_to_km": 0.001,
    "km_to_m": 1000.0,
    "s_to_hr": 1/3600.0,
    "hr_to_s": 3600.0,
}