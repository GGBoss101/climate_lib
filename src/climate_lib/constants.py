"""
This module contains constants for common use included in the library.

Attributes:
    g_earth (float): Earth's gravitational acceleration (m/s^2)
    R_earth (float): Earth's radius (m)
    P0pa (float): Standard pressure at sea level (Pa)
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