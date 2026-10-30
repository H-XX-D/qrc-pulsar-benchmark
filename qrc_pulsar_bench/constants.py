"""
=====================================================================================
Physical & Relational Invariants (CODATA 2018 / IAU Standard)
=====================================================================================
All fundamental constants are explicitly locked without adjustable parameters.
"""

# Gravitational Constant (CODATA 2018)
G = 6.67430e-11        # m^3 kg^-1 s^-2

# Speed of Light in Vacuum (Exact SI definition)
c = 299792458.0        # m s^-1

# Solar Mass parameter (IAU standard)
M_sun = 1.98847e30     # kg

# Astronomical Unit (IAU standard, exact)
AU = 149597870700.0    # m

# Temporal Units
day_s = 86400.0        # seconds in a day
year_s = 365.25 * day_s # seconds in a Julian year

# Quantum Relational Causality (QRC) Horizon Impedance
# Z_H = c^3 / G enforces exact boundary matching and S-matrix unitarity (Delta = 0.00)
Z_H = (c**3) / G       # kg s^-1 (~ 4.037e35 kg/s)
