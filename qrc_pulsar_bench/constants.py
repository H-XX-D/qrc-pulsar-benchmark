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

# IAU 2015 Resolution B3 nominal solar mass parameter (exact conversion value).
# Prsa et al. 2016, https://arxiv.org/abs/1605.09788
# Use GM directly in orbital formulas rather than multiplying rounded G and M.
GM_sun = 1.3271244e20  # m^3 s^-2
T_sun = GM_sun / c**3 # seconds
M_sun = GM_sun / G    # kg, derived using the adopted G

# Astronomical Unit (IAU standard, exact)
AU = 149597870700.0    # m

# Temporal Units
day_s = 86400.0        # seconds in a day
year_s = 365.25 * day_s # seconds in a Julian year

# Legacy dimensional quantity; unused by the calculators. Its definition alone
# supplies no proof of unitarity, boundary matching, or the equivalence principle.
Z_H = (c**3) / G       # kg s^-1 (~ 4.037e35 kg/s)
