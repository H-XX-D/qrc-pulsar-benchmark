# qrc-pulsar-benchmark: Relativistic Binary Post-Keplerian Calculator & 3-Body Dynamics Toolkit

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.9+](https://img.shields.io/badge/Python-3.9%2B-brightgreen.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-22%20passed-success.svg)](tests/)

An open-source Python educational and benchmarking toolkit implementing:
1. **Standard textbook Post-Keplerian General Relativity** for relativistic binaries.
2. **Relational 3-body orbital dynamics** in gauge-free Jacobi coordinates with continuous Montgomery-Dragt shape sphere ($S^2$) trajectory tracking.
3. **Side-by-side benchmark comparison** between relational relative ODE integration and 18-variable Cartesian ODE integration.
4. **Descriptive inspection utilities** for published pulsar timing residuals (Archibald et al. 2018; NANOGrav 12.5-year data release).

---

## 1. Scope, Limitations & Scientific Clarifications

To ensure complete scientific transparency and avoid overstatement, here is an explicit statement of what this repository does and does not do:

### What This Toolkit Is
* **A clean textbook GR calculator:** Evaluates standard post-Keplerian formulas (Damour & Deruelle 1986; Peters & Mathews 1963) for relativistic binary pulsars using published parameters from the observational literature.
* **A true time-dependent 3-body integrator:** Solves the coupled differential equations of motion in relative Jacobi coordinates $(\mathbf{r}_1, \mathbf{v}_1, \mathbf{r}_2, \mathbf{v}_2)$ using high-order adaptive Runge-Kutta (`DOP853`), tracking instantaneous energy conservation, Montgomery-Dragt shape coordinates on $S^2$, and projected pulsar Roemer timing delays.
* **A gauge-elimination benchmark:** Demonstrates that formulating 3-body dynamics in Jacobi relative vectors eliminates 6 translational degrees of freedom, maintaining an identically zero center-of-mass drift ($0.0\text{ m}$) compared to the accumulating truncation drift of 18-variable Cartesian integrators.
* **A residual inspection utility:** Provides scripts to load and bin published residual time series (e.g. Archibald et al. 2018; NANOGrav 12.5-year data release) using true RMS about zero ($\sqrt{\frac{1}{N}\sum r_i^2}$) and sample standard deviation.

### What This Toolkit Is Not
* **Does NOT claim to eliminate three-body chaos or solve the general non-integrable 3-body problem:** Poincaré's theorem (1890) proved that general, non-hierarchical 3-body systems are non-integrable. While relational coordinates eliminate unphysical external coordinate gauge redundancies, the physical chaotic dynamics remain.
* **Does NOT prove or provide new evidence that $\Delta = 0$:** Evaluating the Archibald et al. (2018) Zenodo residuals does not independently test the Strong Equivalence Principle. The Zenodo data file contains *post-fit* residuals from a 40-parameter TEMPO fit that *already converged with $\Delta$ included*. 
* **Covariance note on $\Delta$:** Projecting post-fit residuals onto the $\Delta$ derivative column while holding the other 39 parameters fixed yields an artificial formal uncertainty of $\sigma \sim 10^{-12}$. When accounting for the full parameter covariance matrix and red noise, the true marginalized uncertainty is $\sigma = 4.9 \times 10^{-6}$ (Archibald et al. 2018).
* **Walk-Forward limitations:** Running a naive linear drift check across all 47 pulsars in the NANOGrav 12.5-year release flags **23 of 47 pulsars as "DRIFT DETECTED"**. This drift is astrophysical (interstellar dispersion measure variations and intrinsic pulsar spin noise), not a failure of GR. Furthermore, a straight-line drift test is blind to quadratic phase signals (such as orbital period decay $\Delta t \propto \frac{1}{2}\dot{P}_b t^2$).

---

## 2. Standard Post-Keplerian (PK) Binary Relations

For isolated relativistic binary pulsars, the toolkit implements standard textbook relations of General Relativity:

* **Periastron Advance (Robertson 1938; Damour & Deruelle 1986):**
  $$\dot{\omega} = 3 \, T_\odot^{2/3} \left(\frac{P_b}{2\pi}\right)^{-5/3} \frac{(m_1 + m_2)^{2/3}}{1 - e^2}$$

* **Gravitational Wave Orbital Decay (Peters & Mathews 1963):**
  $$\dot{P}_b = -\frac{192\pi}{5} \, T_\odot^{5/3} \left(\frac{P_b}{2\pi}\right)^{-5/3} \frac{m_1 m_2}{(m_1 + m_2)^{1/3}} \, f(e)$$
  where $f(e) = \frac{1 + \frac{73}{24}e^2 + \frac{37}{96}e^4}{(1 - e^2)^{7/2}}$.

* **Einstein Delay (Gravitational Redshift + Transverse Doppler):**
  $$\gamma = e \, T_\odot^{2/3} \left(\frac{P_b}{2\pi}\right)^{1/3} \frac{m_2 (m_1 + 2m_2)}{(m_1 + m_2)^{4/3}}$$

* **Shapiro Delay Parameters:**
  $$r = T_\odot \, m_2, \qquad s = \sin i$$

where $T_\odot \equiv \frac{G M_\odot}{c^3} \approx 4.925490947 \times 10^{-6}\text{ s}$, and all stellar masses are expressed in solar masses ($M_\odot$).

---

## 3. Reference Systems & Empirical Comparisons

Parameters are sourced directly from peer-reviewed literature.

| System & Parameter | Formula Output | Published Value | Agreement | Reference | Notes |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **PSR J0737−3039A/B** | | | | Kramer et al. (2021) | Double Pulsar |
| • Periastron advance $\dot{\omega}$ | $16.8997^\circ/\text{yr}$ | $16.89932^\circ/\text{yr}$ | $99.998\%$ | | |
| • Orbital decay $\dot{P}_b$ | $-1.2479 \times 10^{-12}$ | $-1.24792 \times 10^{-12}$ | $99.998\%$ | | Intrinsic decay |
| • Einstein delay $\gamma$ | $0.3840\text{ ms}$ | $0.38405\text{ ms}$ | $99.99\%$ | | |
| • Shapiro delay range $r$ | $6.151\,\mu\text{s}$ | $6.151\,\mu\text{s}$ | $100.00\%$ | | |
| **PSR B1913+16** | | | | Weisberg & Huang (2016) | Hulse-Taylor |
| • Periastron advance $\dot{\omega}$ | $4.2267^\circ/\text{yr}$ | $4.22660^\circ/\text{yr}$ | $99.998\%$ | | |
| • Intrinsic orbital decay $\dot{P}_b^{\rm int}$ | $-2.4027 \times 10^{-12}$ | $-2.398 \times 10^{-12}$ | $99.80\%$ | Damour & Taylor (1991) | Corrected for Galactic acceleration |
| • Observed orbital decay $\dot{P}_b^{\rm obs}$ | $-2.4027 \times 10^{-12}$ | $-2.423 \times 10^{-12}$ | $99.16\%$ | Weisberg & Huang (2016) | Includes Galactic contribution |
| • Einstein delay $\gamma$ | $4.2992\text{ ms}$ | $4.2950\text{ ms}$ | $99.90\%$ | | |
| **PSR J1738+0333** | | | | Freire et al. (2012) | Pulsar-WD Binary |
| • Quadrupole decay $\dot{P}_b$ | $-2.75 \times 10^{-14}$ | $-2.59 \times 10^{-14}$ | Order of mag | | Constrains scalar dipole radiation |
| **PSR J0337+1715** | | | | Archibald et al. (2018) | Hierarchical Triple |
| • Outer quadrupole amplitude | $6.11\,\mu\text{s}$ | $6.06\,\mu\text{s}$ | Scaling | | Secular 3-body tidal torque |
| • Published SEP bound $\Delta$ | — | $(-0.6 \pm 4.9) \times 10^{-6}$ | Consistent | | Full 40-param TEMPO fit |
| **Kepler-16b** | | | | Doyle et al. (2011) | Circumbinary Planet |
| • Nodal precession period $P_{\rm prec}$ | $45.26\text{ yr}$ | $44.0 \pm 2.0\text{ yr}$ | $97.1\%$ | | Secular quadrupole torque |

---

## 4. Relational Dynamics & The Shape Sphere ($S^2$)

### The Geometric Montgomery-Dragt Map
The toolkit implements the standard **Montgomery-Dragt shape map** (Dragt 1965; Montgomery 2002) mapping triangle side lengths $(a, b, c)$ to the unit 2-sphere $S^2$:

$$R^2 = a^2 + b^2 + c^2$$
$$w_1 = \frac{2c^2 - a^2 - b^2}{R^2}, \qquad w_2 = \frac{\sqrt{3}(a^2 - b^2)}{R^2}, \qquad w_3 = \frac{4\sqrt{3}\,\text{Area}}{R^2}$$

By construction from Heron's formula for the triangle area:
$$w_1^2 + w_2^2 + w_3^2 \equiv 1$$

* **Equator ($w_3 = 0$):** All collinear degenerate triangle configurations ($\text{Area} = 0$).
* **Poles ($w_3 = \pm 1$):** Equilateral triangles.

### Time-Dependent Relational Integration
The `dynamics` module integrates the 12 coupled differential equations for relative Jacobi vectors:
$$\ddot{\mathbf{r}}_1 = \mathbf{a}_2 - \mathbf{a}_1$$
$$\ddot{\mathbf{r}}_2 = \mathbf{a}_3 - \frac{m_1 \mathbf{a}_1 + m_2 \mathbf{a}_2}{m_1 + m_2}$$

Along the trajectory:
1. **Gauge Invariance:** Center-of-mass motion is eliminated by construction ($\mathbf{x}_{\rm COM} \equiv 0.0\text{ m}$ for all time).
2. **Sphere Tracking:** At every time step, $(w_1, w_2, w_3)$ lie on the unit sphere to machine precision ($< 10^{-14}$).
3. **Pulsar Roemer Delay:** The physical projected light travel time delay is calculated directly from the trajectory:
   $$\Delta t_{\rm Roemer}(t) = -\frac{\mathbf{x}_1(t) \cdot \hat{\mathbf{n}}}{c}$$

---

## 5. Installation & Usage

### Installation
```bash
git clone https://github.com/H-XX-D/qrc-pulsar-benchmark.git
cd qrc-pulsar-benchmark
pip install -e .
```

### Running the Calculations
```bash
# 1. Run master benchmark across all 5 reference systems:
qrc-bench all

# 2. Inspect a specific system:
qrc-bench run --system PSR_J0737-3039
qrc-bench run --system PSR_B1913+16
qrc-bench run --system PSR_J0337+1715
qrc-bench run --system Kepler-16b

# 3. Run live time-dependent 3-body simulation on the shape sphere S²:
qrc-bench simulate --system PSR_J0337+1715 --orbits 10 --points 500

# 4. Run live side-by-side benchmark (Relational Jacobi vs Cartesian 18-ODE):
qrc-bench compare --orbits 5

# 5. Inspect published Archibald et al. (2018) residuals:
qrc-bench zenodo --path psr_j0337_zenodo_residuals.csv

# 6. Inspect custom or NANOGrav residual files:
qrc-bench custom --path my_residuals.res --windows 10

# 7. Run full automated test suite (22 unit & integration tests):
pytest tests/
```

---

## 6. Structure of the Repository

```
qrc-pulsar-benchmark/
├── qrc_pulsar_bench/
│   ├── __init__.py       # Package metadata and public exports
│   ├── constants.py      # Standard physical and astronomical constants
│   ├── systems.py        # Reference astrophysical systems with literature citations
│   ├── engine.py         # Post-Keplerian equations and residual loaders
│   ├── dynamics.py       # Relational 3-body ODE integrator & Cartesian benchmark
│   ├── shape_sphere.py   # Montgomery-Dragt shape sphere mapping
│   └── cli.py            # Command-line interface (qrc-bench)
├── tests/
│   ├── test_benchmarks.py # Verification tests for PK formulas and shape sphere
│   ├── test_dynamics.py   # Tests for 3-body time evolution, energy, and gauge elimination
│   └── test_cli.py        # Automated CLI execution tests
├── run_benchmarks.py     # Standalone runner
├── setup.py              # Package installation script
└── README.md             # This documentation
```

---

## 7. License & Acknowledgments

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

### Acknowledgments
Grateful acknowledgment is made to the authors of the foundational observational timing papers:
* **PSR J0737−3039:** Kramer et al., *Phys. Rev. X* 11, 041050 (2021).
* **PSR B1913+16:** Weisberg & Huang, *Astrophys. J.* 829, 55 (2016); Damour & Taylor, *Astrophys. J.* 366, 501 (1991).
* **PSR J0337+1715:** Archibald et al., *Nature* 559, 73–76 (2018); Voisin et al., *A&A* 638, A24 (2020).
* **PSR J1738+0333:** Freire et al., *Mon. Not. R. Astron. Soc.* 423, 3301–3311 (2012).
* **Kepler-16b:** Doyle et al., *Science* 333, 1602–1606 (2011).
* The **NANOGrav Collaboration** for public data releases deposited on Zenodo.
* **Shape Dynamics References:** Montgomery, *Ergodic Theory and Dynamical Systems* 22 (2002); Moeckel & Montgomery, *Pacific Journal of Mathematics* 262 (2013).
