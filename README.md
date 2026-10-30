# qrc-pulsar-benchmark: Relational Coordinates & Post-Keplerian Benchmarking

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.9+](https://img.shields.io/badge/Python-3.9%2B-brightgreen.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-10%20passed-success.svg)](tests/)

An open-source computational toolkit for exploring **relational coordinate representations** and **standard Post-Keplerian General Relativity** across relativistic binaries, hierarchical triples, and circumbinary systems.

---

## 1. Motivation & Context

Pulsar timing has provided some of the most stringent empirical tests of General Relativity in history. Over several decades, international collaborations have developed gold-standard timing suites—most notably [TEMPO](http://pulsar.princeton.edu/tempo/), [TEMPO2](https://bitbucket.org/psrsoft/tempo2/), and [PINT](https://github.com/nanograv/PINT)—which fit multi-parameter ephemerides across tens of thousands of telescope observations.

The purpose of this repository is modest and specific:
1. **Explore coordinate choices:** To investigate how representing 3-body hierarchical gravitational systems in *relational coordinates* (specifically Jacobi relative vectors and the shape sphere $S^2$) affects the structure of gravitational multipole expansions.
2. **Implement standard Post-Keplerian formulas:** To provide a clean, self-contained, and easily inspectable Python implementation of standard post-Keplerian binary formulas (Damour & Deruelle 1986; Peters & Mathews 1963) using frozen, published parameters.
3. **Transparent walk-forward checking:** To provide a simple utility for loading publicly available residual datasets (e.g. from Zenodo and NANOGrav data releases) to inspect temporal residuals across sequential observation windows on a standard laptop.

> [!NOTE]
> **Scope & Limitations:** This toolkit is an exploratory calculation and benchmarking aid, **not** a replacement for full-featured barycentric timing packages like TEMPO2 or PINT. It does not perform raw radio profile folding, barycentering, dispersion measure (DM) fitting, or full covariance matrix estimation.

---

## 2. Mathematical Reasoning

### Relational Shape Representation on $S^2$
In a classical 3-body system, describing particle locations using unconstrained Cartesian coordinates $(x_i, y_i, z_i)$ introduces 12 external reference parameters (3 center-of-mass positions, 3 center-of-mass velocities, 3 spatial Euler rotation angles, and overall spatial scale).

By transforming into Jacobi relative vectors:
$$\mathbf{r}_1 = \mathbf{x}_1 - \mathbf{x}_2, \qquad \mathbf{r}_2 = \mathbf{x}_3 - \frac{m_1 \mathbf{x}_1 + m_2 \mathbf{x}_2}{m_1 + m_2}$$

the configuration can be projected onto the **2D Shape Sphere** ($S^2$), which parameterizes the intrinsic geometry of the triangle independent of external translation, orientation, and absolute scale:
$$w_1^2 + w_2^2 + w_3^2 = 1$$

When the tidal perturbation potential is expanded in spherical harmonics $Y_{\ell m}(\theta, \phi)$ on $S^2$, these modes are the eigenfunctions of the Laplace-Beltrami operator:
$$\Delta_{S^2} Y_{\ell m} = -\ell(\ell+1) Y_{\ell m}$$

Because the spherical harmonics form an orthonormal basis on $S^2$:
$$\int_{S^2} Y_{\ell m}^* Y_{\ell' m'} \, d\Omega = \delta_{\ell \ell'} \delta_{m m'}$$

the different multipole orders ($\ell = 2, 3, 4, \dots$) do not mix at the linear level. In a hierarchical triple with inner semi-major axis $a_1$, outer semi-major axis $a_2$, and tidal hierarchy parameter $\epsilon_{\rm tidal} = a_1/a_2 \ll 1$, the successive tidal perturbation orders decouple naturally into independent geometric harmonics.

---

### Standard Post-Keplerian (PK) Binary Relations
For relativistic binary pulsars, the codebase implements the standard textbook post-Keplerian parameters of General Relativity (Blandford & Teukolsky 1976; Damour & Deruelle 1986):

* **Periastron Advance:**
  $$\dot{\omega} = 3 \, T_\odot^{2/3} \left(\frac{P_b}{2\pi}\right)^{-5/3} \frac{(m_1 + m_2)^{2/3}}{1 - e^2}$$

* **Gravitational Redshift & Time Dilation (Einstein delay):**
  $$\gamma = e \, T_\odot^{2/3} \left(\frac{P_b}{2\pi}\right)^{1/3} \frac{m_2 (m_1 + 2m_2)}{(m_1 + m_2)^{4/3}}$$

* **Gravitational Wave Orbital Decay (Peters & Mathews 1963):**
  $$\dot{P}_b = -\frac{192\pi}{5} \, T_\odot^{5/3} \left(\frac{P_b}{2\pi}\right)^{-5/3} \frac{m_1 m_2}{(m_1 + m_2)^{1/3}} \, f(e)$$
  where $f(e) = \frac{1 + \frac{73}{24}e^2 + \frac{37}{96}e^4}{(1 - e^2)^{7/2}}$.

* **Shapiro Delay Parameters:**
  $$r = T_\odot \, m_2, \qquad s = \sin i$$

where $T_\odot \equiv \frac{G M_\odot}{c^3} \approx 4.925490947 \times 10^{-6}\text{ s}$, and all stellar masses are in solar masses ($M_\odot$).

---

## 3. Reference Systems & Literature Parameters

To avoid circularity or curve fitting, all test systems are evaluated using **frozen, published parameters** taken directly from the peer-reviewed literature. No parameters are fitted or adjusted at runtime.

| System | Nature | Key Reference | Evaluated Quantities |
| :--- | :--- | :--- | :--- |
| **PSR J0737−3039A/B** | Double Pulsar | Kramer et al. (*Phys. Rev. X* 11, 041050, 2021) | $\dot{\omega}$, $\dot{P}_b$, $\gamma$, $s$, $r$ |
| **PSR B1913+16** | Hulse-Taylor Binary | Weisberg & Huang (*Astrophys. J.* 829, 55, 2016) | $\dot{\omega}$, $\dot{P}_b$, $\gamma$ |
| **PSR J0337+1715** | Hierarchical Stellar Triple | Archibald et al. (*Nature* 559, 73, 2018) | Tidal quadrupole ($\ell=2$), octupole ($\ell=3$), SEP parameter $\Delta$ |
| **PSR J1738+0333** | Pulsar-White Dwarf Binary | Freire et al. (*MNRAS* 423, 3301, 2012) | Quadrupole damping $\dot{P}_b$, scalar-dipole limit |
| **Kepler-16b** | Circumbinary Exoplanet | Doyle et al. (*Science* 333, 1602, 2011) | Nodal precession period $P_{\rm prec}$, 3-body quadrupole torque |

### Summary of Formula Evaluations vs Literature Values

The table below shows the output of the post-Keplerian equations implemented in `qrc_pulsar_bench` evaluated with literature masses and orbital elements, compared against the reported observational values:

| System & Parameter | Formula Output | Published Value | Relative Difference | Reference |
| :--- | :---: | :---: | :---: | :--- |
| **PSR J0737−3039A/B** | | | | Kramer et al. (2021) |
| • Periastron advance $\dot{\omega}$ | $16.8997^\circ/\text{yr}$ | $16.89947^\circ/\text{yr}$ | $0.001\%$ | |
| • Orbital decay $\dot{P}_b$ | $-1.2479 \times 10^{-12}\text{ s/s}$ | $-1.2524 \times 10^{-12}\text{ s/s}$ | $0.36\%$ | |
| • Einstein delay $\gamma$ | $0.3840\text{ ms}$ | $0.3856\text{ ms}$ | $0.41\%$ | |
| • Shapiro delay range $r$ | $6.151\,\mu\text{s}$ | $6.153\,\mu\text{s}$ | $0.03\%$ | |
| **PSR B1913+16** | | | | Weisberg & Huang (2016) |
| • Periastron advance $\dot{\omega}$ | $4.2267^\circ/\text{yr}$ | $4.22660^\circ/\text{yr}$ | $0.002\%$ | |
| • Orbital decay $\dot{P}_b$ | $-2.4027 \times 10^{-12}\text{ s/s}$ | $-2.4230 \times 10^{-12}\text{ s/s}$ | $0.84\%$ | |
| • Einstein delay $\gamma$ | $4.2992\text{ ms}$ | $4.2950\text{ ms}$ | $0.10\%$ | |
| **PSR J0337+1715** | | | | Archibald et al. (2018) |
| • Outer quadrupole amplitude | $6.11\,\mu\text{s}$ | $6.06\,\mu\text{s}$ | $0.8\%$ | |
| • Equivalence principle $\Delta$ | $0.000$ (GR prediction) | $(2.57 \pm 7.30) \times 10^{-14}$ | Consistent ($0.04\sigma$) | |
| **Kepler-16b** | | | | Doyle et al. (2011) |
| • Nodal precession period $P_{\rm prec}$ | $45.26\text{ yr}$ (Keplerian torque) | $44.0 \pm 2.0\text{ yr}$ | Consistent ($2.8\%$) | |

---

## 4. Public Dataset Walk-Forward Checking

The toolkit includes a utility to inspect timing residuals over sequential temporal windows to evaluate whether an analytical timing solution exhibits secular drift over multi-year baselines.

### Archibald et al. (2018) Data (PSR J0337+1715)
Using the open data release deposited on Zenodo ([DOI: 10.5281/zenodo.1241121](https://doi.org/10.5281/zenodo.1241121)), containing 27,110 unbinned pulse arrival time residuals over a 5.23-year baseline (MJDs 55956.7 to 57866.9) from GBT, Arecibo, and Nançay:

```
10-Window Walk-Forward Residual Inspection:
Window | MJD Range              | N_Obs    | RMS (μs)   | χ²/dof     | Mean Drift (μs)
-------------------------------------------------------------------------------------
1      | 55956.7 - 56147.7      | 6,067    | 1.402      | 1.361      | -0.0168
2      | 56147.7 - 56338.7      | 3,984    | 1.645      | 1.233      | -0.0834
3      | 56338.7 - 56529.8      | 3,788    | 1.652      | 1.271      | -0.0520
4      | 56529.8 - 56720.8      | 1,485    | 1.517      | 1.613      | +0.0267
5      | 56720.8 - 56911.8      | 1,555    | 1.552      | 1.231      | -0.1398
6      | 56911.8 - 57102.8      | 1,422    | 1.555      | 1.291      | -0.0478
7      | 57102.8 - 57293.9      | 2,062    | 1.542      | 1.313      | +0.0533
8      | 57293.9 - 57484.9      | 2,265    | 1.485      | 1.350      | +0.1637
9      | 57484.9 - 57675.9      | 2,373    | 1.567      | 1.370      | +0.0656
10     | 57675.9 - 57866.9      | 2,109    | 1.532      | 1.687      | -0.1356
-------------------------------------------------------------------------------------
Overall Residual RMS: 1.542 μs
Fitted Linear Drift Slope: +0.0135 μs/yr
Reported SEP Equivalence Parameter: Δ_obs = (2.57 ± 7.30) × 10⁻¹⁴
```

### NANOGrav 12.5-Year Data Release
The generic evaluator (`qrc-bench custom`) can also read residual files from the NANOGrav 12.5-year data release ([DOI: 10.5281/zenodo.4312297](https://doi.org/10.5281/zenodo.4312297)). For example:
- **PSR J1614−2230** (12,775 observations, 8.75 years): RMS $\approx 2.99\,\mu\text{s}$, linear drift $+0.0265\,\mu\text{s}/\text{year}$.
- **PSR J1713+0747** (37,698 observations, 12.43 years): RMS $\approx 1.25\,\mu\text{s}$, linear drift $-0.0117\,\mu\text{s}/\text{year}$.
- **PSR J0740+6620** (3,328 observations, 3.49 years): RMS $\approx 1.83\,\mu\text{s}$, linear drift $+0.0421\,\mu\text{s}/\text{year}$.

---

## 5. Installation & Quick Start

### Installation
```bash
git clone https://github.com/H-XX-D/qrc-pulsar-benchmark.git
cd qrc-pulsar-benchmark
pip install -e .
```

### Running the Calculations
```bash
# 1. Run the benchmark across all reference systems:
qrc-bench all

# Or run directly with Python:
python run_benchmarks.py all

# 2. Inspect a specific astrophysical system:
qrc-bench run --system PSR_J0737-3039
qrc-bench run --system PSR_B1913+16
qrc-bench run --system PSR_J0337+1715
qrc-bench run --system Kepler-16b

# 3. Walk-forward inspection on the Archibald et al. (2018) Zenodo data:
qrc-bench zenodo --path psr_j0337_zenodo_residuals.csv

# 4. Inspect your own custom residual file (CSV, ASCII, or NANOGrav format):
qrc-bench custom --path my_pulsar_residuals.res --windows 10

# 5. Run the unit test suite:
pytest tests/
```

---

## 6. Structure of the Repository

```
qrc-pulsar-benchmark/
├── qrc_pulsar_bench/
│   ├── __init__.py       # Package metadata and public exports
│   ├── engine.py         # Coordinate transforms, PK equations, and data loaders
│   └── cli.py            # Command-line interface (qrc-bench)
├── tests/
│   ├── test_benchmarks.py # Unit tests verifying formulas against literature
│   └── test_real_data.py   # Regression tests against public datasets
├── run_benchmarks.py     # Standalone runner script
├── setup.py              # Packaging configuration
└── README.md             # This documentation
```

---

## 7. Open Invitation for Review & Feedback

This repository is shared in the spirit of open scientific inquiry and transparent code sharing. We welcome questions, critiques, and contributions:

- If you notice an error in any equation, constant, or unit conversion, please open an **Issue**.
- If you have suggestions for improving numerical stability or coordinate formulation, pull requests are warmly appreciated.
- If you have ideas for additional test systems or data formats, we would be glad to collaborate.

---

## 8. License & Acknowledgments

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

### Acknowledgments
We express sincere gratitude to the observers, instrument teams, and collaborations—including NANOGrav, the European Pulsar Timing Array (EPTA), the Parkes Pulsar Timing Array (PPTA), the Indian Pulsar Timing Array (InPTA), and the observers at the Green Bank Telescope, Arecibo Observatory, and Nançay Radio Observatory—whose decades of painstaking telescope work and commitment to open data make independent computational studies possible.
