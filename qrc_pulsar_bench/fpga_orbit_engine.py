"""Host numerical model of a proposed FPGA orbit datapath.

Cycle counts, initiation interval, DSP/BRAM use and power are design estimates,
not synthesis or device measurements. The NumPy inverse-cube approximation is
not bit-accurate RTL simulation. The original C++ core uses std::sqrt instead.
The separate native_cislunar bridge provides the tested external-force kernel.
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional
import numpy as np

from .constants import G, c, M_sun, AU, day_s
from .systems import HierarchicalTripleSystem, PSR_J0337_1715
from .dynamics import _initial_state


@dataclass(frozen=True)
class FPGASpecification:
    """Target space-grade FPGA hardware specifications."""
    name: str = "AMD/Xilinx Kintex UltraScale XQRKU060"
    clock_frequency_mhz: float = 150.0  # Conservative space-grade clock
    dsp_slices_total: int = 2760
    bram_36k_total: int = 1080
    logic_cells_total: int = 725550
    typical_power_w: float = 3.5


@dataclass(frozen=True)
class HardwareCycleReport:
    """Estimated resource accounting and measured host numerical diagnostics."""
    pipeline_latency_cycles: int
    initiation_interval: int
    clock_frequency_mhz: float
    single_step_latency_ns: float
    max_throughput_steps_per_sec: float
    estimated_dsp_slices: int
    estimated_bram_blocks: int
    dsp_utilization_pct: float
    position_error_vs_fp64_m: Optional[float]
    energy_error: float
    hardware_validated: bool = False


class PipelinedInvCubeUnit:
    """
    Hardware pipeline model for r^-3 = (r^2)^(-1.5).
    Uses a 1024-entry seed LUT followed by 2 pipelined Newton-Raphson iterations:
        y_{n+1} = y_n * (1.5 - 0.5 * x * y_n^2)  [Fast Inv-Sqrt]
    followed by cubing: y^3.
    """

    def __init__(self, lut_bits: int = 10):
        self.lut_bits = lut_bits
        self.lut_size = 1 << lut_bits
        # Seed table for normalized mantissas in [0.5, 2.0]
        x_vals = np.linspace(0.5, 2.0, self.lut_size)
        self.seed_table = 1.0 / np.sqrt(x_vals)

    def evaluate(self, r_sq: np.ndarray) -> np.ndarray:
        """Host float64 approximation of (r_sq)^(-1.5), not RTL emulation."""
        r_sq = np.asarray(r_sq, dtype=np.float64)
        # Avoid zero division
        r_sq = np.maximum(r_sq, 1e-18)

        # Floating-point exponent & mantissa extraction
        mantissa, exp = np.frexp(r_sq)
        # Adjust odd exponents
        odd_exp = (exp % 2) != 0
        exp_adj = np.where(odd_exp, exp - 1, exp)
        mantissa_adj = np.where(odd_exp, mantissa * 2.0, mantissa)

        # LUT indexing on top mantissa bits
        norm_idx = np.clip(
            ((mantissa_adj - 0.5) / 1.5 * (self.lut_size - 1)).astype(int),
            0,
            self.lut_size - 1
        )
        y0 = self.seed_table[norm_idx]

        # Newton-Raphson Iteration 1: y1 = y0 * (1.5 - 0.5 * x * y0^2)
        y1 = y0 * (1.5 - 0.5 * mantissa_adj * (y0**2))
        # Newton-Raphson Iteration 2 (finite approximation error remains)
        y2 = y1 * (1.5 - 0.5 * mantissa_adj * (y1**2))

        # Re-scale exponent: (2^(exp_adj))^(-0.5) = 2^(-exp_adj / 2)
        inv_sqrt = y2 * (2.0 ** (-exp_adj / 2.0))
        # Cube to obtain r^-3
        return inv_sqrt ** 3


class FPGAOrbitEngine:
    """
    Host model and estimated accounting for 3-body Cartesian time-evolution
    and line-of-sight Roemer correction.
    """

    def __init__(
        self,
        system: HierarchicalTripleSystem = PSR_J0337_1715,
        fpga_spec: FPGASpecification = FPGASpecification(),
        order: int = 4
    ):
        self.system = system
        self.spec = fpga_spec
        self.order = order
        self.inv_cube_unit = PipelinedInvCubeUnit()

        # Hardware pipeline stage latencies (clock cycles)
        self.LATENCY_STAGE_SUB = 3      # Pairwise coordinate subtractions r_ij = r_j - r_i
        self.LATENCY_STAGE_RSQ = 4      # Parallel vector dot product x^2 + y^2 + z^2
        self.LATENCY_STAGE_INVCUBE = 16 # Seed LUT + 2 Newton-Raphson stages + cuber
        self.LATENCY_STAGE_ACCEL = 5    # Multiply by G * m_j and vector components
        self.LATENCY_STAGE_STATE = 3    # Leapfrog coordinate accumulation (v += a*dt, p += v*dt)
        self.LATENCY_STAGE_ROEMER = 4   # Line-of-sight projection: dot(n_los, x_pulsar) / c

    def get_hardware_timing(self) -> Tuple[int, int, float]:
        """
        Estimates total pipeline latency and throughput from assumed stage costs.
        
        Returns:
            pipeline_latency: Total clock cycles from input to output.
            initiation_interval: Clock cycles between consecutive step launches (II = 1).
            latency_ns: Single step execution time in nanoseconds.
        """
        # A 4th-order Yoshida step consists of 3 sub-stages of acceleration + coordinate updates
        stages = 3 if self.order == 4 else 1
        single_substep_cycles = (
            self.LATENCY_STAGE_SUB +
            self.LATENCY_STAGE_RSQ +
            self.LATENCY_STAGE_INVCUBE +
            self.LATENCY_STAGE_ACCEL +
            self.LATENCY_STAGE_STATE
        )
        total_latency_cycles = stages * single_substep_cycles + self.LATENCY_STAGE_ROEMER
        initiation_interval = 1  # Fully pipelined
        clock_period_ns = 1000.0 / self.spec.clock_frequency_mhz
        total_latency_ns = total_latency_cycles * clock_period_ns

        return total_latency_cycles, initiation_interval, total_latency_ns

    def run_hardware_simulation(
        self,
        duration_days: float = 10.0,
        dt_s: float = 600.0,
        line_of_sight: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, HardwareCycleReport]:
        """
        Executes a NumPy numerical model; no HLS, RTL or physical device is run.

        Returns:
            times_days: Timestamps array.
            positions_m: Barycentric position trajectories (T, 3, 3).
            roemer_delays_s: Line-of-sight Roemer timing delay (T,).
            report: Hardware resource and cycle accounting report.
        """
        state, period, inputs = _initial_state(self.system, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        m1, m2, m3 = state.m1, state.m2, state.m3
        m_tot = m1 + m2 + m3
        masses = [m1, m2, m3]

        cart = state.to_barycentric_cartesian()
        pos = np.array([cart[0], cart[2], cart[4]], dtype=np.float64)
        vel = np.array([cart[1], cart[3], cart[5]], dtype=np.float64)

        n_steps = int(np.ceil(duration_days * day_s / dt_s))
        times = np.zeros(n_steps)
        positions = np.zeros((n_steps, 3, 3))
        roemer_delays = np.zeros(n_steps)

        # Yoshida 4th-order coefficients
        w1 = 1.0 / (2.0 - 2.0**(1.0 / 3.0))
        w0 = - (2.0**(1.0 / 3.0)) / (2.0 - 2.0**(1.0 / 3.0))
        c_weights = [w1 * 0.5, (w0 + w1) * 0.5, (w0 + w1) * 0.5, w1 * 0.5]
        d_weights = [w1, w0, w1]

        los_norm = np.array(line_of_sight) / np.linalg.norm(line_of_sight)

        def fpga_accel_stage(p):
            """Simulates the FPGA parallel pairwise acceleration pipeline."""
            a_out = np.zeros((3, 3), dtype=np.float64)
            # Evaluate pairwise displacement and r^-3 via pipelined InvCubeUnit
            for i in range(3):
                for j in range(i + 1, 3):
                    r_ij = p[j] - p[i]
                    r_sq = np.sum(r_ij**2)
                    inv_r3 = self.inv_cube_unit.evaluate(r_sq)
                    f_vec = G * inv_r3 * r_ij
                    a_out[i] += f_vec * masses[j]
                    a_out[j] -= f_vec * masses[i]
            return a_out

        # Hardware execution loop
        for step in range(n_steps):
            times[step] = step * dt_s / day_s
            positions[step] = pos

            # Roemer calculation: - (1/c) * dot(los, r_pulsar)
            roemer_delays[step] = - float(np.dot(pos[0], los_norm) / c)

            # Advance symplectic Yoshida-4 pipeline
            for s in range(3):
                pos += c_weights[s] * dt_s * vel
                acc = fpga_accel_stage(pos)
                vel += d_weights[s] * dt_s * acc
            pos += c_weights[3] * dt_s * vel

        # Evaluate energy conservation on chip
        v2 = np.sum(vel**2, axis=-1)
        ke = 0.5 * np.sum(np.array(masses) * v2)
        pe = 0.0
        for i in range(3):
            for j in range(i + 1, 3):
                dist = np.linalg.norm(pos[i] - pos[j])
                pe -= G * masses[i] * masses[j] / dist
        e_final = ke + pe

        # Initial energy
        v2_0 = np.sum(np.array([cart[1], cart[3], cart[5]])**2, axis=-1)
        ke_0 = 0.5 * np.sum(np.array(masses) * v2_0)
        pe_0 = 0.0
        for i in range(3):
            for j in range(i + 1, 3):
                dist = np.linalg.norm(np.array([cart[0], cart[2], cart[4]])[i] - np.array([cart[0], cart[2], cart[4]])[j])
                pe_0 -= G * masses[i] * masses[j] / dist
        e_init = ke_0 + pe_0
        energy_err = abs((e_final - e_init) / e_init)

        # Hardware accounting
        lat_cycles, ii, lat_ns = self.get_hardware_timing()
        # 12 DSPs per pairwise force (3 coordinates x 4) + 12 for leapfrog + 8 for inv-cube = ~56 DSPs
        estimated_dsps = 72
        estimated_brams = 4  # LUT storage for InvCubeUnit and Roemer parameters

        report = HardwareCycleReport(
            pipeline_latency_cycles=lat_cycles,
            initiation_interval=ii,
            clock_frequency_mhz=self.spec.clock_frequency_mhz,
            single_step_latency_ns=lat_ns,
            max_throughput_steps_per_sec=self.spec.clock_frequency_mhz * 1e6 / ii,
            estimated_dsp_slices=estimated_dsps,
            estimated_bram_blocks=estimated_brams,
            dsp_utilization_pct=(estimated_dsps / self.spec.dsp_slices_total) * 100.0,
            position_error_vs_fp64_m=None,  # No reference trajectory comparison performed here
            energy_error=float(energy_err)
        )

        return times, positions, roemer_delays, report
