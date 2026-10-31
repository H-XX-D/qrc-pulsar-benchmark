"""
=====================================================================================
QRC-PULSAR-BENCH: PROJECT ARGONAUT — PROGRAMMATIC LIFECYCLE & FLIGHT TIMELINE
=====================================================================================
Establishes the definitive flight and engineering development timeline for:
  Project ARGONAUT: Autonomous Relational Guidance for Outer-planet Navigation &
                    Automated Unmanned Transit

Anchored on Summer 2027 (July 2027) funding approval and program authorization:
  - Phase A: Concept & Architecture Definition          (Jul 2027 - Jun 2028)
  - Phase B: Preliminary Design & Hardware-in-Loop      (Jul 2028 - Dec 2029)
  - Phase C: Final Design, Fabrication & Qualification  (Jan 2030 - Mar 2032)
  - Phase D: Integration, Environmental Test & Launch   (Apr 2032 - Aug 2033)
  - Phase E1: Earth-to-Jupiter Interplanetary Cruise    (Aug 2033 - Nov 2036)
  - Phase E2: Callisto Station Staging & Commissioning  (Dec 2036 - May 2037)
  - Phase E3: Primary Europa Plume Sampling & Return    (Jun 2037 - Dec 2039)
  - Phase E4: Cargo Tanker Depot Restock                (Jun 2041)
  - Phase E5: Earth GEO Station Capture & Analysis      (Apr 2043)
  - Phase F: Fleet Expansion (Saturn, Uranus, Neptune)  (2042 - 2055+)
=====================================================================================
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List


@dataclass(frozen=True)
class ArgonautMilestone:
    """Detailed metadata for a key programmatic or flight milestone."""
    phase: str
    milestone_name: str
    start_date: str
    end_date: str
    duration_months: int
    deliverables: str
    critical_technology: str
    budget_milestone: str


class ArgonautTimelineManager:
    """
    Manages and schedules programmatic and astrodynamic milestones for Project ARGONAUT.
    """

    # Master sequence of milestones starting Summer 2027
    MILESTONES: List[ArgonautMilestone] = [
        ArgonautMilestone(
            phase="Phase A",
            milestone_name="Architecture Formulation & Requirements Definition",
            start_date="2027-07-01",
            end_date="2028-06-30",
            duration_months=12,
            deliverables="System Requirements Review (SRR), Mission Concept Review (MCR), Tapeout architecture for FPGA/QRC XNAV ASIC",
            critical_technology="Synthesizable FPGA symplectic orbit predictor architecture & CR3BP manifold corridors",
            budget_milestone="Phase A Formulation Budget Authorized ($180M)",
        ),
        ArgonautMilestone(
            phase="Phase B",
            milestone_name="Preliminary Design & Hardware-in-the-Loop Prototyping",
            start_date="2028-07-01",
            end_date="2029-12-31",
            duration_months=18,
            deliverables="Preliminary Design Review (PDR), Breadboard FPGA HIL timing testbed (<50 ns loop), 85 K vacuum laser welding mockups",
            critical_technology="Microgravity quick-disconnect fluid couplers & graded silica aerogel impact cassettes",
            budget_milestone="Phase B Non-Recurring Engineering Confirmation ($420M)",
        ),
        ArgonautMilestone(
            phase="Phase C",
            milestone_name="Final Design, Flight Hardware Fabrication & Vault Qualification",
            start_date="2030-01-01",
            end_date="2032-03-31",
            duration_months=27,
            deliverables="Critical Design Review (CDR Q1 2031), Callisto Depot Module, Reusable Skimmer Rover, 10 Dual-Mode Canisters, GEO Receiver Payload",
            critical_technology="150 krad radiation vault qualification, 3,000 s Isp micro-ion thrusters, Clohessy-Wiltshire flight software",
            budget_milestone="Phase C Prime Manufacturing Contract Execution ($1.65B)",
        ),
        ArgonautMilestone(
            phase="Phase D",
            milestone_name="System Integration, TVAC Testing & Launch Campaign",
            start_date="2032-04-01",
            end_date="2033-08-31",
            duration_months=17,
            deliverables="Thermal-vacuum (TVAC) qualification, Flight Readiness Review (FRR), Dual launch campaign: (1) Callisto Stack to Jupiter, (2) GEO Station Receiver",
            critical_technology="Heavy-lift planetary injection (SLS / Starship / New Glenn), Earth-Jupiter synodic window alignment",
            budget_milestone="Launch Operations & Ground Support Authorization ($650M)",
        ),
        ArgonautMilestone(
            phase="Phase E1",
            milestone_name="Earth-to-Jupiter Autonomous Interplanetary Cruise",
            start_date="2033-08-15",
            end_date="2036-11-30",
            duration_months=39,
            deliverables="Autonomous XNAV cruise across 800M km, 3 closed-loop TCM burns, zero ground DSN navigation tracking",
            critical_technology="Autonomous FPGA pulsar timing navigation & real-time B-plane dispersion control (<2.4 km accuracy)",
            budget_milestone="Cruise Operations & Optical Comms Commissioning ($120M)",
        ),
        ArgonautMilestone(
            phase="Phase E2",
            milestone_name="Jupiter Orbit Insertion & Callisto Depot Commissioning",
            start_date="2036-12-01",
            end_date="2037-05-31",
            duration_months=6,
            deliverables="Jupiter gravity capture, Callisto halo orbit insertion (L1/L2), automated deployment of 85 K glovebox & rover checkout",
            critical_technology="Callisto low-energy manifold capture (delta-v < 70 m/s), radiation haven verification (7.08 rad/hr)",
            budget_milestone="Orbital Depot Operations Authorization ($85M)",
        ),
        ArgonautMilestone(
            phase="Phase E3",
            milestone_name="Primary Sampling Campaign: Europa Plume Skimming & Canister 1 Launch",
            start_date="2037-06-01",
            end_date="2039-12-31",
            duration_months=30,
            deliverables="Sorties 1-5 to Europa plumes (50 km alt, 2,100 m/s), 1.25 kg ocean brine harvested, Canister 1 sealed and launched to Earth (Oct 2038)",
            critical_technology="Graded aerogel shock-free grain capture, 30-min Callisto refueling, robotic laser weld (<5.2e-11 mbar*L/s)",
            budget_milestone="Active Science Sampling Operations ($160M)",
        ),
        ArgonautMilestone(
            phase="Phase E4",
            milestone_name="Earth Logistics Tanker Restock Mission Arrival",
            start_date="2041-06-01",
            end_date="2041-08-31",
            duration_months=3,
            deliverables="Uncrewed cargo tanker arrives at Callisto Station, delivering +300 kg fuel, 15 cassettes, 8 canisters; enables >200 total sorties",
            critical_technology="Autonomous microgravity quick-disconnect fluid transfer & long-duration propellant storage",
            budget_milestone="Depot Restock Campaign Allocation ($280M)",
        ),
        ArgonautMilestone(
            phase="Phase E5",
            milestone_name="Canister 1 Earth GEO Station Capture & Astrobiological Analysis",
            start_date="2043-04-15",
            end_date="2043-06-30",
            duration_months=3,
            deliverables="Canister 1 arrives at Earth after 4.5-yr resonant cruise; Aero-GEO brake scrubs 1,204 m/s; Orbiting Rover soft-latch capture into GEO cleanroom",
            critical_technology="Two-impulse CW relative docking (<0.05 m/s latch), permanent GEO quarantine (>1M yr life), 24/7 optical downlink",
            budget_milestone="GEO Astrobiology Science Lab Operations ($95M/yr)",
        ),
        ArgonautMilestone(
            phase="Phase F",
            milestone_name="Argonaut Fleet Expansion: Saturn, Uranus, & Neptune Missions",
            start_date="2042-01-01",
            end_date="2055-12-31",
            duration_months=168,
            deliverables="Deployment of Argonaut-II (Titan/Enceladus), Argonaut-III (Titania/Miranda), and Argonaut-IV (Triton platform)",
            critical_technology="Reusable fleet architecture standardization, 38 K deep-cryocoolers, outer solar system multi-decade reliability",
            budget_milestone="Outer Solar System Decadal Program Line ($3.2B over 15 yrs)",
        ),
    ]

    @classmethod
    def get_milestones(cls) -> List[ArgonautMilestone]:
        """Return the complete sequence of program milestones."""
        return cls.MILESTONES

    @classmethod
    def generate_timeline_summary(cls) -> str:
        """Format an executive programmatic timeline from 2027 authorization to 2043 sample arrival."""
        lines = [
            "==================================================================================================",
            "PROJECT ARGONAUT: MASTER PROGRAMMATIC FLIGHT & DEVELOPMENT TIMELINE",
            "Authorization Target: Summer 2027 (July 2027) | First Ocean Sample at Earth GEO: April 2043",
            "==================================================================================================",
            f"{'Phase':<10} | {'Dates':<23} | {'Duration':<10} | {'Milestone Name & Core Objectives'}",
            "-----------+-------------------------+------------+-------------------------------------------------------",
        ]
        for m in cls.MILESTONES:
            dates = f"{m.start_date[:7]} to {m.end_date[:7]}"
            dur = f"{m.duration_months} mos"
            lines.append(f"{m.phase:<10} | {dates:<23} | {dur:<10} | {m.milestone_name}")
            lines.append(f"{'':<10} | Deliverables: {m.deliverables[:75]}...")
            lines.append(f"{'':<10} | Tech Enabler: {m.critical_technology[:75]}...")
            lines.append("-----------+-------------------------+------------+-------------------------------------------------------")

        lines.extend([
            "",
            "KEY ASTRODYNAMIC & FLIGHT HIGHLIGHTS:",
            "  1. 2027 - 2033 (6.0 Years)  : Development, FPGA tapeout, 150 krad qualification, and integration.",
            "  2. August 2033              : Launch of Callisto Depot + Rover stack to Jupiter (optimal synodic alignment).",
            "  3. late 2036 / early 2037   : Arrival at Jupiter; Callisto halo station deployed (7.08 rad/hr haven).",
            "  4. June 2037 - Oct 2038     : Europa plume sorties (50 km alt); Canister 1 launched to Earth in Oct 2038.",
            "  5. June 2041                : Support tanker arrives at Callisto with +300 kg propellant (>200 sorties enabled).",
            "  6. April 2043               : First pristine Europa ocean water captured into Earth GEO Station (>1M yr quarantine)!",
            "  7. 2042 - 2055+             : Reusable fleet dispatched to Saturn (Enceladus), Uranus (Miranda), Neptune (Triton).",
            "==================================================================================================",
        ])
        return "\n".join(lines)
