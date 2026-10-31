"""
=====================================================================================
Tests for Planet-Specific Rover Restocking & Surface Relay Network Architecture
=====================================================================================
Verifies:
  1. Planet-specific rover specifications tailored to 5 distinct outer solar system environments.
  2. End-of-mission soft-landing mechanics and delta-v budgets across Ceres, Callisto, Titan, Titania, and Triton.
  3. Radioisotope power longevity (15 to 30 year operational lifespans).
  4. Surface science instrumentation (seismology, exospheric spectrometry, cryovolcanism).
  5. Permanent surface-to-orbital-depot relay network synthesis.
=====================================================================================
"""

import pytest

from qrc_pulsar_bench.surface_relay_network import (
    PlanetSpecificRover,
    SurfaceObservationPost,
    SurfaceRelayNetworkArchitect,
)


def test_planet_specific_rover_adaptations():
    """Verify rovers are tailored to distinct atmospheric/vacuum environments."""
    rovers = SurfaceRelayNetworkArchitect.get_planet_specific_rovers()
    assert len(rovers) == 5

    targets = [r.target_planet for r in rovers]
    assert any("Ceres" in t for t in targets)
    assert any("Jupiter" in t for t in targets)
    assert any("Saturn" in t for t in targets)
    assert any("Uranus" in t for t in targets)
    assert any("Neptune" in t for t in targets)

    # Titan rover handles dense atmosphere
    titan = next(r for r in rovers if "Saturn" in r.target_planet)
    assert titan.atmospheric_pressure_bar > 1.0
    assert "Rotorcraft" in titan.rover_type
    assert titan.ambient_temperature_k == 94.0

    # Triton rover operates at 38 K deep freeze
    triton = next(r for r in rovers if "Neptune" in r.target_planet)
    assert triton.ambient_temperature_k == 38.0
    assert "cryo-gold" in triton.sampling_mechanism.lower()


def test_surface_observation_posts_and_landing_mechanics():
    """Verify landing delta-v and power systems for permanent surface posts."""
    posts = SurfaceRelayNetworkArchitect.get_surface_observation_posts()
    assert len(posts) == 5

    bodies = [p.landing_body for p in posts]
    assert "Dwarf Planet 1 Ceres" in bodies
    assert "Callisto (Jupiter)" in bodies
    assert "Titan (Saturn)" in bodies
    assert "Titania (Uranus)" in bodies
    assert "Triton (Neptune)" in bodies

    # Titan landing requires negligible propulsive dV due to dense atmosphere
    titan_post = next(p for p in posts if "Titan" in p.landing_body)
    assert titan_post.surface_landing_dv_mps <= 30.0
    assert "parafoil" in titan_post.braking_mechanism.lower() or "chute" in titan_post.braking_mechanism.lower()

    # Callisto landing requires propulsive burn
    callisto_post = next(p for p in posts if "Callisto" in p.landing_body)
    assert callisto_post.surface_landing_dv_mps > 1000.0
    assert "MMRTG" in callisto_post.power_source

    # All posts operate for 15+ years
    for post in posts:
        assert post.operational_lifespan_years >= 15.0
        assert len(post.science_payload) >= 3
        assert "relay" in post.relay_telecom_link.lower() or "link" in post.relay_telecom_link.lower()


def test_interplanetary_surface_relay_network_evaluation():
    """Verify full network aggregation and science value."""
    net = SurfaceRelayNetworkArchitect.evaluate_interplanetary_network()

    assert net["total_planet_tailored_rovers"] == 5
    assert net["total_permanent_surface_posts"] == 5
    assert net["cumulative_astronomical_span_au"] > 27.0  # Spans 27+ AU across the Solar System
    assert net["total_surface_instruments_deployed"] >= 15
    assert net["average_post_lifespan_years"] >= 20.0
    assert "Zero hardware is wasted" in net["permanent_infrastructure_value"]
