#!/usr/bin/env python3
"""
Relational 3-Body Dynamical Visualizer & Shape Sphere S² Animation Generator.

Provides CLI rendering of 3-body gravitational choreographies and astrophysical triples:
1. Physical Orbits in Barycentric COM Frame (3D / 2D with dynamic bounding).
2. Montgomery-Dragt Shape Sphere S² Coordinate Trajectory:
   - w₁ = (2c² - a² - b²) / R²
   - w₂ = √3 (a² - b²) / R²
   - w₃ = 4√3 Area / R²
   - Equator (w₃ = 0): Collinear Syzygies (Euler points)
   - North/South Poles (w₃ = ±1): Lagrange Equilateral Configurations (L₄ / L₅)

Supports export to animated GIF or MP4 via matplotlib.animation.FuncAnimation.
"""

import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.integrate import solve_ivp

# System Presets with exact physical / choreographic initial conditions
SYSTEMS = {
    "j0337": {
        "name": "PSR J0337+1715 (Hierarchical Pulsar Triple)",
        "masses": [1.4378, 0.19751, 0.4101],
        "colors": ["#00f0ff", "#ffbe3b", "#bf55ec"],
        "labels": ["Pulsar (1.44 M☉)", "Inner WD (0.20 M☉)", "Outer WD (0.41 M☉)"],
        "t_span": (0.0, 15.0),
        "n_frames": 300,
        "is_3d": False,
        "init": lambda: _init_j0337(),
    },
    "figure8": {
        "name": "Figure-8 Orbit (Chenciner & Montgomery 2000)",
        "masses": [1.0, 1.0, 1.0],
        "colors": ["#00f0ff", "#ffbe3b", "#f43f5e"],
        "labels": ["Body 1", "Body 2", "Body 3"],
        "t_span": (0.0, 6.3259),
        "n_frames": 240,
        "is_3d": False,
        "init": lambda: _init_figure8(),
    },
    "lagrange": {
        "name": "Lagrange Equilateral Solution (Rigid Triangle Rotator)",
        "masses": [1.0, 1.0, 1.0],
        "colors": ["#10b981", "#3b82f6", "#bf55ec"],
        "labels": ["Body 1", "Body 2", "Body 3"],
        "t_span": (0.0, 12.0),
        "n_frames": 240,
        "is_3d": False,
        "init": lambda: _init_lagrange(),
    },
    "burrau": {
        "name": "Burrau's Pythagorean 3-4-5 Problem (Chaotic Ejection)",
        "masses": [3.0, 4.0, 5.0],
        "colors": ["#f43f5e", "#ffbe3b", "#3b82f6"],
        "labels": ["Body 1 (M=3)", "Body 2 (M=4)", "Body 3 (M=5)"],
        "t_span": (0.0, 16.0),
        "n_frames": 360,
        "is_3d": False,
        "init": lambda: _init_burrau(),
    },
    "kepler16": {
        "name": "Kepler-16b Circumbinary System (Binary + Exoplanet)",
        "masses": [0.6897, 0.20255, 0.000318],
        "colors": ["#ffbe3b", "#f43f5e", "#00f0ff"],
        "labels": ["Primary K-Dwarf", "Secondary M-Dwarf", "Kepler-16b Planet"],
        "t_span": (0.0, 18.0),
        "n_frames": 300,
        "is_3d": False,
        "init": lambda: _init_kepler16(),
    },
}


def _init_j0337():
    m1, m2, m3 = 1.4378, 0.19751, 0.4101
    m12 = m1 + m2
    mtot = m12 + m3
    a1, a2 = 1.2, 6.2
    v1 = np.sqrt(m12 / a1)
    v2 = np.sqrt(mtot / a2)

    x1, y1, vx1, vy1 = -(m2 / m12) * a1, 0.0, 0.0, -(m2 / m12) * v1
    x2, y2, vx2, vy2 = (m1 / m12) * a1, 0.0, 0.0, (m1 / m12) * v1
    x12_shift, v12_shift = -(m3 / mtot) * a2, -(m3 / mtot) * v2
    x3_shift, v3_shift = (m12 / mtot) * a2, (m12 / mtot) * v2

    return [
        [x1 + x12_shift, y1, 0.0, vx1, vy1 + v12_shift, 0.0],
        [x2 + x12_shift, y2, 0.0, vx2, vy2 + v12_shift, 0.0],
        [x3_shift, 0.0, 0.0, 0.0, v3_shift, 0.0],
    ]


def _init_figure8():
    x1, y1 = -0.97000436, 0.24308753
    x2, y2 = 0.97000436, -0.24308753
    x3, y3 = 0.0, 0.0
    vx1, vy1 = 0.46620531, 0.43236573
    vx2, vy2 = 0.46620531, 0.43236573
    vx3, vy3 = -2 * vx1, -2 * vy1
    return [
        [x1, y1, 0.0, vx1, vy1, 0.0],
        [x2, y2, 0.0, vx2, vy2, 0.0],
        [x3, y3, 0.0, vx3, vy3, 0.0],
    ]


def _init_lagrange():
    r0 = 2.0
    v = np.sqrt(1.0 / np.sqrt(3)) * np.sqrt(r0)
    theta = [0.0, 2 * np.pi / 3, 4 * np.pi / 3]
    return [
        [r0 * np.cos(th), r0 * np.sin(th), 0.0, -v * np.sin(th), v * np.cos(th), 0.0]
        for th in theta
    ]


def _init_burrau():
    return [
        [1.0, -5.0 / 3.0, 0.0, 0.0, 0.0, 0.0],
        [-2.0, -5.0 / 3.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 7.0 / 3.0, 0.0, 0.0, 0.0, 0.0],
    ]


def _init_kepler16():
    m1, m2, m3 = 0.6897, 0.20255, 0.000318
    m12 = m1 + m2
    mtot = m12 + m3
    a1, a2 = 1.0, 3.8
    v1 = np.sqrt(m12 / a1)
    v2 = np.sqrt(mtot / a2)
    return [
        [-(m2 / m12) * a1, 0.0, 0.0, 0.0, -(m2 / m12) * v1, 0.0],
        [(m1 / m12) * a1, 0.0, 0.0, 0.0, (m1 / m12) * v1, 0.0],
        [a2, 0.0, 0.0, 0.0, v2, 0.0],
    ]


def integrate_system(preset_key, n_frames=None, enable_1pn=False):
    cfg = SYSTEMS[preset_key]
    m = cfg["masses"]
    init_state = np.array(cfg["init"]()).flatten()
    t_span = cfg["t_span"]
    frames = n_frames or cfg["n_frames"]
    t_eval = np.linspace(t_span[0], t_span[1], frames)

    # In geometric units G = 1, choose c_speed = 25.0 so 1PN relativistic parameter
    # (v/c)^2 ~ (1/25)^2 ~ 0.0016 exhibits genuine relativistic periastron precession
    # without blowing up the perturbation hierarchy.
    c_speed = 25.0

    def ode_derivs(t, y):
        r1, v1 = y[0:3], y[3:6]
        r2, v2 = y[6:9], y[9:12]
        r3, v3 = y[12:15], y[15:18]

        r12 = r2 - r1
        d12 = np.linalg.norm(r12) ** 3 + 1e-12
        r13 = r3 - r1
        d13 = np.linalg.norm(r13) ** 3 + 1e-12
        r23 = r3 - r2
        d23 = np.linalg.norm(r23) ** 3 + 1e-12

        a1_N = m[1] * r12 / d12 + m[2] * r13 / d13
        a2_N = -m[0] * r12 / d12 + m[2] * r23 / d23
        a3_N = -m[0] * r13 / d13 - m[1] * r23 / d23

        if not enable_1pn:
            return np.concatenate([v1, a1_N, v2, a2_N, v3, a3_N])

        pos = [r1, r2, r3]
        vel = [v1, v2, v3]
        a_N = [a1_N, a2_N, a3_N]
        a_1pn = [np.zeros(3), np.zeros(3), np.zeros(3)]
        inv_c2 = 1.0 / (c_speed**2)

        for A in range(3):
            for B in range(3):
                if A == B: continue
                r_BA = pos[A] - pos[B]
                r_AB = max(1e-6, np.linalg.norm(r_BA))
                n_AB = r_BA / r_AB
                n_BA = -n_AB
                v_A = vel[A]; v_B = vel[B]
                v_A_sq = np.dot(v_A, v_A); v_B_sq = np.dot(v_B, v_B)
                v_A_dot_v_B = np.dot(v_A, v_B); n_AB_dot_v_B = np.dot(n_AB, v_B)

                phi_A = sum(m[C] / max(1e-6, np.linalg.norm(pos[A] - pos[C])) for C in range(3) if C != A)
                phi_B = sum(m[C] / max(1e-6, np.linalg.norm(pos[B] - pos[C])) for C in range(3) if C != B)
                x_dot_a_B = np.dot(pos[B] - pos[A], a_N[B])

                scalar = (v_A_sq + 2.0*v_B_sq - 4.0*v_A_dot_v_B - 1.5*(n_AB_dot_v_B**2)
                          - 4.0*phi_A - phi_B + 0.5*x_dot_a_B)
                term1 = (m[B] / (r_AB**2)) * n_BA * scalar
                term2 = (m[B] / (r_AB**2)) * np.dot(n_AB, 4.0*v_A - 3.0*v_B) * (v_A - v_B)
                term3 = 3.5 * (m[B] / r_AB) * a_N[B]
                a_1pn[A] += term1 + term2 + term3

        a1 = a1_N + inv_c2 * a_1pn[0]
        a2 = a2_N + inv_c2 * a_1pn[1]
        a3 = a3_N + inv_c2 * a_1pn[2]
        return np.concatenate([v1, a1, v2, a2, v3, a3])

    sol = solve_ivp(
        ode_derivs,
        t_span,
        init_state,
        t_eval=t_eval,
        method="DOP853",
        rtol=1e-10,
        atol=1e-10,
    )

    r1 = sol.y[0:3].T
    r2 = sol.y[6:9].T
    r3 = sol.y[12:15].T

    # Montgomery-Dragt Shape Sphere Coordinates
    r12_sq = np.sum((r2 - r1) ** 2, axis=1)
    r23_sq = np.sum((r3 - r2) ** 2, axis=1)
    r31_sq = np.sum((r1 - r3) ** 2, axis=1)
    r_sq = r12_sq + r23_sq + r31_sq

    cross_prod = np.cross(r2 - r1, r3 - r1)
    area = 0.5 * np.linalg.norm(cross_prod, axis=1)

    w1 = (2 * r31_sq - r12_sq - r23_sq) / r_sq
    w2 = np.sqrt(3) * (r12_sq - r23_sq) / r_sq
    w3 = 4 * np.sqrt(3) * area / r_sq

    # Validate shape sphere invariant |w|^2 == 1
    sphere_res = np.max(np.abs(w1**2 + w2**2 + w3**2 - 1.0))

    return {
        "t": sol.t,
        "r1": r1,
        "r2": r2,
        "r3": r3,
        "w1": w1,
        "w2": w2,
        "w3": w3,
        "sphere_res": sphere_res,
        "cfg": cfg,
    }


def create_animation(sim_data, save_path=None, fps=30):
    cfg = sim_data["cfg"]
    r1, r2, r3 = sim_data["r1"], sim_data["r2"], sim_data["r3"]
    w1, w2, w3 = sim_data["w1"], sim_data["w2"], sim_data["w3"]
    t = sim_data["t"]
    n_frames = len(t)

    # Style configuration
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(14, 6.5), dpi=100)
    fig.patch.set_facecolor("#0a0c14")

    # Left: Physical Orbit (COM Barycentric Frame)
    ax_orbit = fig.add_subplot(1, 2, 1)
    ax_orbit.set_facecolor("#0e121e")
    ax_orbit.set_title(f"{cfg['name']}\nCOM Physical Orbit", fontsize=11, color="#00f0ff", pad=10)
    ax_orbit.set_xlabel("X (Barycentric)", fontsize=9, color="#8a9bb8")
    ax_orbit.set_ylabel("Y (Barycentric)", fontsize=9, color="#8a9bb8")
    ax_orbit.grid(True, linestyle="--", alpha=0.2, color="#3c5078")

    all_x = np.concatenate([r1[:, 0], r2[:, 0], r3[:, 0]])
    all_y = np.concatenate([r1[:, 1], r2[:, 1], r3[:, 1]])
    pad_x = (np.max(all_x) - np.min(all_x)) * 0.15 + 0.1
    pad_y = (np.max(all_y) - np.min(all_y)) * 0.15 + 0.1
    ax_orbit.set_xlim(np.min(all_x) - pad_x, np.max(all_x) + pad_x)
    ax_orbit.set_ylim(np.min(all_y) - pad_y, np.max(all_y) + pad_y)
    ax_orbit.set_aspect("equal", adjustable="datalim")

    # Orbit Elements
    trail1, = ax_orbit.plot([], [], color=cfg["colors"][0], alpha=0.7, lw=1.5, label=cfg["labels"][0])
    trail2, = ax_orbit.plot([], [], color=cfg["colors"][1], alpha=0.7, lw=1.5, label=cfg["labels"][1])
    trail3, = ax_orbit.plot([], [], color=cfg["colors"][2], alpha=0.7, lw=1.5, label=cfg["labels"][2])
    body1, = ax_orbit.plot([], [], "o", color=cfg["colors"][0], markersize=8)
    body2, = ax_orbit.plot([], [], "o", color=cfg["colors"][1], markersize=6)
    body3, = ax_orbit.plot([], [], "o", color=cfg["colors"][2], markersize=7)
    triangle, = ax_orbit.plot([], [], "--", color="#00f0ff", alpha=0.3, lw=1.0)
    barycenter, = ax_orbit.plot([0], [0], "+", color="#ffffff", markersize=8, alpha=0.5, label="Barycenter")
    time_text = ax_orbit.text(0.03, 0.05, "", transform=ax_orbit.transAxes, color="#e0e6f0", fontsize=9, fontfamily="monospace")
    ax_orbit.legend(loc="upper right", fontsize=8, facecolor="#161c2d", edgecolor="#3c5078")

    # Right: Montgomery-Dragt Shape Sphere S²
    ax_sphere = fig.add_subplot(1, 2, 2, projection="3d")
    ax_sphere.set_facecolor("#0e121e")
    ax_sphere.set_title("Montgomery-Dragt Shape Sphere $S^2$\nRelational Gauge Invariant State", fontsize=11, color="#ffbe3b", pad=10)
    ax_sphere.xaxis.set_pane_color((0.05, 0.07, 0.12, 1.0))
    ax_sphere.yaxis.set_pane_color((0.05, 0.07, 0.12, 1.0))
    ax_sphere.zaxis.set_pane_color((0.05, 0.07, 0.12, 1.0))
    ax_sphere.grid(False)

    # Wireframe Sphere
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 15)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones(np.size(u)), np.cos(v))
    ax_sphere.plot_wireframe(xs, ys, zs, color="#3c5078", alpha=0.15, lw=0.5)

    # Equator (Collinear Syzygies w3 = 0)
    theta_eq = np.linspace(0, 2 * np.pi, 100)
    ax_sphere.plot(np.cos(theta_eq), np.sin(theta_eq), np.zeros_like(theta_eq), color="#f43f5e", lw=1.8, label="Collinear Syzygy ($w_3=0$)")

    # Poles (Lagrange Equilateral L4 / L5)
    ax_sphere.plot([0], [0], [1], "o", color="#10b981", markersize=6, label="$L_4$ Equilateral (North)")
    ax_sphere.plot([0], [0], [-1], "o", color="#3b82f6", markersize=6, label="$L_5$ Equilateral (South)")

    # Trajectory & Current State Marker
    sphere_trail, = ax_sphere.plot([], [], [], color="#ffbe3b", lw=2.0, alpha=0.85, label="Trajectory")
    sphere_curr, = ax_sphere.plot([], [], [], "o", color="#ffffff", markersize=7)
    sphere_text = ax_sphere.text2D(0.03, 0.05, "", transform=ax_sphere.transAxes, color="#ffbe3b", fontsize=9, fontfamily="monospace")

    ax_sphere.set_xlim(-1.1, 1.1)
    ax_sphere.set_ylim(-1.1, 1.1)
    ax_sphere.set_zlim(-1.1, 1.1)
    ax_sphere.set_xlabel("$w_1$ (Scalene)", fontsize=8, color="#8a9bb8")
    ax_sphere.set_ylabel("$w_2$ (Isosceles)", fontsize=8, color="#8a9bb8")
    ax_sphere.set_zlabel("$w_3$ (Chirality / Area)", fontsize=8, color="#8a9bb8")
    ax_sphere.legend(loc="upper left", fontsize=7, facecolor="#161c2d", edgecolor="#3c5078")
    ax_sphere.view_init(elev=25, azim=45)

    fig.tight_layout()

    trail_window = min(150, n_frames)

    def update(frame):
        # Orbit trails
        start = max(0, frame - trail_window)
        trail1.set_data(r1[start : frame + 1, 0], r1[start : frame + 1, 1])
        trail2.set_data(r2[start : frame + 1, 0], r2[start : frame + 1, 1])
        trail3.set_data(r3[start : frame + 1, 0], r3[start : frame + 1, 1])

        body1.set_data([r1[frame, 0]], [r1[frame, 1]])
        body2.set_data([r2[frame, 0]], [r2[frame, 1]])
        body3.set_data([r3[frame, 0]], [r3[frame, 1]])

        tri_x = [r1[frame, 0], r2[frame, 0], r3[frame, 0], r1[frame, 0]]
        tri_y = [r1[frame, 1], r2[frame, 1], r3[frame, 1], r1[frame, 1]]
        triangle.set_data(tri_x, tri_y)
        time_text.set_text(f"t = {t[frame]:.2f}\nFrame {frame+1}/{n_frames}")

        # Shape Sphere
        sphere_trail.set_data(w1[: frame + 1], w2[: frame + 1])
        sphere_trail.set_3d_properties(w3[: frame + 1])
        sphere_curr.set_data([w1[frame]], [w2[frame]])
        sphere_curr.set_3d_properties([w3[frame]])

        sphere_text.set_text(
            f"w₁={w1[frame]:+.3f}  w₂={w2[frame]:+.3f}\nw₃={w3[frame]:+.3f}  |w|²={w1[frame]**2+w2[frame]**2+w3[frame]**2:.4f}"
        )

        # Smooth camera rotation on sphere
        ax_sphere.view_init(elev=25, azim=45 + frame * 0.4)

        return trail1, trail2, trail3, body1, body2, body3, triangle, time_text, sphere_trail, sphere_curr, sphere_text

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False)

    if save_path:
        print(f"[+] Rendering animation to: {save_path} ({n_frames} frames @ {fps} fps)...")
        if save_path.endswith(".gif"):
            writer = PillowWriter(fps=fps)
            anim.save(save_path, writer=writer)
        else:
            anim.save(save_path, fps=fps, extra_args=["-vcodec", "libx264"])
        print(f"[✓] Successfully saved: {save_path}")
    else:
        plt.show()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Relational 3-Body Dynamical Visualizer & Montgomery-Dragt Shape Sphere S² Animator"
    )
    parser.add_argument(
        "--system",
        choices=["j0337", "figure8", "lagrange", "burrau", "kepler16", "all"],
        default="figure8",
        help="System preset to simulate (default: figure8)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save output animation (.gif or .mp4)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Number of animation frames to render (default: system preset standard)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for output animation (default: 30)",
    )
    parser.add_argument(
        "--enable-1pn",
        action="store_true",
        help="Enable 1PN Einstein-Infeld-Hoffmann general relativistic cross-accelerations",
    )

    args = parser.parse_args()

    systems_to_run = list(SYSTEMS.keys()) if args.system == "all" else [args.system]

    for sys_key in systems_to_run:
        mode_str = "1PN Einstein-Infeld-Hoffmann (GR)" if args.enable_1pn else "Newtonian Gravitation"
        print(f"\n=================================================================")
        print(f" Simulating: {SYSTEMS[sys_key]['name']} [{mode_str}]")
        print(f"=================================================================")
        sim_data = integrate_system(sys_key, n_frames=args.frames, enable_1pn=args.enable_1pn)
        print(f" Integration complete. Time steps: {len(sim_data['t'])}")
        print(f" Montgomery-Dragt Invariant Error: {sim_data['sphere_res']:.2e}")

        out_path = args.save
        if out_path and args.system == "all":
            base, ext = os.path.splitext(out_path)
            out_path = f"{base}_{sys_key}{ext}"

        create_animation(sim_data, save_path=out_path, fps=args.fps)


if __name__ == "__main__":
    main()
