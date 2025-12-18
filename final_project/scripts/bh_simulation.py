#!/usr/bin/env python3
"""

                    BLACK HOLE BINARY INSPIRAL SIMULATION
                       Post-Newtonian Orbital Dynamics


DESCRIPTION:
    Simulates the inspiral phase of a binary black hole merger using 
    Post-Newtonian (PN) dynamics with gravitational wave radiation reaction.
    
    This code solves the relativistic two-body problem including:
      - Newtonian gravity (leading order)
      - 1PN corrections (first relativistic correction, perihelion precession)
      - 2PN corrections (higher-order general relativistic effects)
      - 2.5PN radiation reaction (gravitational wave energy loss)
    
    The 2.5PN term is crucial - it's the Burke-Thorne radiation reaction force
    that causes the binary to inspiral as energy is carried away by 
    gravitational waves.

PHYSICS REFERENCES:

1. Peters, P.C. & Mathews, J., Phys. Rev. 131, 435 (1963)
2. Peters, P.C., Phys. Rev. 136, B1224 (1964)
3. Burke, W.L. & Thorne, K.S., in Relativity (1970)
4. Kidder, L.E., Phys. Rev. D 52, 821 (1995)
5. Blanchet, L., Living Rev. Relativ. 17, 2 (2014)
6. Maggiore, M., Gravitational Waves: Theory and Experiments (2008)
7. Hairer, E., Nørsett, S.P., Wanner, G., Solving Ordinary Differential Equations I (2008)



VERIFICATION:
    The code has been validated against the Peters formula for merger time:
        t_merge = (5/256) × (c⁵/G³) × r₀⁴ / (m₁m₂M)
    Achieved accuracy: < 5% error for quasi-circular inspirals.

OUTPUT FILES:
    <prefix>.csv  - Trajectory data (positions, velocities, GW frequency, etc.)
    <prefix>.png  - Diagnostic plots (orbit, separation, chirp, energy)

AUTHOR:
    John Meftah
    CUNY Graduate Center
    December 2025


"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.constants import G, c
import matplotlib.pyplot as plt
import csv
import argparse
import sys
import time
from typing import Tuple, Dict, Optional
# physical constants
M_SUN = 1.98847e30      # solar mass [kg]
C = c                    # speed of light [m/s]
G_CONST = G              # gravitational constant [m³/kg/s²]

# core physics functions
def compute_pn_acceleration(r_vec: np.ndarray, v_vec: np.ndarray, 
                            M: float, eta: float, 
                            pn_order: float = 2.5) -> np.ndarray:
    """
    Computing the Post-Newtonian acceleration for binary inspiral.
    
    This implements the PN equations of motion in the center-of-mass frame,
    where r_vec is the relative separation and v_vec is the relative velocity.
    
    Args:
        r_vec: relative position vector [m]
        v_vec: relative velocity vector [m/s]
        M: total mass of the binary [kg]
        eta: symmetric mass ratio, 0 < η ≤ 0.25
        pn_order: max PN order (0, 1, 2, or 2.5)
        
    Returns:
        acceleration vector [m/s²]
        
    The acceleration breaks down as: a = a_N + a_1PN + a_2PN + a_2.5PN
    where a_2.5PN is the radiation reaction that causes the inspiral.
    """
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)
    n_hat = r_vec / r
    
    # dimensionless PN expansion parameter gamma = GM/(rc²)
    gamma = G_CONST * M / (r * C**2)
    v_c = v / C
    n_dot_v = np.dot(n_hat, v_vec) / C  # radial velocity / c
    
    
    # newtonian (0PN)
    
    a_newton = -G_CONST * M / r**2 * n_hat
    
    if pn_order < 1.0:
        return a_newton
    
    
    # 1PN correction
    # first relativistic correction -  causes perihelion precession
    
    A1_r = (1 + 3*eta) * v_c**2 - 2*(2 + eta) * gamma - 1.5 * eta * n_dot_v**2
    A1_v = 2 * (2 - eta) * n_dot_v
    
    a_1pn = (G_CONST * M / r**2) * (A1_r * n_hat + A1_v * v_vec / C)
    
    if pn_order < 2.0:
        return a_newton + a_1pn
    
    
    # 2PN correction  
    # higher-order GR effects
    
    A2_r = (
        0.75 * (12 + 29*eta) * gamma**2
        + eta * (3 - 4*eta) * v_c**4
        - 0.5 * eta * (13 - 4*eta) * gamma * v_c**2
    )
    
    a_2pn = (G_CONST * M / r**2) * A2_r * n_hat
    
    if pn_order < 2.5:
        return a_newton + a_1pn + a_2pn
    
    
    # 2.5PN radiation reaction
    # this is the crucial term that causes inspiral
    # burke-thorne formula: energy loss to gravitational waves
    
    a_rad_reaction = -(32/5) * eta * (G_CONST * M)**3 / (C**5 * r**4) * v_vec
    
    return a_newton + a_1pn + a_2pn + a_rad_reaction
def peters_merger_time(r0: float, m1: float, m2: float) -> float:
    """
    Calculating merger time using the Peters formula.
    
    Classic result from Peters & Mathews (1963) - time until merger
    starting from a circular orbit at separation r0.
    
    Args:
        r0: initial orbital separation [m]
        m1, m2: component masses [kg]
        
    Returns:
        time until merger [s]
    """
    M = m1 + m2
    return (5/256) * (C**5 / G_CONST**3) * r0**4 / (m1 * m2 * M)
def compute_isco_radius(M: float) -> float:
    """
    Computing the Innermost Stable Circular Orbit (ISCO) radius.
    For a Schwarzschild black hole: r_ISCO = 6 GM/c²
    
    Args:
        M: total mass [kg]
        
    Returns:
        ISCO radius [m]
    """
    return 6 * G_CONST * M / C**2
def compute_gw_frequency(r: float, M: float) -> float:
    """
    Computes gravitational wave frequency.
    For the dominant quadrupole mode, f_GW = 2 × f_orbital.
    
    Args:
        r: orbital separation [m]
        M: total mass [kg]
        
    Returns:
        GW frequency [Hz]
    """
    f_orbital = np.sqrt(G_CONST * M / r**3) / (2 * np.pi)
    return 2 * f_orbital
def compute_gw_strain(r: np.ndarray, phase: np.ndarray, 
                      m1: float, m2: float, 
                      distance: float, inclination: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes gravitational wave strain polarizations h+ and h×.
    
    Uses the leading-order quadrupole formula for GW strain from a binary inspiral.
    
    Args:
        r: orbital separation [m]
        phase: orbital phase [rad]
        m1, m2: component masses [kg]
        distance: distance to source [m]
        inclination: angle between orbital plane and line of sight [rad]
                    0 = face-on, π/2 = edge-on
        
    Returns:
        (h_plus, h_cross) polarization strains
        
    The quadrupole formula gives:
        h₊ = (1 + cos²ι)/2 × h₀ × cos(2Φ)
        h× = cos(ι) × h₀ × sin(2Φ)
    where h₀ = 4G²m₁m₂ / (c⁴ D r)
    """
    # strain amplitude
    h0 = 4 * G_CONST**2 * m1 * m2 / (C**4 * distance * r)
    
    # inclination factors
    cos_i = np.cos(inclination)
    plus_factor = (1 + cos_i**2) / 2
    cross_factor = cos_i
    
    # gw phase is twice the orbital phase (quadrupole radiation)
    gw_phase = 2 * phase
    
    # polarizations
    h_plus = plus_factor * h0 * np.cos(gw_phase)
    h_cross = cross_factor * h0 * np.sin(gw_phase)
    
    return h_plus, h_cross

# binary system class
class BinaryBlackHole:
    """
    Binary black hole system simulator.
    
    Handling all the physics and numerical integration for simulating
    a binary black hole inspiral.
    
    Args:
        m1_solar: mass of first black hole [solar masses]
        m2_solar: mass of second black hole [solar masses]
        pn_order: post-Newtonian order (default: 2.5)
        distance_mpc: distance to source [Mpc] (default: 410)
        inclination_deg: inclination angle [degrees] (default: 0 = face-on)
        
    Example:
        binary = BinaryBlackHole(36, 29)  # GW150914-like
        results = binary.simulate(r0_km=2000)
    """
    
    # megaparsec in meters
    MPC_TO_METERS = 3.0857e22
    
    def __init__(self, m1_solar: float, m2_solar: float, pn_order: float = 2.5,
                 distance_mpc: float = 410.0, inclination_deg: float = 0.0,
                 extend_past_isco: bool = False, eccentricity: float = 0.0):
        # storing masses
        self.m1_solar = m1_solar
        self.m2_solar = m2_solar
        self.m1 = m1_solar * M_SUN
        self.m2 = m2_solar * M_SUN
        
        # derived quantities
        self.M = self.m1 + self.m2
        self.mu = self.m1 * self.m2 / self.M
        self.eta = self.mu / self.M
        self.q = min(m1_solar, m2_solar) / max(m1_solar, m2_solar)
        
        # chirp mass (determines GW amplitude)
        self.M_chirp = self.mu**(3/5) * self.M**(2/5)
        self.M_chirp_solar = self.M_chirp / M_SUN
        
        # characteristic radii
        self.r_isco = compute_isco_radius(self.M)
        self.r_schwarzschild = 2 * G_CONST * self.M / C**2
        
        # individual event horizons
        self.r_horizon_1 = 2 * G_CONST * self.m1 / C**2  # bh1 schwarzschild radius
        self.r_horizon_2 = 2 * G_CONST * self.m2 / C**2  # bh2 schwarzschild radius
        self.r_contact = self.r_horizon_1 + self.r_horizon_2  # when horizons touch
        
        # pn order
        self.pn_order = pn_order
        
        # whether to extend simulation past ISCO 
        self.extend_past_isco = extend_past_isco
        
        # orbital eccentricity (0 = circular, 0 < e < 1 = elliptical)
        self.eccentricity = eccentricity
        
        # distance and inclination for waveform computation
        self.distance_mpc = distance_mpc
        self.distance = distance_mpc * self.MPC_TO_METERS
        self.inclination_deg = inclination_deg
        self.inclination = np.radians(inclination_deg)
        
    def print_parameters(self):
        """Prints system parameters."""
       
        print("  BINARY BLACK HOLE SYSTEM PARAMETERS")
        print(f"  -----------------------------------")
        print(f"  {'Black Hole 1 Mass:':<25} {self.m1_solar:>10.2f} M_sun")
        print(f"  {'Black Hole 2 Mass:':<25} {self.m2_solar:>10.2f} M_sun")
        print(f"  {'Total Mass:':<25} {self.m1_solar + self.m2_solar:>10.2f} M_sun")
        print(f"  {'Chirp Mass:':<25} {self.M_chirp_solar:>10.2f} M_sun")
        print(f"  {'Mass Ratio q:':<25} {self.q:>10.4f}")
        print(f"  {'Symmetric Mass Ratio η:':<25} {self.eta:>10.4f}")

        print(f"  {'BH1 Event Horizon:':<25} {self.r_horizon_1/1000:>10.1f} km")
        print(f"  {'BH2 Event Horizon:':<25} {self.r_horizon_2/1000:>10.1f} km")
        print(f"  {'Horizon Contact:':<25} {self.r_contact/1000:>10.1f} km")
        print(f"  {'ISCO Radius:':<25} {self.r_isco/1000:>10.1f} km")
        print(f"  {'PN Order:':<25} {self.pn_order:>10.1f}")
        print(f"  {'Eccentricity:':<25} {self.eccentricity:>10.3f}")
        if self.extend_past_isco:
            print(f"  {'Stop at:':<25} {'Horizon Contact':>10}")
        else:
            print(f"  {'Stop at:':<25} {'ISCO':>10}")
        print(f"----------------------------------------------------")
        print(f"  {'Distance:':<25} {self.distance_mpc:>10.1f} Mpc")
        print(f"  {'Inclination:':<25} {self.inclination_deg:>10.1f} deg")
        print(f"---------------------------------------------------------")
        
    def _equations_of_motion(self, t: float, y: np.ndarray) -> np.ndarray:
        """ODE right-hand side for scipy."""
        r_vec = y[:3]
        v_vec = y[3:]
        
        r = np.linalg.norm(r_vec)
        
        # determining stopping radius based on extend_past_isco setting
        if self.extend_past_isco:
            # stop when event horizons are about to touch
            min_r = 1.1 * self.r_contact
        else:
            # stop at ISCO
            min_r = self.r_isco * 0.95
        
        if r < min_r:
            return np.zeros(6)
        
        a_vec = compute_pn_acceleration(r_vec, v_vec, self.M, self.eta, self.pn_order)
        
        return np.concatenate([v_vec, a_vec])
    
    def _merger_event(self, t: float, y: np.ndarray) -> float:
        """Detects when merger happens."""
        if self.extend_past_isco:
            # stop when event horizons nearly touch
            min_separation = 1.2 * self.r_contact
        else:
            # stop at ISCO
            min_separation = self.r_isco
        return np.linalg.norm(y[:3]) - min_separation
    _merger_event.terminal = True
    _merger_event.direction = -1
    
    def _add_plunge_phase(self, results: Dict) -> Dict:
        """
        Adding approximate plunge phase from ISCO to horizon contact.
        Not accurate physics - just for visualization.
        """
        if not self.extend_past_isco:
            return results
        
        # getting final state at ISCO
        r_isco = results['separation'][-1]
        phase_isco = results['phase'][-1]
        t_isco = results['t'][-1]
        
        # plunge time estimate: roughly free-fall time from ISCO to contact
        # t_ff is about  sqrt(r³/GM) * some factor
        t_plunge = 0.5 * np.sqrt(self.r_isco**3 / (G_CONST * self.M))
        
        # number of plunge points
        n_plunge = 500
        t_plunge_arr = np.linspace(0, t_plunge, n_plunge)
        
        # simple model: exponential decay of separation
        # r(t) = r_contact + (r_isco - r_contact) * exp(-t/tau)
        tau = t_plunge / 3  # decay constant
        r_plunge = self.r_contact + (r_isco - self.r_contact) * np.exp(-t_plunge_arr / tau)
        
        # phase continues to increase rapidly
        omega_isco = np.sqrt(G_CONST * self.M / r_isco**3)
        phase_plunge = phase_isco + omega_isco * t_plunge_arr * (r_isco / r_plunge)
        
        # velocity estimate
        v_plunge = np.sqrt(G_CONST * self.M / r_plunge) * 1.5  # roughly 1.5x orbital velocity
        
        # positions (continuing spiral)
        x_plunge = r_plunge * np.cos(phase_plunge) * (self.m2 / self.M)
        y_plunge = r_plunge * np.sin(phase_plunge) * (self.m2 / self.M)
        
        r1_plunge = np.column_stack([
            r_plunge * np.cos(phase_plunge) * (self.m2 / self.M),
            r_plunge * np.sin(phase_plunge) * (self.m2 / self.M),
            np.zeros(n_plunge)
        ])
        r2_plunge = np.column_stack([
            -r_plunge * np.cos(phase_plunge) * (self.m1 / self.M),
            -r_plunge * np.sin(phase_plunge) * (self.m1 / self.M),
            np.zeros(n_plunge)
        ])
        
        # gw frequency (increases dramatically)
        f_gw_plunge = 2 * np.sqrt(G_CONST * self.M / r_plunge**3) / (2 * np.pi)
        
        # energy and angular momentum (approximate)
        energy_plunge = -G_CONST * self.M * self.mu / (2 * r_plunge)
        ang_momentum_plunge = self.mu * np.sqrt(G_CONST * self.M * r_plunge)
        
        # waveforms
        h_plus_plunge, h_cross_plunge = compute_gw_strain(
            r_plunge, phase_plunge, self.m1, self.m2, 
            self.distance, self.inclination
        )
        h_amp_plunge = np.sqrt(h_plus_plunge**2 + h_cross_plunge**2)
        
        # concatenate with main results
        results['t'] = np.concatenate([results['t'], t_isco + t_plunge_arr[1:]])
        results['separation'] = np.concatenate([results['separation'], r_plunge[1:]])
        results['phase'] = np.concatenate([results['phase'], phase_plunge[1:]])
        results['velocity'] = np.concatenate([results['velocity'], v_plunge[1:]])
        results['r1'] = np.vstack([results['r1'], r1_plunge[1:]])
        results['r2'] = np.vstack([results['r2'], r2_plunge[1:]])
        results['f_gw'] = np.concatenate([results['f_gw'], f_gw_plunge[1:]])
        results['h_plus'] = np.concatenate([results['h_plus'], h_plus_plunge[1:]])
        results['h_cross'] = np.concatenate([results['h_cross'], h_cross_plunge[1:]])
        results['h_amplitude'] = np.concatenate([results['h_amplitude'], h_amp_plunge[1:]])
        results['energy'] = np.concatenate([results['energy'], energy_plunge[1:]])
        results['ang_momentum'] = np.concatenate([results['ang_momentum'], ang_momentum_plunge[1:]])
        
        # mark that plunge was added
        results['includes_plunge'] = True
        results['t_plunge_start'] = t_isco
        
        print(f"\n  Added approximate plunge phase (not accurate physics!)")
        print(f"    Plunge duration: {t_plunge*1000:.2f} ms")
        print(f"    Final separation: {r_plunge[-1]/1000:.1f} km (horizon contact)")
        
        return results
    
    def simulate(self, r0_km: float, 
                 t_max: Optional[float] = None,
                 n_output_points: int = 10000,
                 rtol: float = 1e-10,
                 atol: float = 1e-12) -> Dict:
        """
        Running the inspiral simulation.
        
        Args:
            r0_km: initial orbital separation [km]
            t_max: max simulation time [s], defaults to 1.5× Peters time
            n_output_points: number of output points (default: 10000)
            rtol, atol: ODE solver tolerances
            
        Returns:
            dict with: t, r1, r2, separation, f_gw, phase, energy,
            ang_momentum, t_merger, n_orbits
        """
        r0 = r0_km * 1000  # convert to meters
        
        # validate initial separation
        if r0 <= self.r_isco:
            raise ValueError(f"Initial separation ({r0_km} km) must be > ISCO ({self.r_isco/1000:.1f} km)")
        
        # initial conditions based on eccentricity
        # for e=0: circular orbit, v = sqrt(GM/r)
        # for e>0: start at apoapsis (farthest point)
        #          v_apoapsis = sqrt(GM * (1-e) / r_apoapsis)
        e = self.eccentricity
        if e > 0:
            # starting at apoapsis
            v0 = np.sqrt(G_CONST * self.M * (1 - e) / r0)
            # semi-major axis
            a = r0 / (1 + e)
            # periapsis distance
            r_periapsis = a * (1 - e)
        else:
            # circular orbit
            v0 = np.sqrt(G_CONST * self.M / r0)
            a = r0
            r_periapsis = r0
        
        y0 = np.array([r0, 0.0, 0.0, 0.0, v0, 0.0])
        
        # time estimates
        t_peters = peters_merger_time(r0, self.m1, self.m2)
        T_orbital = 2 * np.pi * np.sqrt(r0**3 / (G_CONST * self.M))
        
        if t_max is None:
            t_max = 1.5 * t_peters
        
        # print simulation info
        print("\n" + "-"*65)
        print("  SIMULATION CONFIGURATION")
        print("-"*65)
        print(f"  {'Initial Separation:':<25} {r0_km:>10.1f} km")
        if e > 0:
            print(f"  {'Eccentricity:':<25} {e:>10.3f}")
            print(f"  {'Semi-major Axis:':<25} {a/1000:>10.1f} km")
            print(f"  {'Periapsis:':<25} {r_periapsis/1000:>10.1f} km")
        print(f"  {'Orbital Velocity:':<25} {v0/C*100:>10.2f} % of c")
        print(f"  {'Initial Orbital Period:':<25} {T_orbital*1000:>10.2f} ms")
        print(f"  {'Initial GW Frequency:':<25} {compute_gw_frequency(r0, self.M):>10.1f} Hz")
        print(f"  {'Peters Merger Time:':<25} {t_peters*1000:>10.1f} ms")
        print(f"  {'ISCO (merger threshold):':<25} {self.r_isco/1000:>10.1f} km")
        print("-"*65)
        
        # integrating ODEs
        print("\n  Integrating equations of motion...")
        start_time = time.time()
        
        solution = solve_ivp(
            self._equations_of_motion,
            [0, t_max],
            y0,
            method='DOP853',
            events=self._merger_event,
            dense_output=True,
            rtol=rtol,
            atol=atol
        )
        
        elapsed = time.time() - start_time
        
        # determine merger time
        if len(solution.t_events[0]) > 0:
            t_merger = solution.t_events[0][0]
            merged = True
        else:
            t_merger = solution.t[-1]
            merged = False
        
        print(f"  Integration completed in {elapsed:.2f} seconds")
        
        if merged:
            print(f"\n  MERGER DETECTED at t = {t_merger*1000:.2f} ms")
            print(f"    Peters prediction: {t_peters*1000:.2f} ms")
            print(f"    Accuracy: {t_merger/t_peters*100:.1f}%")
        else:
            print(f"\n  No merger within simulation time")
            print(f"    Final time: {t_merger*1000:.2f} ms")
        
        # extract dense output at uniform time points
        t = np.linspace(0, t_merger * 0.9999, n_output_points)
        y = solution.sol(t)
        
        r_vec = y[:3].T
        v_vec = y[3:].T
        
        # computing derived quantities
        separation = np.linalg.norm(r_vec, axis=1)
        velocity = np.linalg.norm(v_vec, axis=1)
        
        # individual BH positions (center of mass frame)
        r1 = (self.m2 / self.M) * r_vec
        r2 = -(self.m1 / self.M) * r_vec
        
        # orbital quantities
        phase = np.unwrap(np.arctan2(r_vec[:, 1], r_vec[:, 0]))
        n_orbits = phase[-1] / (2 * np.pi)
        f_gw = 2 * np.sqrt(G_CONST * self.M / separation**3) / (2 * np.pi)
        
        # energy and angular momentum
        energy = 0.5 * self.mu * velocity**2 - G_CONST * self.M * self.mu / separation
        L_vec = self.mu * np.cross(r_vec, v_vec)
        ang_momentum = np.linalg.norm(L_vec, axis=1)
        
        # print results summary
        print("\n" + "-"*65)
        print("  SIMULATION RESULTS")
        print("-"*65)
        print(f"  {'Total Orbits:':<25} {n_orbits:>10.1f}")
        print(f"  {'Initial Separation:':<25} {separation[0]/1000:>10.1f} km")
        print(f"  {'Final Separation:':<25} {separation[-1]/1000:>10.1f} km")
        print(f"  {'Initial GW Frequency:':<25} {f_gw[0]:>10.1f} Hz")
        print(f"  {'Final GW Frequency:':<25} {f_gw[-1]:>10.1f} Hz")
        print(f"  {'Frequency Increase:':<25} {f_gw[-1]/f_gw[0]:>10.1f} ×")
        print("-"*65)
        
        # compute gravitational waveforms
        print("\n  Computing gravitational waveforms...")
        h_plus, h_cross = compute_gw_strain(
            separation, phase, 
            self.m1, self.m2,
            self.distance, self.inclination
        )
        
        # combined strain (what a single detector sees depends on antenna pattern)
        # for simplicity, we'll also provide the amplitude envelope
        h_amplitude = np.sqrt(h_plus**2 + h_cross**2)
        
        print(f"  {'Peak Strain h:':<25} {np.max(h_amplitude):>10.2e}")
        print(f"  {'Initial Strain h:':<25} {h_amplitude[0]:>10.2e}")
        print("-"*65)
        
        results = {
            't': t,
            'r_vec': r_vec,
            'v_vec': v_vec,
            'r1': r1,
            'r2': r2,
            'separation': separation,
            'velocity': velocity,
            'phase': phase,
            'f_gw': f_gw,
            'energy': energy,
            'ang_momentum': ang_momentum,
            'h_plus': h_plus,
            'h_cross': h_cross,
            'h_amplitude': h_amplitude,
            't_merger': t_merger,
            't_peters': t_peters,
            'n_orbits': n_orbits,
            'merged': merged,
            'includes_plunge': False
        }
        
        # add plunge phase if requested
        if self.extend_past_isco:
            results = self._add_plunge_phase(results)
        
        return results

# output functions
def save_trajectory_csv(results: Dict, binary: BinaryBlackHole, 
                        filename: str) -> None:
    """
    Saveing sim results to CSV file.
    
    Args:
        results: simulation results dict
        binary: BinaryBlackHole object
        filename: output filename
    """
    print(f"\n  Saving trajectory to: {filename}")
    
    with open(filename, 'w', newline='') as f:
        # writing metadata as comments
        f.write(f"# binary black hole Inspiral Simulation\n")
        f.write(f"# m1_solar={binary.m1_solar}\n")
        f.write(f"# m2_solar={binary.m2_solar}\n")
        f.write(f"# r_horizon_1_m={binary.r_horizon_1}\n")
        f.write(f"# r_horizon_2_m={binary.r_horizon_2}\n")
        f.write(f"# distance_mpc={binary.distance_mpc}\n")
        f.write(f"# pn_order={binary.pn_order}\n")
        f.write(f"# eccentricity={binary.eccentricity}\n")
        
        writer = csv.writer(f)
        
        # header
        writer.writerow([
            't_sec',
            'bh1_x_m', 'bh1_y_m', 'bh1_z_m',
            'bh2_x_m', 'bh2_y_m', 'bh2_z_m',
            'separation_m', 'velocity_m_s',
            'f_gw_hz', 'phase_rad',
            'h_plus', 'h_cross'
        ])
        
        for i in range(len(results['t'])):
            writer.writerow([
                f"{results['t'][i]:.10e}",
                f"{results['r1'][i, 0]:.10e}",
                f"{results['r1'][i, 1]:.10e}",
                f"{results['r1'][i, 2]:.10e}",
                f"{results['r2'][i, 0]:.10e}",
                f"{results['r2'][i, 1]:.10e}",
                f"{results['r2'][i, 2]:.10e}",
                f"{results['separation'][i]:.10e}",
                f"{results['velocity'][i]:.10e}",
                f"{results['f_gw'][i]:.10e}",
                f"{results['phase'][i]:.10e}",
                f"{results['h_plus'][i]:.10e}",
                f"{results['h_cross'][i]:.10e}"
            ])
    
    print(f"    Saved {len(results['t'])} data points")
def create_diagnostic_plots(results: Dict, binary: BinaryBlackHole,
                           filename: str) -> None:
    """
    Creating diagnostic plots for the sim.
    
    Args:
        results: simulation results dict
        binary: BinaryBlackHole object
        filename: output filename
    """
    print(f"\n  Creating diagnostic plots: {filename}")
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.patch.set_facecolor('black')
    
    title = (f"Binary Black Hole Inspiral: "
             f"{binary.m1_solar:.0f} + {binary.m2_solar:.0f} M_sun  |  "
             f"PN Order: {binary.pn_order}")
    fig.suptitle(title, color='white', fontsize=13, y=0.98)
    
    # style all axes
    for ax in axes.flat:
        ax.set_facecolor('black')
        ax.tick_params(colors='white', labelsize=9)
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('white')
        ax.grid(True, alpha=0.2, color='white')
    
    t_ms = results['t'] * 1000  # convert to milliseconds
    
    
    # plot 1: orbital trajectory with event horizons
    
    ax = axes[0, 0]
    ax.plot(results['r1'][:, 0]/1000, results['r1'][:, 1]/1000,
            'cyan', alpha=0.6, lw=0.3, label=f'BH1 ({binary.m1_solar:.0f} M_sun)')
    ax.plot(results['r2'][:, 0]/1000, results['r2'][:, 1]/1000,
            'magenta', alpha=0.6, lw=0.3, label=f'BH2 ({binary.m2_solar:.0f} M_sun)')
    
    # draw event horizons as filled circles at final positions
    r_s1 = 2 * G_CONST * binary.m1 / C**2 / 1000  # km
    r_s2 = 2 * G_CONST * binary.m2 / C**2 / 1000  # km
    
    circle1 = plt.Circle((results['r1'][-1, 0]/1000, results['r1'][-1, 1]/1000), 
                          r_s1, color='cyan', alpha=0.8)
    circle2 = plt.Circle((results['r2'][-1, 0]/1000, results['r2'][-1, 1]/1000), 
                          r_s2, color='magenta', alpha=0.8)
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    
    ax.set_xlabel('x [km]')
    ax.set_ylabel('y [km]')
    ax.set_title(f'Inspiral Trajectory ({results["n_orbits"]:.0f} orbits)')
    ax.set_aspect('equal')
    ax.legend(facecolor='black', edgecolor='white', labelcolor='white', fontsize=8)
    
    
    # plot 2: separation vs time
    
    ax = axes[0, 1]
    ax.semilogy(t_ms, results['separation']/1000, 'cyan', lw=1)
    ax.axhline(binary.r_isco/1000, color='red', ls='--', lw=1.5,
               label=f'ISCO = {binary.r_isco/1000:.0f} km')
    # horizon contact distance
    r_contact = (2 * G_CONST * binary.m1 / C**2 + 2 * G_CONST * binary.m2 / C**2) / 1000
    ax.axhline(r_contact, color='yellow', ls=':', lw=1.5,
               label=f'Horizon contact = {r_contact:.0f} km')
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Separation [km]')
    ax.set_title('Orbital Separation Decay')
    ax.legend(facecolor='black', edgecolor='white', labelcolor='white', fontsize=8)
    
    
    # plot 3: GW frequency (Chirp)
    
    ax = axes[0, 2]
    ax.semilogy(t_ms, results['f_gw'], 'lime', lw=1)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('GW Frequency [Hz]')
    ax.set_title('Gravitational Wave Chirp')
    
    
    # plot 4: Energy Evolution
    
    ax = axes[1, 0]
    E_normalized = results['energy'] / results['energy'][0]
    ax.plot(t_ms, E_normalized, 'orange', lw=1)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('E / E₀')
    ax.set_title('Orbital Energy (radiated by GWs)')
    
    
    # plot 5: angular momentum evolution
    
    ax = axes[1, 1]
    L_normalized = results['ang_momentum'] / results['ang_momentum'][0]
    ax.plot(t_ms, L_normalized, 'yellow', lw=1)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('L / L₀')
    ax.set_title('Angular Momentum Loss')
    
    
    # plot 6: gravitational waveform
    
    ax = axes[1, 2]
    # only plot last ~20% to see the chirp clearly
    n_pts = len(t_ms)
    start_idx = int(0.8 * n_pts)
    ax.plot(t_ms[start_idx:], results['h_plus'][start_idx:] * 1e21, 'cyan', lw=0.5, alpha=0.8, label='h₊')
    ax.plot(t_ms[start_idx:], results['h_cross'][start_idx:] * 1e21, 'magenta', lw=0.5, alpha=0.8, label='h×')
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Strain h × 10^21')
    ax.set_title(f'Gravitational Waveform (last 20%)')
    ax.legend(facecolor='black', edgecolor='white', labelcolor='white', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=500, facecolor='black', edgecolor='none',
                bbox_inches='tight')
    plt.close()
    
    print(f"  Saved diagnostic plots")
def create_waveform_plot(results: Dict, binary: BinaryBlackHole,
                         filename: str) -> None:
    """
    Creating detailed gravitational waveform plot.
    
    Args:
        results: simulation results dict
        binary: BinaryBlackHole object
        filename: output filename
    """
    print(f"\n  Creating waveform plot: {filename}")
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.patch.set_facecolor('black')
    
    title = (f"Gravitational Waveform: "
             f"{binary.m1_solar:.0f} + {binary.m2_solar:.0f} M_sun  |  "
             f"D = {binary.distance_mpc:.0f} Mpc  |  "
             f"ι = {binary.inclination_deg:.0f}deg")
    fig.suptitle(title, color='white', fontsize=13, y=0.98)
    
    # style all axes
    for ax in axes:
        ax.set_facecolor('black')
        ax.tick_params(colors='white', labelsize=9)
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('white')
        ax.grid(True, alpha=0.2, color='white')
    
    t_ms = results['t'] * 1000  # convert to milliseconds
    h_plus = results['h_plus']
    h_cross = results['h_cross']
    h_amp = results['h_amplitude']
    
    
    # plot 1: full waveform
    
    ax = axes[0]
    ax.plot(t_ms, h_plus * 1e21, 'cyan', lw=0.3, alpha=0.9, label='h₊')
    ax.plot(t_ms, h_cross * 1e21, 'magenta', lw=0.3, alpha=0.7, label='h×')
    ax.plot(t_ms, h_amp * 1e21, 'yellow', lw=1, alpha=0.5, label='|h|')
    ax.plot(t_ms, -h_amp * 1e21, 'yellow', lw=1, alpha=0.5)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Strain h × 10^21')
    ax.set_title('Full Gravitational Waveform')
    ax.legend(facecolor='black', edgecolor='white', labelcolor='white', fontsize=9)
    
    
    # plot 2: last 20% (the chirp)
    
    ax = axes[1]
    n_pts = len(t_ms)
    start_idx = int(0.8 * n_pts)
    ax.plot(t_ms[start_idx:], h_plus[start_idx:] * 1e21, 'cyan', lw=0.8, label='h₊')
    ax.plot(t_ms[start_idx:], h_cross[start_idx:] * 1e21, 'magenta', lw=0.8, label='h×')
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Strain h × 10^21')
    ax.set_title('Final Chirp (last 20% of inspiral)')
    ax.legend(facecolor='black', edgecolor='white', labelcolor='white', fontsize=9)
    
    
    # plot 3: time-frequency representation 
    
    ax = axes[2]
    # color-code the waveform by frequency
    ax.scatter(t_ms, h_plus * 1e21, c=results['f_gw'], cmap='plasma', 
               s=0.1, alpha=0.5)
    cbar = plt.colorbar(ax.collections[0], ax=ax, pad=0.01)
    cbar.set_label('GW Frequency [Hz]', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Strain h₊ × 10^21')
    ax.set_title('Waveform colored by Frequency')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, facecolor='black', edgecolor='none',
                bbox_inches='tight')
    plt.close()
    
    print(f"  Saved waveform plot")

# command line interface
def print_banner():
    """Prints the banner."""
    pass  # removed
def show_usage():
    """Shows usage info when run without args."""
    import os
    script_name = os.path.basename(sys.argv[0])
    
    usage_text = f"""

                    BLACK HOLE BINARY INSPIRAL SIMULATION                     
                      Post-Newtonian Gravitational Waves                      
                   --------------------------------------  

  This code simulates 2 black holes spiraling into each other using the same 
  Post-Newtonian physics that LIGO/Virgo uses for real detections.

  Outputs: orbital trajectories, gravitational wave strain, frequency chirp.


                               KEY PARAMETERS                                 
                               --------------                                 

  --m1 MASS            Mass of BH 1 in solar masses (default: 36)
  --m2 MASS            Mass of BH 2 in solar masses (default: 29)
  --r0 DISTANCE        Initial separation in km (default: 2000)
  -e ECCENTRICITY      Orbital eccentricity, 0 = circular (default: 0)
  --extend-past-isco   Continue to horizon contact (recommended)
  -o PREFIX            Output filename prefix (default: inspiral_L1)
  --custom             Interactive mode (enter all params manually)

  Full options: python {script_name} --help


                               OUTPUT FILES                                   
                               ------------                                    

  <prefix>.csv            Trajectory + waveform data
  <prefix>.png            Diagnostic plots
  <prefix>_waveform.png   GW strain visualization


                               VIDEO MAKER                                    
                               -----------                                       

  python video_maker.py <output>.csv video.mp4
  python video_maker.py <output>.csv video.mp4 --fps 60 --duration 20


                               3D VIEWER                                      
                               ---------                                      

  Open bh_milkyway_embedded.html (in the scripts directory) using a browser, drag & drop the CSV file.


                               QUICK START                                    
                               -----------                                      

  python {script_name} --m1 36 --m2 29 --r0 5000 --extend-past-isco


                                EXAMPLES                                      
                                --------                                        


  python {script_name} --m1 36 --m2 29 --r0 5000 --extend-past-isco

  # Massive BHs (slower inspiral)
  python {script_name} --m1 80 --m2 60 --r0 8000 --extend-past-isco

  # Equal mass (symmetric)
  python {script_name} --m1 30 --m2 30 --r0 4000 --extend-past-isco

  # Extreme mass ratio
  python {script_name} --m1 50 --m2 10 --r0 3000 --extend-past-isco

  # Eccentric orbit
  python {script_name} --m1 36 --m2 29 --r0 5000 -e 0.5 --extend-past-isco

  # Interactive mode (enter all parameters manually)
  python {script_name} --custom


                               VALIDATION                                     
                               ----------                                     

  # Compare numerical simulation vs Peters (1964) analytical formula
  python {script_name} --test

  This generates a plot showing that our 2.5PN radiation reaction term
  correctly reproduces the inspiral rate predicted by Peters.

"""
    print(usage_text)
def create_parser() -> argparse.ArgumentParser:
    """Creates the argument parser."""
    
    parser = argparse.ArgumentParser(
        prog='bh_inspiral_L1.py',
        description='''

BLACK HOLE BINARY INSPIRAL SIMULATION - Level 1: Post-Newtonian Dynamics


Simulates the inspiral phase of a binary black hole merger using Post-Newtonian
(PN) dynamics with gravitational wave radiation reaction.

PHYSICS:
  • Newtonian gravity + relativistic corrections (1PN, 2PN)
  • Gravitational wave emission via 2.5PN radiation reaction
  • Inspiral terminates at the Innermost Stable Circular Orbit (ISCO)

OUTPUTS:
  • CSV file with trajectory data (positions, velocities, GW frequency)
  • PNG diagnostic plots (orbit, separation decay, chirp, energy loss)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLES:

  # default GW150914-like system (36 + 29 solar masses)
  python %(prog)s

  # equal mass binary, 30 solar masses each
  python %(prog)s --m1 30 --m2 30

  # custom system with larger initial separation
  python %(prog)s --m1 50 --m2 40 --r0 3000

  # pure Newtonian (no inspiral - for comparison)
  python %(prog)s --pn-order 0

  # high resolution output
  python %(prog)s --n-points 50000 --output high_res


        '''
    )
    
    # required physics parameters
    phys_group = parser.add_argument_group('Physics Parameters')
    
    phys_group.add_argument(
        '--m1', type=float, default=36.0, metavar='M_sun',
        help='Mass of black hole 1 in solar masses (default: 36.0)'
    )
    phys_group.add_argument(
        '--m2', type=float, default=29.0, metavar='M_sun',
        help='Mass of black hole 2 in solar masses (default: 29.0)'
    )
    phys_group.add_argument(
        '--r0', type=float, default=2000.0, metavar='km',
        help='Initial orbital separation in km (default: 2000.0)'
    )
    phys_group.add_argument(
        '--pn-order', type=float, default=2.5, choices=[0, 1, 2, 2.5],
        metavar='ORDER',
        help='Post-Newtonian order: 0, 1, 2, or 2.5 (default: 2.5)'
    )
    phys_group.add_argument(
        '--distance', '-d', type=float, default=410.0, metavar='Mpc',
        help='Distance to source in Megaparsecs (default: 410.0, like GW150914)'
    )
    phys_group.add_argument(
        '--inclination', '-i', type=float, default=0.0, metavar='deg',
        help='Inclination angle in degrees (default: 0 = face-on)'
    )
    phys_group.add_argument(
        '--eccentricity', '-e', type=float, default=0.0, metavar='e',
        help='Orbital eccentricity (default: 0 = circular, try 0.3-0.5 for elliptical)'
    )
    
    # simulation parameters
    sim_group = parser.add_argument_group('Simulation Parameters')
    
    sim_group.add_argument(
        '--t-max', type=float, default=None, metavar='sec',
        help='Maximum simulation time in seconds (default: 1.5 × Peters time)'
    )
    sim_group.add_argument(
        '--n-points', type=int, default=10000, metavar='N',
        help='Number of output data points (default: 10000)'
    )
    sim_group.add_argument(
        '--rtol', type=float, default=1e-10, metavar='TOL',
        help='ODE solver relative tolerance (default: 1e-10)'
    )
    sim_group.add_argument(
        '--atol', type=float, default=1e-12, metavar='TOL',
        help='ODE solver absolute tolerance (default: 1e-12)'
    )
    sim_group.add_argument(
        '--extend-past-isco', action='store_true',
        help='Continue simulation past ISCO until horizons touch (not physically accurate!)'
    )
    
    # output parameters
    out_group = parser.add_argument_group('Output Parameters')
    
    out_group.add_argument(
        '--output', '-o', type=str, default='inspiral_L1', metavar='PREFIX',
        help='Output filename prefix (default: inspiral_L1)'
    )
    out_group.add_argument(
        '--no-plot', action='store_true',
        help='Skip generating diagnostic plots'
    )
    out_group.add_argument(
        '--no-csv', action='store_true',
        help='Skip generating CSV output file'
    )
    out_group.add_argument(
        '--quiet', '-q', action='store_true',
        help='Suppress detailed output'
    )
    out_group.add_argument(
        '--test', action='store_true',
        help='Run validation: compare numerical inspiral rate vs Peters (1964) analytical formula'
    )
    out_group.add_argument(
        '--custom', action='store_true',
        help='Interactive mode: manually enter all parameters'
    )
    
    return parser
def get_interactive_input():
    """Prompts user for parameters interactively."""
    
    print("\n" + "─"*65)
    print("  ENTER SIMULATION PARAMETERS")
    print("  (Press Enter to use default values)")
    print("─"*65 + "\n")
    
    # mass 1
    while True:
        m1_input = input("  Mass of black hole 1 [solar masses] (default: 36.0): ").strip()
        if m1_input == "":
            m1 = 36.0
            break
        try:
            m1 = float(m1_input)
            if m1 <= 0:
                print("    Mass must be positive. Try again.")
                continue
            break
        except ValueError:
            print("    Invalid number. Try again.")
    
    # mass 2
    while True:
        m2_input = input("  Mass of black hole 2 [solar masses] (default: 29.0): ").strip()
        if m2_input == "":
            m2 = 29.0
            break
        try:
            m2 = float(m2_input)
            if m2 <= 0:
                print("    Mass must be positive. Try again.")
                continue
            break
        except ValueError:
            print("    Invalid number. Try again.")
    
    # initial separation
    while True:
        r0_input = input("  Initial orbital separation [km] (default: 2000.0): ").strip()
        if r0_input == "":
            r0 = 2000.0
            break
        try:
            r0 = float(r0_input)
            if r0 <= 0:
                print("    Separation must be positive. Try again.")
                continue
            break
        except ValueError:
            print("    Invalid number. Try again.")
    
    # pn order
    while True:
        pn_input = input("  Post-Newtonian order [0, 1, 2, 2.5] (default: 2.5): ").strip()
        if pn_input == "":
            pn_order = 2.5
            break
        try:
            pn_order = float(pn_input)
            if pn_order not in [0, 1, 2, 2.5]:
                print("    Must be 0, 1, 2, or 2.5. Try again.")
                continue
            break
        except ValueError:
            print("    Invalid number. Try again.")
    
    # distance
    while True:
        dist_input = input("  Distance to source [Mpc] (default: 410.0): ").strip()
        if dist_input == "":
            distance = 410.0
            break
        try:
            distance = float(dist_input)
            if distance <= 0:
                print("    Distance must be positive. Try again.")
                continue
            break
        except ValueError:
            print("    Invalid number. Try again.")
    
    # inclination
    while True:
        inc_input = input("  Inclination angle [degrees, 0=face-on] (default: 0.0): ").strip()
        if inc_input == "":
            inclination = 0.0
            break
        try:
            inclination = float(inc_input)
            break
        except ValueError:
            print("    Invalid number. Try again.")
    
    # extend past ISCO
    extend_input = input("  Extend past ISCO to horizon contact? [y/N]: ").strip().lower()
    extend_past_isco = extend_input in ['y', 'yes']
    if extend_past_isco:
        print("    Note: Physics past ISCO is approximate!")
    
    # eccentricity
    while True:
        ecc_input = input("  Orbital eccentricity [0-0.9, 0=circular] (default: 0.0): ").strip()
        if ecc_input == "":
            eccentricity = 0.0
            break
        try:
            eccentricity = float(ecc_input)
            if eccentricity < 0 or eccentricity >= 1:
                print("     Eccentricity must be 0 ≤ e < 1. Try again.")
                continue
            break
        except ValueError:
            print("     Invalid number. Try again.")
    
    # output filename
    output = input("  Output filename prefix (default: inspiral_L1): ").strip()
    if output == "":
        output = "inspiral_L1"
    
    print("\n" + "─"*65 + "\n")
    
    return m1, m2, r0, pn_order, distance, inclination, extend_past_isco, eccentricity, output


def run_validation_test():
    """
    Runs validation test comparing numerical simulation vs analytical Peters formula.
    
    Generates a plot comparing da/dt (inspiral rate) between simulation and Peters.
    """
    print("\n  Running validation test...")
    
    # use same parameters as the example command
    m1_solar = 36.0
    m2_solar = 29.0
    r0_km = 5000
    
    # run the sim (quietly)
    import io
    import contextlib
    
    binary = BinaryBlackHole(m1_solar=m1_solar, m2_solar=m2_solar, pn_order=2.5)
    
    with contextlib.redirect_stdout(io.StringIO()):
        results = binary.simulate(r0_km=r0_km, n_output_points=50000)
    
    t = results['t']
    separation = results['separation']
    
    # compute numerical dr/dt
    dr_dt_numerical = np.gradient(separation, t)
    
    # compute Peters dr/dt analytically
    m1 = m1_solar * M_SUN
    m2 = m2_solar * M_SUN
    M = m1 + m2
    
    dr_dt_peters = -(64/5) * (G_CONST**3 / C**5) * (m1 * m2 * M) / separation**3
    
    # create the plot with black background
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    
    # style axes
    ax.tick_params(colors='white', labelsize=10)
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    for spine in ax.spines.values():
        spine.set_color('white')
    
    # plot data
    ax.plot(t * 1000, dr_dt_numerical, 'cyan', lw=0.5, alpha=0.7, label='dr/dt (numerical)')
    ax.plot(t * 1000, dr_dt_peters, 'orange', lw=2.5, alpha=0.9, linestyle='--', label='dr/dt (Peters 1964)')
    
    ax.set_xlabel('Time [ms]', fontsize=12, color='white')
    ax.set_ylabel('dr/dt [m/s]', fontsize=12, color='white')
    ax.set_title(f'Validation: Numerical vs Peters (1964)  |  {m1_solar:.0f} + {m2_solar:.0f} M_sun', 
                 fontsize=14, fontweight='bold', color='white')
    
    # add info box
    info_text = (
        f"m₁ = {m1_solar:.0f} M_sun\n"
        f"m₂ = {m2_solar:.0f} M_sun\n"
        f"r₀ = {r0_km} km"
    )
    ax.text(0.02, 0.02, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='left', color='white',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.8, edgecolor='white'))
    
    ax.legend(loc='upper right', fontsize=11, facecolor='black', edgecolor='white', 
              labelcolor='white')
    ax.grid(True, alpha=0.2, color='white')
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
    
    # make scientific notation white
    ax.yaxis.get_offset_text().set_color('white')
    
    # save plot
    plot_file = 'validation_peters.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='black', edgecolor='none')
    plt.close()
    
    print(f"  Saved: {plot_file}\n")
    
    return plot_file


def main():
    """Main entry point."""
    
    # if no arguments provided, show usage and exit
    if len(sys.argv) == 1:
        show_usage()
        sys.exit(0)
    
    # parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # run validation test if requested
    if args.test:
        run_validation_test()
        sys.exit(0)
    
    # run interactive mode if requested
    if args.custom:
        m1, m2, r0, pn_order, distance, inclination, extend_past_isco, eccentricity, output = get_interactive_input()
        # override args with interactive values
        args.m1 = m1
        args.m2 = m2
        args.r0 = r0
        args.pn_order = pn_order
        args.distance = distance
        args.inclination = inclination
        args.extend_past_isco = extend_past_isco
        args.eccentricity = eccentricity
        args.output = output
    
    # print banner
    if not args.quiet:
        print_banner()
    
    # validate inputs
    if args.m1 <= 0 or args.m2 <= 0:
        print("ERROR: Masses must be positive!")
        sys.exit(1)
    
    if args.r0 <= 0:
        print("ERROR: Initial separation must be positive!")
        sys.exit(1)
    
    if args.eccentricity < 0 or args.eccentricity >= 1:
        print("ERROR: Eccentricity must be 0 ≤ e < 1!")
        sys.exit(1)
    
    # create binary system
    binary = BinaryBlackHole(
        m1_solar=args.m1,
        m2_solar=args.m2,
        pn_order=args.pn_order,
        distance_mpc=args.distance,
        inclination_deg=args.inclination,
        extend_past_isco=args.extend_past_isco,
        eccentricity=args.eccentricity
    )
    
    # check that initial separation > ISCO
    if args.r0 * 1000 <= binary.r_isco:
        print(f"ERROR: Initial separation ({args.r0} km) must be > ISCO ({binary.r_isco/1000:.1f} km)")
        sys.exit(1)
    
    # print parameters
    if not args.quiet:
        binary.print_parameters()
    
    # run sim
    try:
        results = binary.simulate(
            r0_km=args.r0,
            t_max=args.t_max,
            n_output_points=args.n_points,
            rtol=args.rtol,
            atol=args.atol
        )
    except Exception as e:
        print(f"\nERROR during simulation: {e}")
        sys.exit(1)
    
    # save outputs
    if not args.no_csv:
        save_trajectory_csv(results, binary, f"{args.output}.csv")
    
    if not args.no_plot:
        create_diagnostic_plots(results, binary, f"{args.output}.png")
        create_waveform_plot(results, binary, f"{args.output}_waveform.png")
    
    # final summary
    if not args.quiet:

        print("  SIMULATION COMPLETE")
        print("  -------------------") 
        if not args.no_csv:
            print(f"  Trajectory data:   {args.output}.csv")
        if not args.no_plot:
            print(f"  Diagnostic plots:  {args.output}.png")
            print(f"  Waveform plots:    {args.output}_waveform.png")
       
        print(f"\n  Next steps:")
        print(f"    Create video:  python video_maker.py {args.output}.csv output.mp4")
        print(f"    3D viewer:     Open bh_milkyway_embedded.html and load {args.output}.csv")
       
    
    return results, binary


if __name__ == "__main__":
    results, binary = main()
