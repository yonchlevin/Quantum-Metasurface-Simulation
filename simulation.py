"""
Main script to run the metasurface simulation.

This script simulates the reflection of a Gaussian beam from a 2D atomic array,
studying the effect of thermal disorder (temperature) on reflectivity and phase.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import epsilon_0
from numpy.typing import NDArray

# Import the core calculation functions from the refactored module
from electric_field_calculator import (
    calculate_greens_matrix_for_lattice, 
    get_incident_field_e0, 
    compute_electric_field, 
    create_full_greens_matrix, 
    create_atom_array_2d
)

COMPLEX_DTYPE = np.complex128


def run_simulation(
    wavelength: float,
    num_atoms: int,
    alpha: complex,
    lattice_constants: NDArray[np.float64],
    beam_waist: float,
    check_temp: bool = True,
):
    """
    Runs the main simulation loop to calculate reflectivity and phase as a
    function of atomic position variance (temperature).

    Args:
        wavelength: Wavelength of the incident light.
        num_atoms: Total number of atoms in the metasurface.
        alpha: Atomic polarizability.
        lattice_constants: An array of lattice constants 'a' to test.
        beam_waist: The waist radius of the incident Gaussian beam.
        check_temp: If True, simulates across a range of temperatures.
    """
    temperatures = np.linspace(0, 0.05, 5) if check_temp else [0.0]
    num_reps = 50  # Number of repetitions for statistical averaging

    # Arrays to store the final averaged results
    reflectivity = np.zeros(len(temperatures))
    reflectivity_std_err = np.zeros(len(temperatures))
    phase_at_max_r = np.zeros(len(temperatures))
    phase_std_err = np.zeros(len(temperatures))

    for i, temp_variance in enumerate(temperatures):
        
        # Arrays to store results for each repetition at a single temperature
        rep_reflectivities = np.zeros(num_reps)
        rep_phases = np.zeros(num_reps)

        print(f"--- Processing Temperature Variance: {temp_variance:.3f} ---")
        for k in range(num_reps):
            # For zero temperature, the lattice is perfect, so all repetitions are identical.
            # We can skip redundant calculations.
            if temp_variance == 0 and k > 0:
                rep_reflectivities[k] = rep_reflectivities[0]
                rep_phases[k] = rep_phases[0]
                continue

            print(f"  Repetition {k + 1}/{num_reps}...")
            
            # 1. Calculate the Green's matrix for the current disordered lattice
            g_matrices = calculate_greens_matrix_for_lattice(
                lattice_constants=lattice_constants,
                wavelength=wavelength,
                num_atoms=num_atoms,
                temperature_variance=temp_variance
            )
            g_matrix = g_matrices[0]

            # 2. Calculate scattered field and resulting reflectivity/phase
            max_r, phase = calculate_reflectivity_and_phase(
                g_matrix=g_matrix,
                num_atoms=num_atoms,
                wavelength=wavelength,
                lattice_constant=lattice_constants[0],
                alpha=alpha,
                beam_waist=beam_waist,
                temp_variance=temp_variance,
                rep_index=k
            )
            rep_reflectivities[k] = max_r
            rep_phases[k] = phase

        # Aggregate results for this temperature
        reflectivity[i] = np.mean(rep_reflectivities)
        reflectivity_std_err[i] = np.std(rep_reflectivities) / np.sqrt(num_reps)
        phase_at_max_r[i] = np.mean(rep_phases)
        phase_std_err[i] = np.std(rep_phases) / np.sqrt(num_reps)

    # Save and plot final results after all temperatures are processed
    save_and_plot_results(
        temperatures=temperatures,
        reflectivity=reflectivity,
        reflectivity_std_err=reflectivity_std_err,
        phase=phase_at_max_r,
        phase_std_err=phase_std_err,
        num_atoms=num_atoms,
        lattice_constant=lattice_constants[0],
        wavelength=wavelength,
    )


def calculate_reflectivity_and_phase(
    g_matrix: NDArray[COMPLEX_DTYPE],
    num_atoms: int,
    wavelength: float,
    lattice_constant: float,
    alpha: complex,
    beam_waist: float,
    temp_variance: float,
    rep_index: int,
) -> tuple[float, float]:
    """
    For a single instance of a disordered array, calculates the scattered
    field, observes it at various z positions, and finds the maximum 
    reflectivity and its corresponding phase.

    Returns:
        A tuple containing:
        - max_reflectivity (float): The maximum calculated intensity.
        - phase (float): The phase of the field at max reflectivity, in units of pi.
    """
    grid_size = int(np.sqrt(num_atoms))
    dx_dy = lattice_constant

    # Incident field on the metasurface at z=0
    e0_incident = get_incident_field_e0(
        grid_size=grid_size, waist_radius=beam_waist, z=0,
        wavelength=wavelength, dx=dx_dy, dy=dx_dy,
    )

    # Total electric field including scattering from the metasurface atoms
    e_scattered_total = compute_electric_field(
        g_matrix, e0_incident, alpha, epsilon_0, wavelength
    )
    
    # Observe the propagated field at various z positions to find the reflected wave
    z_dist_in_wl = 3
    num_z_steps = (z_dist_in_wl * 100) + 1
    z_positions = np.linspace(-z_dist_in_wl * wavelength, z_dist_in_wl * wavelength, num_z_steps)
    
    # Location of the original (perfect) lattice for field calculation
    perfect_lattice_locs = create_atom_array_2d(num_atoms, lattice_constant, lattice_constant, 0.0)

    mean_real_reflected = np.zeros(num_z_steps)
    mean_im_reflected = np.zeros(num_z_steps)
    mean_real_in = np.zeros(num_z_steps)
    mean_im_in = np.zeros(num_z_steps)

    for j, z in enumerate(z_positions):
        e_in_at_z = get_incident_field_e0(grid_size, beam_waist, z, wavelength, dx_dy, dx_dy)
        observer_locs = create_atom_array_2d(num_atoms, lattice_constant, lattice_constant, 0.0, z)
        g_propagate = create_full_greens_matrix(2 * np.pi / wavelength, observer_locs, perfect_lattice_locs)
        
        # E_obs(z) = E_in(z) + Integral(G_prop * P) = E_in(z) + prefactor * G_prop * E_scat
        prefactor = (4 * np.pi**2 * alpha) / (epsilon_0 * wavelength**2)
        e_observed_at_z = e_in_at_z + prefactor * (g_propagate @ e_scattered_total)
        
        e_observed_x = e_observed_at_z[::3].reshape((grid_size, grid_size))
        e_in_x = e_in_at_z[::3].reshape((grid_size, grid_size))
        
        mean_real_reflected[j] = np.mean(np.real(e_observed_x))
        mean_im_reflected[j] = np.mean(np.imag(e_observed_x))
        mean_real_in[j] = np.mean(np.real(e_in_x))
        mean_im_in[j] = np.mean(np.imag(e_in_x))

    # Normalize fields for plotting and comparison
    norm_factor = (mean_im_in.max()) * 2
    if norm_factor == 0: norm_factor = 1 # Avoid division by zero
    
    mean_real_in /= norm_factor
    mean_im_in /= norm_factor
    mean_real_reflected /= norm_factor
    mean_im_reflected /= norm_factor

    # Find max reflectivity in the reflection region (z < -0.5 * lambda)
    reflection_mask = z_positions < -0.5 * wavelength
    intensity = mean_real_reflected**2 + mean_im_reflected**2
    
    if not np.any(reflection_mask): return 0.0, 0.0

    max_reflectivity = np.max(intensity[reflection_mask])
    max_reflectivity_idx = np.argmax(intensity[reflection_mask])
    
    # Get the phase at the point of maximum reflectivity
    phase = np.angle(mean_real_reflected[reflection_mask][max_reflectivity_idx] + 1j * mean_im_reflected[reflection_mask][max_reflectivity_idx])

    _plot_field_vs_z(
        z_positions / wavelength,
        mean_real_in, mean_im_in,
        mean_real_reflected, im_reflected,
        lattice_constant / wavelength, temp_variance, rep_index,
    )
    
    return max_reflectivity, phase / np.pi


def _plot_field_vs_z(
    loc, real_in, im_in, real_reflected, im_reflected,
    a_div_lambda, temp_variance, rep_index
):
    """Helper function to plot the electric field vs. z-position for one repetition."""
    plt.figure(figsize=(10, 6))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    plt.plot(loc, real_in, label="Re[Input]", color=colors[3])
    plt.plot(loc, im_in, label="Im[Input]", color=colors[1])
    plt.plot(loc, im_reflected, label="Im[Observed Field]", linestyle='--', color=colors[2])
    plt.plot(loc, real_reflected, label="Re[Observed Field]", linestyle='--', color=colors[0])
    
    plt.legend()
    plt.title(rf'Mean Field vs. $z$ for $a/\lambda={a_div_lambda:.2f}$, Var={temp_variance:.3f}')
    plt.xlabel(r"$z/\lambda$")
    plt.ylabel(r"Normalized Mean Field")
    plt.ylim(-1.1, 1.1)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    if not os.path.exists("results"):
        os.makedirs("results")
        
    filename = f"results/field_vs_z_a={a_div_lambda:.2f}_t={temp_variance:.3f}_rep={rep_index}.png"
    plt.savefig(filename)
    plt.close()


def save_and_plot_results(
    temperatures, reflectivity, reflectivity_std_err,
    phase, phase_std_err, num_atoms, lattice_constant, wavelength
):
    """Saves the final simulation data and plots the summary graphs."""
    a_div_lambda = lattice_constant / wavelength
    prefix = f"N={num_atoms}_a={a_div_lambda:.3f}"
    
    # --- Save Data ---
    np.save(f"{prefix}_temps.npy", temperatures)
    np.save(f"{prefix}_reflectivity.npy", reflectivity)
    np.save(f"{prefix}_reflectivity_stderr.npy", reflectivity_std_err)
    np.save(f"{prefix}_phase.npy", phase)
    np.save(f"{prefix}_phase_stderr.npy", phase_std_err)
    print(f"\nFinal results saved with prefix: {prefix}")

    # --- Plot Reflectivity vs. Temperature ---
    plt.figure(figsize=(10, 6))
    reflectivity_norm = reflectivity / reflectivity[0] if reflectivity[0] != 0 else reflectivity
    reflectivity_std_err_norm = reflectivity_std_err / reflectivity[0] if reflectivity[0] != 0 else reflectivity_std_err
    
    plt.errorbar(
        temperatures, reflectivity_norm, yerr=reflectivity_std_err_norm,
        fmt='o', markersize=5, capsize=3, label=f"a = {a_div_lambda:.2f}λ"
    )
    plt.title(f"Normalized Reflectivity vs. Temperature (N={num_atoms})")
    plt.xlabel("Standard Deviation of Atom Shift [in units of a]")
    plt.ylabel("Normalized Reflectivity")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.ylim(bottom=max(0, plt.ylim()[0] - 0.1), top=1.1)
    plt.legend()
    plt.savefig(f"reflectivity_vs_temp_{prefix}.png")
    plt.close()

    # --- Plot Phase vs. Temperature ---
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        temperatures, phase, yerr=phase_std_err, fmt='o', markersize=5, capsize=3
    )
    plt.title(f"Reflected Phase vs. Temperature (N={num_atoms})")
    plt.xlabel("Standard Deviation of Atom Shift [in units of a]")
    plt.ylabel(r"Phase at Max Reflectivity [$\pi \cdot$ rad]")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(f"phase_vs_temp_{prefix}.png")
    plt.close()
    
    print("Summary plots generated successfully.")


if __name__ == '__main__':
    # --- Simulation Parameters ---
    WAVELENGTH = 0.7e-6
    # Atomic polarizability for a two-level atom on resonance
    ALPHA = 1j * (3 / (4 * np.pi**2)) * epsilon_0 * (WAVELENGTH**3)
    
    # Metasurface lattice constant
    LATTICE_CONSTANTS = np.array([0.2 * WAVELENGTH])
    
    # Number of atoms (must be a perfect square)
    GRID_SIDE = 20
    NUM_ATOMS = GRID_SIDE**2

    # Gaussian beam waist radius, scaled with the size of the array
    BEAM_WAIST = np.sqrt(NUM_ATOMS) * 0.3 * LATTICE_CONSTANTS[0]

    print(f"### Starting Simulation: N={NUM_ATOMS} ###")
    run_simulation(
        wavelength=WAVELENGTH,
        num_atoms=NUM_ATOMS,
        alpha=ALPHA,
        lattice_constants=LATTICE_CONSTANTS,
        beam_waist=BEAM_WAIST,
        check_temp=True
    )
