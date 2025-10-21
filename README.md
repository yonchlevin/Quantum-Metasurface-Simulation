# Quantum-Metasurface-Simulation
## Metasurface Reflection Simulation

This project simulates the reflection of an electromagnetic wave (modeled as a Gaussian beam) from a 2D atomic metasurface. The primary goal is to analyze how thermal disorder in the atomic positions affects the metasurface's optical properties, specifically its reflectivity and the phase of the reflected field.

## Project Structure

The simulation is organized into two main Python scripts:

* `electric_field_calculator.py`: This module contains the core physics functions required for the simulation. It is responsible for:

  * Calculating the dyadic Green's function.

  * Generating the 2D atomic array, including thermal displacement.

  * Constructing the full Green's matrix for the entire system.

  * Modeling a Gaussian beam as the incident electric field.

  * Solving the linear system to compute the total scattered electric field.

* `simulation.py`: This is the main executable script that orchestrates the simulation. Its key responsibilities include:

  * Setting the physical parameters of the simulation (wavelength, number of atoms, etc.).

  * Looping through a range of temperatures (modeled as variance in atomic positions).

  * Running multiple repetitions for each temperature to ensure statistical significance.

  * Calculating the final reflectivity and phase from the scattered fields.

  * Saving the raw simulation data to `.npy` files.

  * Generating and saving summary plots of the results.

## How to Run the Simulation

Configure Parameters: Open the `simulation.py` file. At the bottom, within the if __name__ == '__main__': block, you can adjust the core parameters of the simulation:

WAVELENGTH: The wavelength of the incident light.

LATTICE_CONSTANTS: The spacing between atoms in the array.

NUM_ATOMS: The total number of atoms in the metasurface (must be a perfect square).

BEAM_WAIST: The waist radius of the incident Gaussian beam.

Execute the Script: Run the simulation from your terminal:

```
python simulation.py
```

## Expected Output

The script will generate the following outputs in the project's root directory:

results/ directory: This folder will be created to store plots showing the mean electric field versus the z-position for each individual simulation run. This helps in visualizing the propagation of the reflected wave.

.png image files: Summary plots showing the final, averaged results:

reflectivity_vs_temp_...png: A plot of normalized reflectivity as a function of temperature.

phase_vs_temp_...png: A plot of the reflected field's phase as a function of temperature.

.npy data files: NumPy files containing the raw data from the simulation, allowing for further analysis if needed.

This simulation provides a framework for studying light-matter interactions in 2D atomic arrays and can be extended to explore other phenomena.