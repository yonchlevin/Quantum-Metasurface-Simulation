"""
Core functions for calculating electromagnetic fields and Green's functions
for metasurface simulations.
"""
import numpy as np
from numpy.typing import NDArray

# Define a consistent complex dtype for arrays
COMPLEX_DTYPE = np.complex128

def dyadic_greens_function(k: float, r1: NDArray[np.float64], r2: NDArray[np.float64]) -> NDArray[COMPLEX_DTYPE]:
    """
    Calculates the dyadic Green's function for electromagnetic waves.

    This function describes the electric field at position r1 due to a point
    source at r2.

    Args:
        k: The wave number (2 * pi / wavelength).
        r1: The position vector of the observation point (shape: (3,)).
        r2: The position vector of the source point (shape: (3,)).

    Returns:
        A 3x3 complex numpy array representing the Green's function tensor.
    """
    r_vec = r1 - r2
    r_norm = np.linalg.norm(r_vec)

    # Handle the singularity at r = 0, where the function is undefined.
    if r_norm == 0:
        return np.zeros((3, 3), dtype=COMPLEX_DTYPE)

    ikr = 1j * k * r_norm
    exp_factor = np.exp(ikr) / (4 * np.pi * r_norm)
    k_r_squared = (k * r_norm) ** 2
    identity = np.identity(3)
    outer_product = np.outer(r_vec, r_vec)

    term1 = (1 + (ikr - 1) / k_r_squared) * identity
    term2 = (-1 + (3 - 3 * ikr) / k_r_squared) / (r_norm ** 2) * outer_product
    
    g_tensor = exp_factor * (term1 + term2)
    return g_tensor

def create_atom_array_2d(
    num_atoms: int, 
    lattice_const_a: float, 
    lattice_const_b: float, 
    variance: float, 
    z_plane: float = 0.0
) -> NDArray[np.float64]:
    """
    Initializes a 2D square array of atom locations with thermal displacement.

    Args:
        num_atoms: Total number of atoms. Assumes sqrt(num_atoms) is an integer.
        lattice_const_a: Lattice constant in the x-axis.
        lattice_const_b: Lattice constant in the y-axis.
        variance: Variance for the normal distribution of atom displacement, 
                  representing thermal effects.
        z_plane: The z-coordinate for all atoms.

    Returns:
        A numpy array of shape (num_atoms, 3) with the 3D coordinates of each atom.
    """
    if not np.sqrt(num_atoms).is_integer():
        raise ValueError("The square root of num_atoms must be an integer.")
    
    side_length = int(np.sqrt(num_atoms))
    locations = np.zeros((num_atoms, 3))

    for i in range(side_length):
        for j in range(side_length):
            # Model thermal displacement with a 2D normal distribution
            # by sampling a random angle and a normally distributed distance.
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.normal(loc=0, scale=np.sqrt(variance))
            
            x_shift = distance * np.cos(angle)
            y_shift = distance * np.sin(angle)
            
            index = i * side_length + j
            locations[index, :] = np.array([
                lattice_const_a * (j + x_shift), 
                lattice_const_b * (i + y_shift), 
                z_plane
            ])
            
    return locations

def calculate_greens_matrix_for_lattice(
    lattice_constants: list[float], 
    num_atoms: int, 
    wavelength: float, 
    temperature_variance: float = 0.0
) -> list[NDArray[COMPLEX_DTYPE]]:
    """
    Calculates the full Green's matrix for each lattice constant provided.

    Args:
        lattice_constants: A list of lattice constants 'a' to simulate.
        num_atoms: The total number of atoms in the array.
        wavelength: The wavelength of the incident light.
        temperature_variance: The variance in atom positions due to temperature.

    Returns:
        A list of full 3N x 3N Green's function matrices, one for each lattice constant.
    """
    k = 2 * np.pi / wavelength
    greens_matrices = []
    
    for a in lattice_constants:
        atom_locations = create_atom_array_2d(
            num_atoms=num_atoms,
            lattice_const_a=a,
            lattice_const_b=a,
            variance=temperature_variance
        )
        g_matrix = create_full_greens_matrix(k, atom_locations)
        greens_matrices.append(g_matrix)
        
    return greens_matrices

def create_full_greens_matrix(
    k: float, 
    positions1: NDArray[np.float64], 
    positions2: NDArray[np.float64] | None = None
) -> NDArray[COMPLEX_DTYPE]:
    """
    Constructs the full 3N x 3M Green's matrix between two sets of atom positions.

    Args:
        k: The wave number.
        positions1: An array of N atom positions (shape: (N, 3)).
        positions2: An optional array of M atom positions (shape: (M, 3)). 
                    If None, positions1 is used (calculating self-interaction).

    Returns:
        The complete 3N x 3M Green's function matrix.
    """
    if positions2 is None:
        positions2 = positions1

    n1 = len(positions1)
    n2 = len(positions2)
    full_matrix = np.zeros((3 * n1, 3 * n2), dtype=COMPLEX_DTYPE)

    for n in range(n1):
        for m in range(n2):
            g_tensor = dyadic_greens_function(k, positions1[n], positions2[m])
            full_matrix[3 * n : 3 * n + 3, 3 * m : 3 * m + 3] = g_tensor
            
    return full_matrix

def compute_electric_field(
    g_matrix: NDArray[COMPLEX_DTYPE], 
    e0_incident: NDArray[COMPLEX_DTYPE], 
    alpha: complex, 
    epsilon_0: float, 
    wavelength: float
) -> NDArray[COMPLEX_DTYPE]:
    """
    Computes the total electric field scattered by an array of atoms.

    This solves the linear system: (I - C*G) * E_total = E_incident,
    where C is a coefficient involving polarizability.

    Args:
        g_matrix: The 3N x 3N Green's function matrix.
        e0_incident: The incident electric field vector (shape: (3N,)).
        alpha: The atomic polarizability.
        epsilon_0: The vacuum permittivity.
        wavelength: The wavelength of the light.

    Returns:
        The total electric field vector (shape: (3N,)).
    """
    prefactor_coeff = (4 * np.pi**2 * alpha) / (epsilon_0 * wavelength**2)
    
    identity_matrix = np.eye(g_matrix.shape[0])
    system_matrix = identity_matrix - prefactor_coeff * g_matrix
    
    # Solve the linear system for the total E field. This is more numerically
    # stable and efficient than computing the inverse directly.
    e_total = np.linalg.solve(system_matrix, e0_incident)
    
    return e_total
    
def gaussian_beam(
    grid_size: int, 
    waist_radius: float, 
    z: float, 
    wavelength: float, 
    dx: float, 
    dy: float
) -> NDArray[COMPLEX_DTYPE]:
    """
    Generates a 2D Gaussian beam field at a specified z-location.

    Args:
        grid_size: The number of points along one dimension of the square grid.
        waist_radius: The beam waist radius at z=0.
        z: The propagation distance.
        wavelength: The wavelength of the light.
        dx: Grid spacing along the x-axis.
        dy: Grid spacing along the y-axis.

    Returns:
        A 2D numpy array representing the complex Gaussian field at z.
    """
    k = 2 * np.pi / wavelength
    rayleigh_range = np.pi * waist_radius**2 / wavelength

    if z == 0:
        w_z = waist_radius
        r_z = np.inf  # Wavefront is planar at the waist
        gouy_phase = 0
    else:
        w_z = waist_radius * np.sqrt(1 + (z / rayleigh_range)**2)
        r_z = z * (1 + (rayleigh_range / z)**2)
        gouy_phase = np.arctan(z / rayleigh_range)

    x = np.linspace(-grid_size / 2, grid_size / 2 - 1, grid_size) * dx
    y = np.linspace(-grid_size / 2, grid_size / 2 - 1, grid_size) * dy
    xx, yy = np.meshgrid(x, y)
    
    radial_dist_sq = xx**2 + yy**2
    
    amplitude = (waist_radius / w_z) * np.exp(-radial_dist_sq / w_z**2)
    phase_term = np.exp(1j * (k * z + (k * radial_dist_sq) / (2 * r_z) - gouy_phase))
    
    return amplitude * phase_term

def get_incident_field_e0(
    grid_size: int, 
    waist_radius: float, 
    z: float, 
    wavelength: float, 
    dx: float, 
    dy: float
) -> NDArray[COMPLEX_DTYPE]:
    """
    Generates the 3N incident electric field vector E0 for a Gaussian beam.

    The beam is circularly polarized in the xy-plane.

    Args:
        grid_size: The number of points along one dimension of the square grid.
        waist_radius: The beam waist radius at z=0.
        z: The propagation distance.
        wavelength: The wavelength of the light.
        dx: Grid spacing along the x-axis.
        dy: Grid spacing along the y-axis.

    Returns:
        A 1D array of shape (3 * grid_size**2,) representing the incident E-field vector.
    """
    num_atoms = grid_size * grid_size
    u_field = gaussian_beam(grid_size, waist_radius, z, wavelength, dx, dy)

    # Flatten the 2D field and create the 3N vector for E0
    e0 = np.zeros(3 * num_atoms, dtype=COMPLEX_DTYPE)
    
    # E_x component
    e0[0::3] = u_field.flatten()
    # E_y component for circular polarization
    e0[1::3] = 1j * u_field.flatten()
    
    return e0
