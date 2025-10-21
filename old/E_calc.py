import matplotlib.pyplot as plt
import numpy as np


def dyadic_greens_function(k, r1, r2):
    # TODO: Find the right polarization: What is the "d"? Ask Nadav
    #  use e_i=[1,i,0] (S3 supp Efi), here denoted by r_vec. Note it is Outer product
    r_vec = r1 - r2
    r = np.linalg.norm(r_vec)

    # Handle the singularity at r = 0
    if r == 0:
        return np.zeros((3, 3), dtype=complex)

    ikr = 1j * k * r
    exp_factor = np.exp(ikr) / (4 * np.pi * r)
    k2_r2 = (k * r) ** 2
    delta_ij = np.identity(3)

    # G_ij = np.zeros((3, 3), dtype=complex)
    # for i in range(3):
    #     for j in range(3):
    #         term1 = (1 + (ikr - 1) / k2_r2) * delta_ij[i, j]
    #         term2 = (-1 + (3 - 3 * ikr) / k2_r2) * (r_vec[i] * r_vec[j]) / r ** 2
    #         G_ij[i, j] = exp_factor * (term1 + term2)

    G_ij = exp_factor * (
                (1 + (ikr - 1) / k2_r2) * delta_ij + ((-1 + (3 - 3 * ikr) / k2_r2) / r ** 2) * np.outer(r_vec, r_vec))
    # if G_ij[G_ij > 0].any():
    #     print("real: ", np.real(G_ij[G_ij > 0]*7e-7).min(), np.real(G_ij[G_ij > 0]*7e-7).max())
    #     print("imaginary: ", np.imag(G_ij[G_ij > 0]*7e-7).min(), np.imag(G_ij[G_ij > 0]*7e-7).max())
    return G_ij


def create_2d_arr(N, a, b, variance, z=0):
    """
    inits an array of size [N,3]. each row is the location of an atom in the
    array.
    :param variance: the temperature of the system. effects the location
    :param N: int - total number of atoms, assumes sqrt(N) is an integer
    :param a: int - lattice constant in x-axis
    :param b: int - lattice constant in y-axis
    :param z: float - z location of the atoms
    """
    ret_loc = np.zeros((N, 3))
    for i in range(int(np.sqrt(N))):
        for j in range(int(np.sqrt(N))):
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.normal(loc=0, scale=np.sqrt(variance))
            x_shift = distance * np.cos(angle)
            y_shift = distance * np.sin(angle)
            # x_shift, y_shift = np.random.multivariate_normal([0, 0], [[np.sqrt(variance), 0], [0, np.sqrt(variance)]])
            # x_shift *= a
            # y_shift *= a
            ret_loc[i * int(np.sqrt(N)) + j, :] = np.array([a * (j + x_shift), b * (i + y_shift), z])
    return ret_loc


def get_G(a_v, N=1600, _lambda=0.7e-6, t=0):
    # a_s = np.linspace(0.18 * _lambda, _lambda, num=10)
    ret, specific_a = [], None
    k = 2 * np.pi / _lambda
    Gs = []
    for i, curr_a in enumerate(a_v):  # iterate over different lattice constants
        # if (i + 1) % 5 == 1:
        #     print(f"{i + 1} out of {len(a_v)}:")
        # print(f"a={format(curr_a, '.1E')}")
        a, b = curr_a, curr_a
        # print(curr_a * t)
        loc_arr = create_2d_arr(N, a, b, t)
        G = create_full_greens_matrix(k, loc_arr)
        Gs.append(G)
    return Gs


def create_full_greens_matrix(k, positions1, positions2=None):
    if positions2 is None:
        positions2 = positions1
    N1 = len(positions1)
    N2 = len(positions1)
    full_matrix = np.zeros((3 * N1, 3 * N2), dtype=complex)

    for n in range(N1):
        for m in range(N2):
            full_matrix[3 * n:3 * n + 3, 3 * m:3 * m + 3] = dyadic_greens_function(k, positions1[n], positions2[m])
    # print("real: ", np.real(full_matrix[full_matrix > 0] * 7e-7).min(), np.real(full_matrix[full_matrix > 0] * 7e-7).max())
    # print("imaginary: ", np.imag(full_matrix[full_matrix > 0] * 7e-7).min(), np.imag(full_matrix[full_matrix > 0] * 7e-7).max())

    return full_matrix


def compute_electric_field(G_matrix, E0, alpha, epsilon_0, lambda_):
    # prefactor = - (4 * np.pi ** 2 * alpha / (epsilon_0 * lambda_ ** 3)) * lambda_ * G_matrix
    prefactor = np.eye(G_matrix.shape[0]) - (4 * np.pi ** 2 * alpha / (epsilon_0 * lambda_ ** 3)) * lambda_ * G_matrix
    # prefactor = 1 - (4 * np.pi ** 2 * alpha / (epsilon_0 * lambda_ ** 3)) * lambda_ * G_matrix
    prefactor_inv = np.linalg.inv(prefactor)
    E = prefactor_inv @ E0
    return E


def compute_field_along_z(k, fixed_positions, E0, alpha, epsilon_0, lambda_, z_positions):
    fields_along_z = []
    N_fixed = len(fixed_positions)

    for z in z_positions:
        positions = np.vstack([fixed_positions, [0, 0, z]])
        E0_extended = np.concatenate([E0, [0, 0, 0]])  # Extend E0 to match the new size
        E = compute_electric_field(k, positions, E0_extended, alpha, epsilon_0, lambda_)
        fields_along_z.append(E[-3:])  # Extract field at the last position [0, 0, z]

    return np.array(fields_along_z)


# Define the parameters for the Gaussian beam
E0_peak = 1.0
w0 = 0.1  # Beam waist
# Example of computing the electric field vector E
alpha = 1.0
epsilon_0 = 8.854187817e-12
lambda_ = 0.5
k = 2 * np.pi / lambda_

grid_size = 4


def gaussian_beam(grid_size, w0, z, wavelength, dx, dy):
    """
    Generates a 2D Gaussian field at a specified z location.

    Parameters:
    grid_size (tuple): The size of the grid (nx, ny).
    w0 (float): Waist radius at z=0.
    x0, y0 (float): Beam center coordinates.
    z (float): Propagation distance.
    wavelength (float): Wavelength of the light.
    dx, dy (float): Grid spacings along the x and y axes.

    Returns:
    numpy.ndarray: 2D array representing the Gaussian field at location z.
    """
    nx, ny = grid_size, grid_size
    k = 2 * np.pi / wavelength
    zR = np.pi * w0 ** 2 / wavelength  # Rayleigh range

    if z == 0:
        w = w0
        R = np.inf
        psi = 0
    else:
        w = w0 * np.sqrt(1 + (z / zR) ** 2)
        R = z * (1 + (zR / z) ** 2)
        psi = np.arctan(z / zR)

    x = np.linspace(-nx / 2, nx / 2 - 1, nx) * dx
    y = np.linspace(-ny / 2, ny / 2 - 1, ny) * dy

    x, y = np.meshgrid(x, y)

    U = (w0 / w) * np.exp(-(x ** 2 + y ** 2) / w ** 2) * np.exp(
        1j * (k * z - k * (x ** 2 + y ** 2) / (2 * R) + psi))
    # normalize it to 1
    # U /= np.sum(U)
    # U /= np.max(U)
    return U


def get_E0(grid_size, w0, z, wavelength, dx, dy):
    # x = np.linspace(-1, 1, grid_size)
    # y = np.linspace(-1, 1, grid_size)
    # z = 0
    # fixed_positions = np.array([[xi, yi, z] for xi in x for yi in y])
    # N = len(fixed_positions)
    # # Initialize the Gaussian beam
    # E0 = np.zeros(3 * N, dtype=complex)
    # for i, (xi, yi, zi) in enumerate(fixed_positions):
    #     E0_amplitude = E0_peak * np.exp(-(xi ** 2 + yi ** 2) / w0 ** 2)
    #     E0[3 * i] = E0_amplitude  # E_x component
    #     E0[3 * i + 1] = 0.0  # E_y component
    #     E0[3 * i + 2] = 0.0  # E_z component
    """
        Generates a 3D electric field vector E0 based on a Gaussian beam.

        Parameters:
        grid_size (tuple): The size of the grid (nx, ny).
        w0 (float): Waist radius at z=0.
        z (float): Propagation distance.
        wavelength (float): Wavelength of the light.
        dx, dy (float): Grid spacings along the x and y axes.

        Returns:
        numpy.ndarray: 1D array representing the electric field vector E0.
        """
    nx, ny = int(grid_size), int(grid_size)
    N = nx * ny
    U = gaussian_beam(int(grid_size), w0, z, wavelength, dx, dy)

    # data = np.abs(U) ** 2
    # x, y = np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[1]))
    # plt.scatter(x.flatten(), y.flatten(), c=data.flatten(), cmap='viridis')
    # plt.colorbar()
    # plt.show()
    # print(f"Total sum input: {np.sum(np.abs(data))}")
    # Initialize the electric field vector E0
    E0 = np.zeros(3 * N, dtype=complex)

    # Fill the E_x components with the Gaussian field values
    E0[0::3] = U.flatten()
    E0[1::3] = 1j * U.flatten()

    return E0
