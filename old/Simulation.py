import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import epsilon_0
# from scipy.fft import fft2
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
from concurrent.futures import ThreadPoolExecutor
# from tqdm import tqdm

# from green_function import get_G
from E_calc import get_G, get_E0, compute_electric_field, create_full_greens_matrix, create_2d_arr


# import warnings
# warnings.simplefilter("always")

def plot_complex(field, title="", ax=None, show=False):
    if ax is None:
        ax = plt.gca()
    ax.imshow(np.abs(field) ** 2)
    ax.set_title(title)
    if show:
        plt.show()


def run(_lambda, N, alpha, resolution, a_v, w0, check_t=True, small_array=False):
    """
    runns a single simulation, for given distributions
    :param small_array: boolean, if true, will run a smaller array by down sampling the resolution
    :param check_t: boolean, if true, will check for different temperatures
    :param _lambda: the field wavelength
    :param N: the size of the array
    :param alpha: coefficient of the scattered wave equation
    :param resolution: the resolution of the simulation, which is the size of the array
    :param a_v: the lattice constant
    :param w0: the waist width of the field
    :param beam: a gaussian field object
    :param array_loc: the distance where the meta-surface is
    """
    # from tqdm import tqdm

    temperatures = np.linspace(0, 0.05, 5) if check_t else [0]  # TODO should be num:4
    # temperatures = np.linspace(0, 0.4, 33) if check_t else [0]
    # temperatures = np.linspace(0, 0.05, 4) if check_t else [0]
    reflectivity = np.zeros(len(temperatures), dtype=complex)
    mean_of_g = np.zeros(len(temperatures), dtype=complex)
    im_mean_of_g = np.zeros(len(temperatures), dtype=complex)
    reflectivity_var = np.zeros(len(temperatures))
    angles_temp = np.zeros(len(temperatures))
    angles_temp_var = np.zeros(len(temperatures))
    sum_angles_temp = np.zeros(len(temperatures))
    sum_angles_temp_std = np.zeros(len(temperatures))
    dom_angles = np.zeros(len(temperatures))
    dom_angles_std = np.zeros(len(temperatures))
    reps = 50  # TODO should be 100
    # reps = 2
    max_reflectivity = np.zeros(len(temperatures))
    for j, t in enumerate(temperatures):
        repetitions = np.zeros(reps, dtype=complex)
        repetitions_max = np.zeros(reps)
        angles = np.zeros(reps)
        sum_angles = np.zeros(reps)
        dominant_angles = np.zeros(reps)
        temp_std = np.zeros(reps)
        temp_std_norm = np.zeros(reps)
        temp_mean = np.zeros(reps)
        # for k in tqdm(range(reps), desc="Processing", unit=" repetition"):
        for k in range(reps):
            print(f"t={t}, k={k}")
            if t == 0 and k > 0:
                repetitions = np.array([repetitions[0] for _ in range(reps)])
                repetitions_max = np.array([repetitions_max[0] for _ in range(reps)])
                print("skips repetitions for t=0")
                break
            Gs = get_G(a_v=a_v, _lambda=_lambda, N=N, t=t)
            # for i, g in enumerate(Gs):
            #     Gs[i] = increase_resolution(g, resolution / N)
            #     Gs[i] *= 1100
            G = Gs[0]
            # print(t, np.mean(G))
            # print("Done.")
            # counter = 1
            tot = np.zeros(len(Gs))
            # G /= np.sum(G)
            # print("eigenvalues: ", np.linalg.eig(G))
            # G /= 15
            res = scatter_calc(G, N, resolution, _lambda, a_v, alpha, t, tot, w0, mean_of_g, im_mean_of_g, j, k)
            # repetitions[k] = res[0]
            # repetitions_max[k] = res[1]
            # sum_angles[k] = res[3]
            # dominant_angles[k] = res[4]
            repetitions_max[k] = res[0]
            angles[k] = res[1]
            temp_mean[k] = np.mean(repetitions_max[:k + 1])
            temp_std_norm[k] = np.std(repetitions_max[:k + 1]) / np.sqrt(k + 1)
            temp_std[k] = np.std(repetitions_max[:k + 1])
            # plot the heatmap of res[5]
            # plt.imshow(res[5])
            # plt.colorbar()
            # plt.title(f"heatmap of t={t}, k={k}")
            # plt.savefig(f"./angles/{t}_{k}.png")
            # print(f"{k + 1}/{reps} rep is done")
        np.save(f"./repetitions N={N},t={t}.npy", repetitions_max)
        reflectivity[j] = np.mean(repetitions_max)
        # save also std
        reflectivity_var[j] = np.std(repetitions_max) / np.sqrt(reps)
        max_reflectivity[j] = np.mean(repetitions_max)
        angles_temp[j] = np.mean(angles)
        angles_temp_var[j] = np.std(angles) / np.sqrt(reps)
        # sum_angles_temp[j] = np.mean(sum_angles)
        # sum_angles_temp_std[j] = np.mean(sum_angles) / np.sqrt(reps)
        # dom_angles[j] = np.mean(dominant_angles)
        # dom_angles_std[j] = np.mean(dominant_angles) / np.sqrt(reps)
        print(f"t={t} is done")
        # # check convergence repetitions as a function of repetitions
        # plt.scatter(range(reps), temp_mean, label="mean")
        # plt.savefig(f"./converge/mean of repetitions for t={t} (max).png")
        # plt.close()
        # plt.scatter(range(reps), temp_std, label="std")
        # plt.savefig(f"./converge/std of repetitions for t={t} (max).png")
        # plt.close()
        # plt.scatter(range(reps), temp_std_norm, label="std norm")
        # plt.savefig(f"./converge/std norm of repetitions for t={t} (max).png")
        # plt.close()
        # plt.scatter(range(reps), repetitions_max, label="repetitions")
        # plt.savefig(f"./converge/raw data of r for t={t} (max).png")
        # plt.close()

    # save data
    np.save(f"./reflectivity N={N},a={a_v[0]}.npy", reflectivity)
    np.save(f"./reflectivity_var N={N},a={a_v[0]}.npy", reflectivity_var)
    print(max_reflectivity.mean(), max_reflectivity.std())  # todo: take first peak

    # plt.scatter(temperatures, reflectivity, label="a=0.2")
    # plt.scatter(temperatures, max_reflectivity, label="(max) reflectivity")
    # plt.scatter(temperatures, mean_of_g, label="real mean"c)
    # plt.scatter(temperatures, im_mean_of_g, label="imginary mean")
    # divide max_reflectivity by its first element
    max_reflectivity /= max_reflectivity[0]
    plt.errorbar(temperatures, max_reflectivity, yerr=reflectivity_var, label="a=0.2", fmt='o', markersize=5)
    plt.scatter(temperatures, max_reflectivity)
    plt.legend()
    plt.title(f"Reflectivity vs. shift from ideal location, a={a_v[0]}")
    plt.xlabel("Standard Deviation of shift in each direction [a]")
    plt.ylabel("Reflectivity")
    plt.ylim((max_reflectivity.min() - 0.1, 1.1))
    # set ticks to be floor(10*max_reflectivity.min())/10 to 1 in 0.1 jumps
    plt.yticks(np.arange(np.floor(10 * max_reflectivity.min()) / 10, 1.1, 0.1))
    plt.savefig(fr"./reflectivity vs tmp N={N},a={a_v[0] / _lambda}.png")
    # plt.show()
    plt.close()

    # plot_phase(N, a_v, (angles_temp - sum_angles_temp) / np.pi, angles_temp_var, temperatures,
    #            "center phase normalized")
    # plot_phase(N, a_v, sum_angles_temp, sum_angles_temp_std, temperatures, "sum of phase")
    plot_phase(N, a_v, angles_temp, angles_temp_var, temperatures, "center phase")
    # # plot_phase(N, a_v, dom_angles, dom_angles_std, temperatures, "dominant phase")

    # save all lists
    np.save(f"./reflectivity N={N},a={a_v[0]}.npy", reflectivity)
    np.save(f"./reflectivity_var N={N},a={a_v[0]}.npy", reflectivity_var)
    np.save(f"./angles_temp N={N},a={a_v[0]}.npy", angles_temp)
    np.save(f"./angles_temp_var N={N},a={a_v[0]}.npy", angles_temp_var)
    # np.save(f"./sum_angles_temp N={N},a={a_v[0]}.npy", sum_angles_temp)
    # np.save(f"./sum_angles_temp_std N={N},a={a_v[0]}.npy", sum_angles_temp_std)
    # np.save(f"./dom_angles N={N},a={a_v[0]}.npy", dom_angles)
    # np.save(f"./dom_angles_std N={N},a={a_v[0]}.npy", dom_angles_std)
    np.save(f"./max_reflectivity N={N},a={a_v[0]}.npy", max_reflectivity)
    # np.save(f"./mean_of_g N={N},a={a_v[0]}.npy", mean_of_g)


def plot_phase(N, a_v, angles_temp, angles_temp_var, temperatures, title):
    plt.errorbar(temperatures, angles_temp, yerr=angles_temp_var, fmt='o', markersize=5)
    plt.scatter(temperatures, angles_temp)
    plt.title(f"{title} Angle of reflected field vs. shift from ideal location, a={a_v[0]}")
    plt.xlabel("Standard Deviation of shift in each direction [a]")
    plt.ylabel("Phase [Pi]")
    if title == "center phase normalized":
        plt.ylabel(r"Angle [$\pi \cdot rad$]")
    plt.savefig(fr"./{title} angle vs tmp N={N},a={a_v[0]}.png")
    # plt.show()
    plt.close()


def propagate_field(field, wavelength, distance, dx, dy):
    """
    Propagates the EM field over a given distance in free space.

    Parameters:
    field (numpy.ndarray): 2D array representing the initial EM field distribution
    dx, dy (float): grid spacings along the x and y axes (assume equal spacing along both axes)
    distance (float): distance to propagate the field (in the same units as wavelength)
    wavelength (float): wavelength of the light (in the same units as distance)

    Returns:
    numpy.ndarray: 2D array representing the propagated EM field
    """

    nx, ny = field.shape
    fx = np.fft.fftfreq(nx, dx)
    fy = np.fft.fftfreq(ny, dy)
    fx, fy = np.meshgrid(fx, fy)

    k = 2 * np.pi / wavelength
    # fx2 = (wavelength * fx) ** 2
    # fy2 = (wavelength * fy) ** 2
    # H = np.exp(1j * k * distance * np.sqrt(1 - fx2 - fy2))
    # H[fx2 + fy2 >= 1] = 0
    mask = (wavelength * fx) ** 2 + (wavelength * fy) ** 2 < 1
    H = np.zeros_like(field, dtype=complex)
    H[mask] = np.exp(1j * k * distance * np.sqrt(1 - (wavelength * fx[mask]) ** 2 - (wavelength * fy[mask]) ** 2))

    field_FT = np.fft.fft2(field)
    # field_FT = np.fft.fftshift(np.fft.fft2(field))
    field_FT_propagated = field_FT * H
    # field_propagated = np.fft.ifft2(np.fft.ifftshift(field_FT_propagated, axes=(0, 1)))
    field_propagated = np.fft.ifft2(field_FT_propagated)

    # field_FT = np.fft.fft2(field)
    # field_FT = np.fft.fftshift(field_FT)  # Apply fftshift
    # field_FT_propagated = field_FT * H
    # field_FT_propagated = np.fft.ifftshift(field_FT_propagated)  # Apply ifftshift
    # field_propagated = np.fft.ifft2(field_FT_propagated)

    return field_propagated


# def propagate_EM_wave(E0, k, z, dx, dy):
#     """
#     Propagates the EM field E0 along the z-axis.
#
#     Parameters:
#     E0 (numpy.ndarray): initial 2D array which is EM field at z=0
#     k (float): wave number in z direction
#     z (float): distance where the light propagates to
#     dx (float): size of the pixel/element in x direction
#     dy (float): size of the pixel/element in y direction
#
#     Returns:
#     numpy.ndarray: Propagated EM field at z
#     """
#     ny, nx = E0.shape
#
#     # Spatial frequencies
#     fx = np.fft.fftfreq(nx, d=dx)
#     fy = np.fft.fftfreq(ny, d=dy)
#     FX, FY = np.meshgrid(fx, fy)
#
#     # Fourier transform the initial field
#     E0_fft = np.fft.fftshift(np.fft.fft2(E0))
#
#     # Propagation kernel
#     kz = np.sqrt(k ** 2 - (2 * np.pi * FX) ** 2 - (2 * np.pi * FY) ** 2)
#     H = np.exp(-1j * kz * z)
#
#     # Propagate in frequency domain and then inverse Fourier transform
#     E_z = np.fft.ifft2(np.fft.ifftshift(E0_fft * H))
#
#     return E_z


# def propagate_light(e_r, k, real_axis, z):
#     # Calculate the spatial step along the real axis
#     delta_x = real_axis[1] - real_axis[0]
#
#     # Perform 2D FFT on the input field
#     e_r_fft = fftpack.fft2(e_r)
#
#     # Get the shape of the input field
#     num_points_x, num_points_y = e_r.shape
#
#     # Create the frequency grid
#     freq_x = fftpack.fftfreq(num_points_x, delta_x)
#     freq_y = fftpack.fftfreq(num_points_y, delta_x)
#     fx, fy = np.meshgrid(freq_x, freq_y)
#
#     # Calculate the propagation factor
#     prop_factor = np.exp(1j * k * z * (fx ** 2 + fy ** 2))
#
#     # Perform inverse 2D FFT on the propagated field
#     e_propagated = fftpack.ifft2(e_r_fft * prop_factor)
#
#     # # normalize
#     # e_propagated /= np.sum(e_propagated)
#
#     return e_propagated


def create_gifs():
    import imageio
    import os
    images = []
    dir_name = "results/new/"
    # Get list of all files only in the given directory
    list_of_files = filter(lambda x: os.path.isfile(os.path.join(dir_name, x)), os.listdir(dir_name))
    # Sort list of files based on last modification time in ascending order
    list_of_files = sorted(list_of_files, key=lambda x: os.path.getmtime(os.path.join(dir_name, x)),
                           reverse=False)
    # Iterate over sorted list of files and print file path along with last modification time of file
    for filename in list_of_files:
        images.append(imageio.imread(f"{dir_name}{filename}"))
    imageio.mimsave(f'{dir_name}/movie.gif', images, duration=0.05)


def print_frequency(x, y):
    """prints the frequency of y, using the highest frequency of fft"""
    from scipy.fft import fft, fftfreq
    # Number of sample points
    N = len(y)
    # sample spacing
    T = x[1] - x[0]
    yf = fft(y)
    xf = fftfreq(N, T)[:N // 2]
    plt.plot(xf, 2.0 / N * np.abs(yf[0:N // 2]))
    plt.xlim(0, 0.5e7)
    # plt.show()
    plt.close()
    print(f"frequency is {xf[np.argmax(np.abs(yf[0:N // 2]))]}")


def check_symmetry(G, G_scat, E_scat, E_0, norm=False):
    if norm:
        # Normalize matrices
        G = G / np.linalg.norm(G)
        G_scat = G_scat / np.linalg.norm(G_scat)
        E_scat = E_scat / np.linalg.norm(E_scat)
        E_0 = E_0 / np.linalg.norm(E_0)

    print("G is symmetric") if np.allclose(G, G.T) else print("G is not symmetric")
    print("G_scat is symmetric") if np.allclose(G_scat, G_scat.T) else print("G_scat is not symmetric")
    print("G_inv is symmetric") if np.allclose(np.linalg.inv(G_scat), np.linalg.inv(G_scat).T) else print(
        "G_inv is not symmetric")
    print("E_0 is symmetric") if np.allclose(E_0, E_0.T) else print("E_0 is not symmetric")
    print("E_scat is symmetric") if np.allclose(E_scat, E_scat.T) else print("E_scat is not symmetric")
    check = np.matmul(np.linalg.inv(G_scat), E_0)
    print("matmul is symmetric") if np.allclose(check, check.T) else print("matmul is not symmetric")
    check = np.dot(np.linalg.inv(G_scat), E_0)
    print("dot is symmetric") if np.allclose(check, check.T) else print("dot is not symmetric")
    # check if np.linalg.inv(G_scat) commutes with E_0
    print("commutes") if np.allclose(np.linalg.inv(G_scat) @ E_0, E_0 @ np.linalg.inv(G_scat)) else print(
        "does not commute")
    # check if np.linalg.inv(G_scat) commutes with E_scat
    print("commutes") if np.allclose(np.linalg.inv(G_scat) @ E_scat, E_scat @ np.linalg.inv(G_scat)) else print(
        "does not commute")


def compute_E_field(E0, G, E, alpha, lambda_val, epsilon0):
    """
    Computes the electric field E according to the given equation.

    Parameters:
    E0 : ndarray; The initial electric field vector, shape (3N,)
    G : ndarray; The 3Nx3N Green's function matrix, shape (3N, 3N)
    alpha : float; Polarizability
    lambda_val : float; Wavelength
    epsilon0 : float; Permittivity of free space

    Returns:
    E : ndarray; The computed electric field vector, shape (3N,)
    """
    E_ret = np.zeros_like(E0)  # Initialize the electric field vector
    # Constant factor
    prefactor = 4 * (np.pi ** 2) * alpha / (epsilon0 * (lambda_val ** 2))

    # Compute the summation term
    for i in range(3):  # Loop over direction (X,Y,Z)
        for j in range(3):  # Loop over direction (X,Y,Z)
            # Take every jth element every ith row
            E_ret[i::3] += G[i::3, j::3] @ E[j::3]

    return E0 + prefactor * E_ret


def scatter_calc(G, N, resolution, _lambda, a_v, alpha, t, tot, w0, mean_of_g, im_mean_of_g, k, i=""):
    length = 3
    factor = (length * 100) + 1
    a_v = a_v[0]
    res = (G.shape[0], G.shape[1])
    E_0 = get_E0(np.sqrt(N), w0, 0, _lambda, dx=a_v / (resolution / N), dy=a_v / (resolution / N))
    sqrtN = int(np.sqrt(N))
    # orig_center = (sqrtN // 2) * sqrtN + (sqrtN // 2)
    # normalization_index = 3 * orig_center
    # normalization_factor = E_0[normalization_index]
    # E_0 /= normalization_factor

    # actual_size = a_v * N  # np.sqrt(N)  # size in m?
    # x = np.linspace(-actual_size / 2, actual_size / 2, num=res[0])
    # y = np.linspace(-actual_size / 2, actual_size / 2, num=res[1])

    E_scat_all = compute_electric_field(G, E_0, alpha, epsilon_0, _lambda)
    # plot_imfield(E_scat_all, sqrtN, r'$\left|E\right|^{2}$ on a ', -1, counter="all")
    # E_scat = E_scat_all[::3].reshape((sqrtN, sqrtN))

    # mean_of_g[k] = np.real(np.mean(E_scat))
    # im_mean_of_g[k] = np.imag(np.mean(E_scat))

    r = np.zeros(factor, dtype=np.complex128)
    zs = np.linspace(-length * _lambda, length * _lambda, factor)
    loc = zs / _lambda
    loc_0 = create_2d_arr(N, a_v, a_v, 0, 0)

    # Pre-compute constants
    dx_dy = a_v / (resolution / N)
    k_ = 2 * np.pi / _lambda

    # # Function to process a single iteration
    # def process_z(z):
    #     # print(f"z={z}")
    #     E_in = get_E0(sqrtN, w0, z, _lambda, dx=dx_dy, dy=dx_dy)
    #     E_in /= normalization_factor
    #     loc_z = create_2d_arr(N, a_v, a_v, 0, z)
    #     G_j = create_full_greens_matrix(k_, loc_0, loc_z)
    #     E_z = compute_E_field(E_in, G_j, E_scat_all, alpha, _lambda, epsilon_0)
    #
    #     # Subsample and reshape
    #     E_z = E_z[::3].reshape((sqrtN, sqrtN))
    #     E_in = E_in[::3].reshape((sqrtN, sqrtN))
    #
    #     # Compute means
    #     return (
    #         np.real(E_z[sqrtN//2, sqrtN//2]),
    #         np.imag(E_z[sqrtN//2, sqrtN//2]),
    #         np.real(E_in[sqrtN//2, sqrtN//2]),
    #         np.imag(E_in[sqrtN//2, sqrtN//2])
    #         # np.mean(np.real(E_z)),
    #         # np.mean(np.imag(E_z)),
    #         # np.mean(np.real(E_in)),
    #         # np.mean(np.imag(E_in))
    #     )
    #
    # # Run the loop in parallel
    # with ThreadPoolExecutor() as executor:
    #     results = list(executor.map(process_z, zs))
    #
    # # Unpack results
    # real_reflected, im_reflected, real_in, im_in = zip(*results)
    #
    # # Convert results back to numpy arrays
    # real_reflected = np.array(real_reflected)
    # im_reflected = np.array(im_reflected)
    # real_in = np.array(real_in)
    # im_in = np.array(im_in)

    real_reflected = np.zeros(factor)
    im_reflected = np.zeros(factor)
    real_in = np.zeros(factor)
    im_in = np.zeros(factor)
    for j, z in enumerate(zs):
        E_in = get_E0(np.sqrt(N), w0, z, _lambda, dx=a_v / (resolution / N), dy=a_v / (resolution / N))
        # E_in /= normalization_factor
        loc_z = create_2d_arr(N, a_v, a_v, 0, z)
        G_j = create_full_greens_matrix(2 * np.pi / _lambda, loc_0, loc_z)
        E_z = compute_E_field(E_in, G_j, E_scat_all, alpha, _lambda, epsilon_0)
        # plot_imfield(E_z, sqrtN, rf'z={z} |E|^2 on a ', -1, counter="all")
        E_z = E_z[::3].reshape((sqrtN, sqrtN))
        E_in = E_in[::3].reshape((sqrtN, sqrtN))
        # real_reflected[j] = np.real(E_z[sqrtN // 2, sqrtN // 2])
        # im_reflected[j] = np.imag(E_z[sqrtN//2, sqrtN//2])
        # real_in[j] = np.real(E_in[sqrtN//2, sqrtN//2])
        # im_in[j] = np.imag(E_in[sqrtN//2, sqrtN//2])
        real_reflected[j] = np.mean(np.real(E_z))
        im_reflected[j] = np.mean(np.imag(E_z))
        real_in[j] = np.mean(np.real(E_in))
        im_in[j] = np.mean(np.imag(E_in))
        # if z == 0:
        #     # get the angle of E_z
        #     dominant_phase = np.angle(E_z.flatten()[np.argmax(np.abs(E_z))])
        #
        #     # see angle at [0,0] or small area and change to E_z/E_0
        #     # phase = np.angle(E_z[0, 0])
        #     if E_z.shape[0] % 2 == 1:
        #         # phase = np.angle(E_z[E_z.shape[0] // 2, E_z.shape[0] // 2])
        #         # field_angle = np.angle(E_in[E_in.shape[0] // 2, E_in.shape[0] // 2])
        #         center = E_z.shape[0] // 2
        #         phase = np.mean(np.angle(E_z[center - 1:center + 2, center - 1:center + 2]))
        #         field_angle = np.mean(np.angle(E_in[center - 1:center + 2, center - 1:center + 2]))
        #     else:
        #         center = E_z.shape[0] // 2
        #         phase = np.mean(np.angle(E_z[center:center + 2, center:center + 2]))
        #         field_angle = np.mean(np.angle(E_in[center:center + 2, center:center + 2]))
        #     all_angles = np.angle(E_z)
        # r[j] = 0
    # create_gifs()
    normalization_factor = (im_in.max()) * 2
    real_in /= normalization_factor
    im_in /= normalization_factor
    real_reflected /= normalization_factor
    im_reflected /= normalization_factor
    # get max index s.t. zs / _lambda is smaller than -0.5
    max_index = np.argmin(loc < -0.5)
    # look for highest absolute value of reflectivity up to this index
    intensity = real_reflected ** 2 + im_reflected ** 2
    max_reflectivity = np.max(intensity[:max_index])
    max_reflectivity_index = np.argmax(intensity[:max_index])
    phase = np.angle(real_reflected[max_reflectivity_index] + 1j * im_reflected[max_reflectivity_index])
    # max_reflectivity = np.max(np.abs(r[:max_index]))

    # get the 4 first colors of default color map
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:4]

    # plot real and imaginary parts of real_in, real_reflected, im_in, im_reflected
    plt.plot(loc, real_in, label="Re[input]", color=colors[3])
    plt.plot(loc, im_in, label="Im[input]", color=colors[1])
    plt.plot(loc, im_reflected, label="Im[field]", linestyle='dashed', color=colors[2])
    plt.plot(loc, real_reflected, label="Re[field]", linestyle='dashed', color=colors[0])
    plt.legend()
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.title(rf'mean(field) vs. z for $a/\lambda={round(a_v / _lambda, 2)}$, s.d.={t}a')
    plt.xlabel(r"$z/\lambda$")
    plt.ylabel(r"mean of field")
    plt.ylim(-1, 1)
    plt.savefig(f"results/t={t} vs z for a={round(a_v / _lambda, 2)}{'_' + str(i) if i else ''}.png")
    # plt.show()
    plt.close()

    return max_reflectivity, phase / np.pi


def plot_imfield(E_scat_all, sqrtN, title, start=0, counter=0):
    # Plot the real part of E_x as a function of grid points
    # field = np.abs(E_scat_all[::3].reshape((sqrtN, sqrtN))) ** 2 + np.abs(
    #     E_scat_all[1::3].reshape((sqrtN, sqrtN))) ** 2 + np.abs(E_scat_all[2::3].reshape((sqrtN, sqrtN))) ** 2
    if start == -1:
        field = np.abs(E_scat_all[::3].reshape((sqrtN, sqrtN))) ** 2 + np.abs(
            E_scat_all[1::3].reshape((sqrtN, sqrtN))) ** 2 + np.abs(
            E_scat_all[2::3].reshape((sqrtN, sqrtN))) ** 2
    else:
        field = np.abs(E_scat_all[start::3].reshape((sqrtN, sqrtN))) ** 2
    # plt.matshow(field, extent=(-1, 1, -1, 1), origin='lower', cmap='viridis', interpolation='none')
    x, y = np.meshgrid(np.arange(field.shape[0]), np.arange(field.shape[1]))
    plt.scatter(x.flatten(), y.flatten(), c=field.flatten(), cmap='viridis')
    plt.colorbar()
    # plt.colorbar(label='Re(E_x)')
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.rcParams['text.usetex'] = True
    plt.title(title + f'{sqrtN} x {sqrtN} grid')
    name_ = f"E_{counter}"
    plt.savefig(f"./{name_}_{sqrtN}x{sqrtN}_grid.png")
    plt.show()
    plt.close()
    print(f"Total sum {title}: {np.sum(np.abs(field)**2)}")


def plot_fields(E_0, E_scat, G, G_scat, N, _lambda, a_v, t):
    fig, axes = plt.subplots(2, 3, figsize=(10, 8))
    plot_complex(np.abs(G), "G", ax=axes[0, 0])
    plot_complex(np.abs(G_scat), "1-aG", ax=axes[0, 1])
    plot_complex(np.abs(np.linalg.inv(G_scat)), r"$(1-aG)^{-1}$", ax=axes[0, 2])
    plot_complex(np.abs(E_0), "input", ax=axes[1, 0])
    plot_complex(np.abs(E_scat), f"E scattered", ax=axes[1, 1])
    plot_complex(np.abs(E_0 / np.sum(E_0) - E_scat / np.sum(E_scat)), f"input - E scattered\n(difference)",
                 ax=axes[1, 2])
    # # make axes[1,2] empty
    # axes[1, 2].axis('off')
    fig.suptitle(fr"matrices for t={t:.2f}, N={N}, a={round(a_v / _lambda, 1)}$\lambda$")
    plt.tight_layout()
    plt.show()
    # print(f"sum: {np.sum(np.linalg.inv(G_scat))} for t={t:.2f}")


def increase_resolution(field, factor):
    # increase field size: add 16 empty pixels after each pixel
    field = np.repeat(np.repeat(field, factor, axis=0), factor, axis=1)
    # # pad all borders with 4 pixels
    # field = np.pad(field, [(4, 4), (4, 4)], mode='constant')
    # use gaussian kernel to blur the matrix
    real = gaussian_filter(np.real(field), sigma=20)
    im = gaussian_filter(np.imag(field), sigma=20)
    field = real + 1j * im
    return field


if __name__ == '__main__':
    _lambda = 0.7e-6
    # alpha = -3j / (4 * np.pi ** 2)
    alpha = 1j * (3 / (4 * np.pi ** 2)) * epsilon_0 * (_lambda ** 3)  # delta=0, lambda_a=lambda, gamma_0=gamma

    # # from Rivka's code
    # gamma = 24e6
    # alpha = -3 / (4 * np.pi ** 2) * (gamma / 2) / (1j * gamma / 2)  # no units
    # # alpha *= 50

    # a_v = np.linspace(0.1 * _lambda, 0.4 * _lambda, num=4)
    a_v = np.array([0.2 * _lambda])
    # a_v = np.array([0.20645 * _lambda])
    # w0 = 1.5 * _lambda
    # array_loc = 1e-25 + 0.1653000201 + _lambda / 2  # todo: check frequency of all beams
    num = 20
    for N in [num ** 2]:
        # for N in [100]:
        print(f"###    N={N}    ###")
        w0 = np.sqrt(N) * 0.3 * a_v[0]  # todo: w0 scale?
        # field = gaussian_source(w0=w0, lam=_lambda, z=0)
        # resolution = N*16

        # beam = increase_resolution(beam, 1024 / N)
        # run(_lambda, N, alpha, N * 16, a_v, w0, beam, check_t=True, small_array=False)
        run(_lambda, N, alpha, N, a_v, w0, check_t=True, small_array=False)
