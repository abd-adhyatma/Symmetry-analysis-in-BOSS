import numpy as np
import pandas as pd
import math
from ase import Atoms
from ase.calculators.emt import EMT
from .create import *

# Utility functions to handle angle and unit conversions
def get_spglib_cell(atoms:Atoms):
    """
    Takes in ASE Atoms object, returns lattice vectors, atomic points, and atomic numbers.
    The outputs are used as input for spglib's get_symmetry function.
    """
    lattice = atoms.cell[:]
    points = atoms.get_scaled_positions()
    numbers = atoms.get_atomic_numbers()

    cell = (lattice, points, numbers)
    return cell

def euler_to_rotation(alpha: float, beta: float, gamma: float, return_Rs: bool=False):
    """
    The function to place molecules on slab typically applies external rotations in the order of X, Y, and Z axes.
    This function converts Euler angles (alpha, beta, gamma, in degrees) in the XYZ convention to a rotation matrix.
    
    Parameters:
    alpha (float): Rollangle (rotation around x-axis) in radians.
    beta (float): Pitch angle (rotation around y-axis) in radians.
    gamma (float): Yaw angle (rotation around z-axis) in radians.
    
    Returns:
    np.ndarray: A 3x3 rotation matrix corresponding to the Euler angles.
    """
    alpha = np.radians(alpha)
    beta = np.radians(beta)
    gamma = np.radians(gamma)

    # Rotation matrix around x-axis (Roll)
    R_x = np.array([
    [1, 0, 0],
    [0, np.cos(alpha), -np.sin(alpha)],
    [0, np.sin(alpha), np.cos(alpha)]
    ])
    
    # Rotation matrix around y-axis (Pitch)
    R_y = np.array([
    [np.cos(beta), 0, np.sin(beta)],
    [0, 1, 0],
    [-np.sin(beta), 0, np.cos(beta)]
    ])
    
    # Rotation matrix around z-axis (Yaw)
    R_z = np.array([
    [np.cos(gamma), -np.sin(gamma), 0],
    [np.sin(gamma), np.cos(gamma), 0],
    [0, 0, 1]
    ])
    
    # Apply dot product R = Rz . Ry . Rx
    R_c = np.dot(R_z, np.dot(R_y, R_x))

    if return_Rs:
        return R_x, R_y, R_z, R_c
    
    else:
        return R_c

def rotation_to_euler(R: np.ndarray):
    """
    Convert a rotation matrix R to Euler angles (gamma, beta, alpha) based on the given pseudocode.
    
    Parameters:
    R (np.ndarray): 3x3 rotation matrix.

    Returns:
    list: A list containing two possible sets of Euler angles (gamma1, beta1, alpha1) and (gamma2, beta2, alpha2) in degrees.
    """

    def to_positive_angle(angle):
        """Ensure the angle is positive."""
        return angle if angle >= 0 else angle + 360

    # Check if R31 is not ±1
    if not np.isclose(R[2, 0], 1.0) and not np.isclose(R[2, 0], -1.0):
        # First possible solution
        beta1 = -np.arcsin(R[2, 0])
        beta2 = np.pi - beta1  # Second possible solution

        # Compute alpha1 and alpha2
        alpha1 = np.arctan2(R[2, 1] / np.cos(beta1), R[2, 2] / np.cos(beta1))
        alpha2 = np.arctan2(R[2, 1] / np.cos(beta2), R[2, 2] / np.cos(beta2))

        # Compute gamma1 and gamma2
        gamma1 = np.arctan2(R[1, 0] / np.cos(beta1), R[0, 0] / np.cos(beta1))
        gamma2 = np.arctan2(R[1, 0] / np.cos(beta2), R[0, 0] / np.cos(beta2))

        # Convert to degrees and ensure positive angles
        return [
            (
                to_positive_angle(np.degrees(alpha1)),
                to_positive_angle(np.degrees(beta1)),
                to_positive_angle(np.degrees(gamma1))
            ),
            (
                to_positive_angle(np.degrees(alpha2)),
                to_positive_angle(np.degrees(beta2)),
                to_positive_angle(np.degrees(gamma2))
            )
        ]
    else:
        # Handle special case where R31 = ±1 (Gimbal lock)
        gamma = 0  # Arbitrary since it can be anything, set to 0

        if np.isclose(R[2, 0], -1.0):
            beta = np.pi / 2 
            alpha = np.arctan2(R[0, 1], R[0, 2])
        else:  # Case when R[2, 0] == 1.0
            beta = -np.pi / 2
            alpha = np.arctan2(-R[0, 1], -R[0, 2])

        # Convert to degrees and ensure positive angles
        return [
            (
                to_positive_angle(np.degrees(alpha)),
                to_positive_angle(np.degrees(beta)),
                to_positive_angle(np.degrees(gamma))
            )
        ]

def rotation_to_euler_rounded(R: np.ndarray):
    """
    Convert a rotation matrix R to Euler angles (gamma, beta, alpha) based on the given pseudocode.
    
    Parameters:
    R (np.ndarray): 3x3 rotation matrix.

    Returns:
    list: A list containing two possible sets of Euler angles (gamma1, beta1, alpha1) and (gamma2, beta2, alpha2) in degrees.
    """

    def to_positive_angle(angle):
        """Ensure the angle is positive."""
        return angle if angle >= 0 else angle + 360

    def round_angle(angle, decimals=3):
        """Round the angle to a specific number of decimal places."""
        return round(angle, decimals)

    # Check if R31 is not ±1
    if not np.isclose(R[2, 0], 1.0) and not np.isclose(R[2, 0], -1.0):
        # First possible solution
        beta1 = -np.arcsin(R[2, 0])
        beta2 = np.pi - beta1  # Second possible solution

        # Compute alpha1 and alpha2
        alpha1 = np.arctan2(R[2, 1] / np.cos(beta1), R[2, 2] / np.cos(beta1))
        alpha2 = np.arctan2(R[2, 1] / np.cos(beta2), R[2, 2] / np.cos(beta2))

        # Compute gamma1 and gamma2
        gamma1 = np.arctan2(R[1, 0] / np.cos(beta1), R[0, 0] / np.cos(beta1))
        gamma2 = np.arctan2(R[1, 0] / np.cos(beta2), R[0, 0] / np.cos(beta2))

        # Convert to degrees and ensure positive angles, then round to 4 decimal points
        return [
            (
                round_angle(to_positive_angle(np.degrees(alpha1))),
                round_angle(to_positive_angle(np.degrees(beta1))),
                round_angle(to_positive_angle(np.degrees(gamma1)))
            ),
            (
                round_angle(to_positive_angle(np.degrees(alpha2))),
                round_angle(to_positive_angle(np.degrees(beta2))),
                round_angle(to_positive_angle(np.degrees(gamma2)))
            )
        ]
    else:
        # Handle special case where R31 = ±1 (Gimbal lock)
        gamma = 0  # Arbitrary since it can be anything, set to 0

        if np.isclose(R[2, 0], -1.0):
            beta = np.pi / 2 
            alpha = np.arctan2(R[0, 1], R[0, 2])
        else:  # Case when R[2, 0] == 1.0
            beta = -np.pi / 2
            alpha = np.arctan2(-R[0, 1], -R[0, 2])

        # Convert to degrees and ensure positive angles, then round to 4 decimal points
        return [
            (
                round_angle(to_positive_angle(np.degrees(alpha))),
                round_angle(to_positive_angle(np.degrees(beta))),
                round_angle(to_positive_angle(np.degrees(gamma)))
            )
        ]

def boss_to_symm(boss_pos: list, slab: Atoms, nx:int=1, ny:int=1):
    """
    Convert positions in x, y, z used as BOSS input to values suitable in symmetry algorithm.
    """
    # l_z = slab.get_cell()[2][2]
    l_z = slab.cell.cellpar()[2]
    z_top = slab.positions[:, 2].max()

    x_symm = boss_pos[0] / nx
    y_symm = boss_pos[1] / ny
    z_symm = (boss_pos[2]+z_top) / l_z

    return [x_symm, y_symm, z_symm]

def symm_to_boss(symm_pos: list, slab: Atoms, nx:int=1, ny:int=1):
    """
    Convert positions in x, y, z results from the symmetry algorithm to values usable as BOSS input.
    """
    l_z = slab.cell.cellpar()[2]
    z_top = slab.positions[:, 2].max()

    x_boss = (symm_pos[0] * nx) % 1.0
    y_boss = (symm_pos[1] * ny) % 1.0
    z_boss = (symm_pos[2] * l_z) - z_top

    return [x_boss, y_boss, z_boss]

def symm_to_boss_rounded(symm_pos: list, slab: Atoms, nx:int=1, ny:int=1):
    """
    Convert positions in x, y, z results from the symmetry algorithm to values usable as BOSS input.
    """
    l_z = slab.cell.cellpar()[2]
    z_top = slab.positions[:, 2].max()

    def round_pos(pos, decimals=3):
        """
        Round the angle to a specific number of decimal places.
        """
        return round(pos, decimals)

    x_boss = (symm_pos[0] * nx) % 1.0
    y_boss = (symm_pos[1] * ny) % 1.0
    z_boss = (symm_pos[2] * l_z) - z_top

    return [round_pos(x_boss), 
            round_pos(y_boss), 
            round_pos(z_boss)]

# Any analysis functions?
def dataframe_from_s_prime(s_prime: list, slab: Atoms, mol: Atoms, init_ads: Atoms, tol: float = 1e-3):
    """
    Converts the s_prime dataset into a comprehensible DataFrame with emphasis on adsorption energy.
    
    Parameters:
        s_prime (list): List of adsorption data tuples.
        slab (Atoms): The slab for adsorption.
        calculator: Calculator to compute energies.
        e_slab (float): Energy of the slab.
        e_co (float): Energy of the adsorbate molecule.
    
    Returns:
        pd.DataFrame: DataFrame with comprehensible adsorption data.
    """
    data = []
    calculator = EMT()

    slab.calc = calculator
    mol.calc = calculator
    init_ads.calc = calculator
    
    e_slab = slab.get_potential_energy()
    e_mol = mol.get_potential_energy()
    e_init = init_ads.get_potential_energy()

    e_ads_init = e_init - e_slab - e_mol
    e_ads_init = np.round(e_ads_init, decimals = 4)

    for i, (pos, ori, trans, uid) in enumerate(s_prime):
        x, y, z = pos
        alpha, beta, gamma = ori
        rotation_matrix = trans[0]
        translation_vector = trans[1]
        equivalent = False
        
        # Generate adsorption geometry using the create function
        symm_ads = create(slab, mol, alpha, beta, gamma, x, y, z)
        
        # Calculate adsorption energy
        symm_ads.calc = calculator

        e_system = symm_ads.get_potential_energy()
        e_ads = e_system - e_slab - e_mol
        e_ads = np.round(e_ads, decimals = 4)

        if math.isclose(e_ads, e_ads_init, abs_tol = tol):
            equivalent = True
        
        # Append data to list
        data.append({
            "Index": i,
            "Position": pos,
            "Orientation (alpha, beta, gamma)": (alpha, beta, gamma),
            "Rotation Matrix": rotation_matrix,
            "Translation Vector": translation_vector,
            "Rotation Determinant": np.linalg.det(rotation_matrix),
            "Adsorption Energy (eV)": e_ads,
            "Is it equivalent?": equivalent
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    return df

# Any analysis functions?
def list_from_s_prime(s_prime: list, slab: Atoms, mol: Atoms, init_ads: Atoms, tol: float = 1e-3):
    """
    Converts the s_prime dataset into a comprehensible DataFrame with emphasis on adsorption energy.
    
    Parameters:
        s_prime (list): List of adsorption data tuples.
        slab (Atoms): The slab for adsorption.
        calculator: Calculator to compute energies.
        e_slab (float): Energy of the slab.
        e_co (float): Energy of the adsorbate molecule.
    
    Returns:
        data: List with comprehensible adsorption data.
    """
    data = []
    calculator = EMT()

    slab.calc = calculator
    mol.calc = calculator
    init_ads.calc = calculator
    
    e_slab = slab.get_potential_energy()
    e_mol = mol.get_potential_energy()
    e_init = init_ads.get_potential_energy()

    e_ads_init = e_init - e_slab - e_mol
    e_ads_init = np.round(e_ads_init, decimals = 4)
    
    for i, (pos, ori, trans, uid) in enumerate(s_prime):
        x, y, z = pos
        alpha, beta, gamma = ori
        rotation_matrix = trans[0]
        translation_vector = trans[1]
        equivalent = False
        
        # Generate adsorption geometry using the create function
        symm_ads = create(slab, mol, alpha, beta, gamma, x, y, z)
        
        # Calculate adsorption energy
        symm_ads.calc = calculator
        e_system = symm_ads.get_potential_energy()
        e_ads = e_system - e_slab - e_mol
        e_ads = np.round(e_ads, decimals = 4)

        if math.isclose(e_ads, e_ads_init, abs_tol = tol):
            equivalent = True
        
        # Append data to list
        data.append({
            "idx": i,
            "position": pos,
            "orientation": (alpha, beta, gamma),
            "rotation": rotation_matrix,
            "translation": translation_vector,
            "det": np.linalg.det(rotation_matrix),
            "e_ads": e_ads,
            "equivalent": equivalent
        })
    
    return data

def dataframe_from_adatom(s_prime: list, slab: Atoms, mol: Atoms, init_ads: Atoms, tol: float = 1e-3):
    """
    Converts the s_prime dataset into a comprehensible DataFrame with emphasis on adsorption energy.
    
    Parameters:
        s_prime (list): List of adsorption data tuples.
        slab (Atoms): The slab for adsorption.
        mol (Atoms): The adsorbate molecule.
        init_ads (Atoms): Initial adsorption structure.
    
    Returns:
        pd.DataFrame: DataFrame with comprehensible adsorption data.
    """
    data = []
    calculator = EMT()

    slab.calc = calculator
    mol.calc = calculator
    init_ads.calc = calculator
    
    e_slab = slab.get_potential_energy()
    e_mol = mol.get_potential_energy()
    e_init = init_ads.get_potential_energy()

    e_ads_init = e_init - e_slab - e_mol
    e_ads_init = np.round(e_ads_init, decimals = 4)

    for i, (pos, trans) in enumerate(s_prime):
        x, y, z = pos
        rotation_matrix = trans[0]
        translation_vector = trans[1]
        equivalent = False
        
        # Generate adsorption geometry using the create function
        symm_ads = create_adatom(slab, mol, x, y, z)
        
        # Calculate adsorption energy
        symm_ads.calc = calculator

        e_system = symm_ads.get_potential_energy()
        e_ads = e_system - e_slab - e_mol

        e_ads = np.round(e_ads, decimals = 4)

        if math.isclose(e_ads, e_ads_init, abs_tol = tol):
            equivalent = True
        
        # Append data to list
        data.append({
            "Index": i,
            "Position": pos,
            "Rotation Matrix": rotation_matrix,
            "Translation Vector": translation_vector,
            "Rotation Determinant": np.linalg.det(rotation_matrix),
            "Adsorption Energy (eV)": e_ads,
            "Is it equivalent?": equivalent
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    return df   

def list_from_adatom(s_prime: list, slab: Atoms, mol: Atoms, init_ads: Atoms, tol: float = 1e-3):
    """
    Converts the s_prime dataset into a comprehensible list with emphasis on adsorption energy.
    
    Parameters:
        s_prime (list): List of adsorption data tuples.
        slab (Atoms): The slab for adsorption.
        mol (Atoms): The adsorbate molecule.
        init_ads (Atoms): Initial adsorption structure.
    
    Returns:
        data: List with comprehensible adsorption data.
    """
    data = []
    calculator = EMT()

    slab.calc = calculator
    mol.calc = calculator
    init_ads.calc = calculator
    
    e_slab = slab.get_potential_energy()
    e_mol = mol.get_potential_energy()
    e_init = init_ads.get_potential_energy()

    e_ads_init = e_init - e_slab - e_mol
    
    for i, (pos, trans) in enumerate(s_prime):
        x, y, z = pos
        rotation_matrix = trans[0]
        translation_vector = trans[1]
        equivalent = False
        
        # Generate adsorption geometry using the create function
        symm_ads = create_adatom(slab, mol, x, y, z)
        
        # Calculate adsorption energy
        symm_ads.calc = calculator
        e_system = symm_ads.get_potential_energy()
        e_ads = e_system - e_slab - e_mol

        e_ads = np.round(e_ads, decimals = 4)

        if math.isclose(e_ads, e_ads_init, abs_tol = tol):
            equivalent = True
        
        # Append data to list
        data.append({
            "idx": i,
            "position": pos,
            "rotation": rotation_matrix,
            "translation": translation_vector,
            "det": np.linalg.det(rotation_matrix),
            "e_ads": e_ads,
            "equivalent": equivalent
        })
    
    return data

def filter_reflections(data:dict):
    # Extract rotations and translations
    rotations = data['rotations']
    translations = data['translations']

    # Find indices of rotation matrices with determinant -1
    determinants = np.linalg.det(rotations)
    indices = np.where((determinants == -1) & np.all(np.isclose(translations, 0), axis=1))[0]

    # Create the filtered dictionary
    filtered_data = {
        'rotations': rotations[indices],
        'translations': translations[indices]
    }

    return filtered_data

def filter_z(data: dict):
    # Extract rotations and translations
    rotations = data['rotations']
    translations = data['translations']

    # Find indices of rotation matrices where the last row is NOT [0, 0, -1]
    indices = np.where(
        ~np.all(np.isclose(rotations[:, 2, :], np.array([0, 0, -1])), axis=1)
    )[0]

    # Create the filtered dictionary
    filtered_data = {
        'rotations': rotations[indices],
        'translations': translations[indices]
    }

    return filtered_data

def index_arrays(arr_list, use_letters=False):
    unique_items = []
    indices = {}

    # Determine labels (letters or numbers starting from 1)
    if use_letters:
        import string
        labels = string.ascii_uppercase
    else:
        labels = range(1, len(arr_list) + 1)

    # Assign each unique array an index
    for item in arr_list:
        if item.ndim == 1:
            key = tuple(item)  # Convert 1D array to a tuple
        else:
            key = tuple(map(tuple, item))  # Convert 2D array to a tuple of tuples

        if key not in indices:
            unique_items.append(key)
            indices[key] = labels[len(unique_items) - 1]

    # Convert list to indexed representation
    indexed_list = [
        indices[tuple(item) if item.ndim == 1 else tuple(map(tuple, item))]
        for item in arr_list
    ]

    return indexed_list

def unique_arrays(arrays):
    return [list(t) for t in {tuple(arr) for arr in arrays}]

def unique_arrays_ordered(arrays):
    seen = set()
    unique_list = []
    for arr in arrays:
        t = tuple(arr)
        if t not in seen:
            seen.add(t)
            unique_list.append(arr)
    return unique_list

def unique_numpy_arrays(arrays):
    seen = set()
    unique_list = []
    for arr in arrays:
        t = tuple(map(tuple, arr))  # Convert numpy array to a hashable type
        if t not in seen:
            seen.add(t)
            unique_list.append(arr)  # Append original numpy array
    return unique_list

def sort_lists(sort_by: list, *arrays):
    assert all(len(arr) == len(sort_by) for arr in arrays), "All lists must have the same length"
    
    sorted_indices = sorted(range(len(sort_by)), key=lambda i: sort_by[i])
    
    sorted_arrays = [[arr[i] for i in sorted_indices] for arr in arrays]
    
    return tuple(sorted_arrays)