import numpy as np
from ase import Atoms
from spglib import get_symmetry_dataset
from .create import *
from .utility import *

def linear_vector(mol:Atoms):
    """ 
    If molecule has only 2 atoms, define a vector that describes the molecule's orientation.
    """
    # positions = mol.get_scaled_positions()
    positions = mol.get_positions()
    vector = positions[1] - positions[0]
    return vector

def triangle_vector(mol:Atoms, idx:list):
    """
    If molecule has more than 2 atoms, define a plane that is formed by 3 atoms in the molecule.
    The vector normal to that plane is used to describe the molecule's orientation
    By default, takes the first three atoms in the molecule.
    """
    # Define two non-parallel vectors in the molecule
    v1 = mol.get_distance(idx[0], idx[1], mic=False, vector=True)
    v2 = mol.get_distance(idx[0], idx[2], mic=False, vector=True)

    # Define vector normal to v1 and v2
    vector = np.cross(v1, v2)
    return  vector    

def ammonia_vectors(mol:Atoms):
    pos = mol.positions
    com_h3 = mol.get_center_of_mass(False, [1,2,3])
    com_nh = mol.get_center_of_mass(False, [0,1])

    v_com_h = com_h3 - pos[1]
    v_com_n = com_h3 - pos[0]
    
    v1 = np.cross(v_com_h, v_com_n)
    v2 = com_h3 - pos[0]

    return v1, v2

def apply_position_transformation(
    pos:np.ndarray, P:np.ndarray, p:np.ndarray, W:np.ndarray, t:np.ndarray
):
    """ 
    Takes in fractional coordinate of an atom in a molecule and applies symmetry transformation.
    """
    # pre-transformation to generalize for non-orthorhombic cases
    W = np.dot(np.linalg.inv(P), np.dot(W, P))
    t = np.dot(np.linalg.inv(P), t)
    x = np.dot(np.linalg.inv(P), pos) - np.dot(np.linalg.inv(P), p)
    
    x_prime = np.dot(W, x) + t
    pos_prime = np.dot(P, x_prime) + p
    return pos_prime

def rotation_matrix_from_vectors(A:np.ndarray, B:np.ndarray):
    """
    Compute the rotation matrix that aligns vector A to vector B.
    
    Parameters:
        A (array-like): Initial vector.
        B (array-like): Target vector.

    Returns:
        R (ndarray): 3x3 rotation matrix.
    """
    # Normalize the input vectors
    A = A / np.linalg.norm(A)
    B = B / np.linalg.norm(B)

    # Compute the cross product and angle
    v = np.cross(A, B)
    c = np.dot(A, B)

    if np.linalg.norm(v) < 1e-8:  # Vectors are parallel or anti-parallel
        if c > 0:  # Same direction
            return np.eye(3)
        else:  # Opposite direction
            # Find a perpendicular vector to construct the rotation matrix
            perp = np.array([1, 0, 0]) if abs(A[0]) < 0.9 else np.array([0, 1, 0])
            v = np.cross(A, perp)
            v /= np.linalg.norm(v)
            K = np.array([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ])
            return np.eye(3) + 2 * K @ K  # 180-degree rotation matrix

    # Normalize the cross product to get the rotation axis
    v = v / np.linalg.norm(v)
    x, y, z = v

    # Define angle of rotation, theta = arccos(c)
    theta = np.arccos(c)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Derive the rotation matrix using an explicit 3x3 matrix form of the Rodriguez formula
    rotation_matrix = np.array([
    [cos_theta + x**2 * (1 - cos_theta),         x * y * (1 - cos_theta) - z * sin_theta, x * z * (1 - cos_theta) + y * sin_theta],
    [y * x * (1 - cos_theta) + z * sin_theta,    cos_theta + y**2 * (1 - cos_theta),      y * z * (1 - cos_theta) - x * sin_theta],
    [z * x * (1 - cos_theta) - y * sin_theta,    z * y * (1 - cos_theta) + x * sin_theta, cos_theta + z**2 * (1 - cos_theta)]
    ])

    return rotation_matrix

def rotation_matrix_from_points_and_angle(point1, point2, degree):
    """
    Compute the rotation matrix using the Rodrigues formula.
    
    Args:
        point1 (list or np.array): Coordinates of the first point [x1, y1, z1].
        point2 (list or np.array): Coordinates of the second point [x2, y2, z2].
        degree (float): Rotation angle in degrees.
        
    Returns:
        np.array: 3x3 rotation matrix.
    """
    # Convert degree to radians
    theta = np.radians(degree)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # Compute the direction vector (v) between the two points
    point1 = np.array(point1)
    point2 = np.array(point2)
    v = point2 - point1
    
    # Normalize the vector to get the rotation axis
    v = v / np.linalg.norm(v)
    x, y, z = v
    
    # Derive the rotation matrix using Rodrigues' formula
    rotation_matrix = np.array([
        [cos_theta + x**2 * (1 - cos_theta),         x * y * (1 - cos_theta) - z * sin_theta, x * z * (1 - cos_theta) + y * sin_theta],
        [y * x * (1 - cos_theta) + z * sin_theta,    cos_theta + y**2 * (1 - cos_theta),      y * z * (1 - cos_theta) - x * sin_theta],
        [z * x * (1 - cos_theta) - y * sin_theta,    z * y * (1 - cos_theta) + x * sin_theta, cos_theta + z**2 * (1 - cos_theta)]
    ])
    
    return rotation_matrix

def align_vectors(set1, set2):
    """
    Compute the rotation matrix that aligns two sets of two vectors, from set1 to set2.

    Parameters:
    set1 (list of np.ndarray): The first set of two vectors [v1, v2].
    set2 (list of np.ndarray): The second set of two vectors [w1, w2].

    Returns:
    np.ndarray: A 3x3 rotation matrix aligning set1 to set2.
    """
    # Normalize the input vectors
    v1, v2 = [v / np.linalg.norm(v) for v in set1]
    w1, w2 = [w / np.linalg.norm(w) for w in set2]

    # Compute the third vector using the cross product to ensure orthonormality
    v3 = np.cross(v1, v2)
    w3 = np.cross(w1, w2)

    # Form orthonormal basis matrices
    U1 = np.column_stack((v1, v2, v3))
    U2 = np.column_stack((w1, w2, w3))

    # Compute the rotation matrix
    R = U2 @ U1.T

    return R

def rotation_from_transformed(
    mol:Atoms, slab:Atoms, orientation:list, rotation:np.ndarray, translation:np.ndarray, idx:list
):
    """
    This function generates a rotation matrix that aligns a molecule in its "initial" orientation to its transformed image.
    If molecule only has 2 atoms, linear_vector function is used. Else, use triangle_vector.
    """
    ### PROCESSING INPUTS
    # Getting P, p, and cell from slab
    dataset = get_symmetry_dataset(get_spglib_cell(slab))
    P = dataset.transformation_matrix
    p = dataset.origin_shift
    cell = slab.get_cell()
    
    # Rotate molecule to initial orientation
    mol_rotate = rotate_molecule(mol, orientation)
    mol_rotate.set_cell(cell)

    ### APPLYING p' = pW + w
    # Get initial positions
    pos_init = mol_rotate.get_scaled_positions()
    # Transform positions
    pos_transformed = [apply_position_transformation(pos, P, p, rotation, translation) for pos in pos_init]

    # Get list of reflected positions
    mol_transformed = mol_rotate.copy()
    mol_transformed.set_scaled_positions(pos_transformed)

    if len(mol) == 2:
        # Get v_init from initial orientation
        v_init = linear_vector(mol_rotate)

        # Define vector correspoding to O'
        v_transformed = linear_vector(mol_transformed)

        R_prime = rotation_matrix_from_vectors(v_init, v_transformed)

    if len(mol) > 2:
        # Get set_init from initial orientation
        v1 = triangle_vector(mol_rotate, idx=idx)
        v2 = mol_rotate.positions[idx[0]] - mol_rotate.get_center_of_mass(False, [idx[1], idx[2]])
        set_init = [v1, v2]

        # Get set_reflected from final orientation
        w1 = triangle_vector(mol_transformed, idx=idx)       
        w2 = mol_transformed.positions[idx[0]] - mol_transformed.get_center_of_mass(False, [idx[1], idx[2]])
        set_transformed = [w1, w2]

        ### GETTING ROTATION MATRIX, R'
        R_prime = align_vectors(set_init, set_transformed)

    determinant = np.linalg.det(R_prime)
    if determinant != 0:  # Avoid division by zero
        R_prime /= determinant

    return R_prime
