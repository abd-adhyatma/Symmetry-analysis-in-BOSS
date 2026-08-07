import numpy as np
import sys
from ase import Atoms
from ase.io import read, write
from spglib import get_symmetry, get_symmetry_dataset
from .utility import *
from .transformations import *
import ira_mod

def mol_to_symm(mol:Atoms):
    nat = len(mol)
    typ = np.array([s for s in mol.get_atomic_numbers()])
    coords = np.array([p for p in mol.get_positions()])
    
    sofi = ira_mod.SOFI()
    sym_thr = 0.05
    
    n_mat, mat_list = sofi.get_symm_ops(nat, typ, coords, sym_thr)
    rot_list = [x for x in mat_list if np.linalg.det(x)>0]

    # sym = sofi.compute(nat, typ, coords, sym_thr)
    # mat_list = sym.matrix
    # rot_list = [x for x in mat_list if np.linalg.det(x)>0]
    return rot_list

def get_symmetric_positions_and_angles_with_SOFI(
    pos: list, angles: list, slab: Atoms, mol: Atoms, idx=None, 
    x_bounds:list=[0., 1.], y_bounds:list=[0., 1.], chiral:bool=False, 
    output_filename:str='log.txt', tolerance:float=1e-3
):
    """
    Get unique equivalent positions in x, y, z and orientations (alpha, beta, gamma) based on the symmetries of a slab.
    Ensures unique pairs and that the z-coordinates match the original position's z within tolerance.
    Additionally logs symmetry operations to a file, including the determinant of each rotation.
    Parameters:
    pos (list): containing scaled coordinates x, y, z, respectively, of molecule center of mass position on slab.
    angles (list): containing alpha, beta, gamma angles, respectively, of molecule rotation.
    slab (ase.Atoms object): slab system isolated from the adsorbed molecule.
    mol (ase.Atoms object): molecule isolated from the slab. Also provides molecule symmetries via SOFI.
    idx (list): list of atom indeces in the molecule that will be used for plane reference in reflection cases.
    lower, upper_bound (int): Sets bounds for symmetric position search.
    chiral (bool): whether or not molecule is chiral
    output_filename (str): name of the file to write the symmetry operations and results.
    tolerance: tolerance for similarity in z position (in scaled coordinates).

    Returns:
    unique, duplicate, out_of_bounds: lists of unique, duplicate, and out-of-bounds adsorption configurations.
    """
    if len(x_bounds) != 2 or len(y_bounds) != 2:
        raise ValueError("Bounds for x or y must be specified as a list, i.e. [lowerbound, upperbound]")
        
    # if mol_symmetries is None:
    #     mol_symmetries = [np.eye(3)]
    
    if idx is None:
        idx = [0, 1, 2]
    if len(angles) != 3:
        raise ValueError("Angles must have exactly three elements (alpha, beta, gamma).")

    slab = slab.copy()
    mol = mol.copy()
    
    cell = get_spglib_cell(slab)
    symmetry = filter_z(get_symmetry(cell, 1e-5))
    dataset = get_symmetry_dataset(cell, 1e-5)
    P = dataset.transformation_matrix
    p = dataset.origin_shift
    pos = boss_to_symm(pos, slab)
    rotations = symmetry['rotations']
    translations = symmetry['translations']
    R_init = euler_to_rotation(angles[0], angles[1], angles[2], return_Rs=False)

    # integration of SOFI
    mol_symmetries = mol_to_symm(mol)
    
    unique_positions_and_angles = []
    duplicate_positions_and_angles = []
    out_of_bounds_points = []

    rotations_idx = index_arrays(rotations)
    translations_idx = index_arrays(translations)
    mol_symm_idx = index_arrays(mol_symmetries)
    
    with open(output_filename, 'w') as file:
        file.write("Symmetry Operations Check Log\n")
        file.write(f"Rotation matrices passed for operations are the following:\n")
        for i, rot in enumerate(unique_numpy_arrays(rotations)):
            file.write(f"{i+1}. \n{rot}\n")
        file.write(f"Translation vectors passed for operations are the following:\n")
        for i, trans in enumerate(unique_arrays(translations)):
            file.write(f"{i+1}. {trans}\n")
        file.write(f"Molecule symmetry matrices passed for operations are the following:\n")
        for i, symm in enumerate(unique_numpy_arrays(mol_symmetries)):
            file.write(f"{i+1}. \n{symm}\n")
        file.write("=" * 80 + "\n")
        file.write("Index | Pass/Fail | Failed Check | Position | Orientation | Det(W_surf) | W_surf | w_surf | R_symm | R_O'| UID\n")
        
        for i, (rotation, translation, rot_idx, trans_idx) in enumerate(zip(rotations, translations, rotations_idx, translations_idx)):
            determinant = np.linalg.det(rotation)
            if chiral and determinant < 0:
                file.write(f"{i+1} | Fail | Reflecting chiral molecule | {symm_to_boss(new_pos, slab)} | N/A | {determinant:.3f} | {rot_idx} | {trans_idx} | N/A | N/A | N/A\n")
                continue
            
            # new_pos = np.dot(rotation, pos) + translation
            new_pos = apply_position_transformation(pos, P, p, rotation, translation)
            new_pos = new_pos % 1.0
            new_pos_boss = symm_to_boss(new_pos, slab)

            out_of_bounds = False
            if not (x_bounds[0] <= new_pos_boss[0] <= x_bounds[1]) or not (y_bounds[0] <= new_pos_boss[1] <= y_bounds[1]):
                out_of_bounds = True

            # # if not (lb <= new_pos_boss[0] <= ub) or not (lb <= new_pos_boss[1] <= ub): CONDITION FOR X AND Y TO BE BETWEEN BOUNDS
            # if not (x_bounds[0] <= new_pos_boss[0] <= x_bounds[1]) or not (y_bounds[0] <= new_pos_boss[1] <= y_bounds[1]):
            #     uid = f"W{rot_idx}_w{trans_idx}_R0_O0"
            #     out_of_bounds_points.append(
            #         (
            #             np.array(new_pos_boss),
            #             np.array([]), # Empty array because O' not yet processed
            #             (rotation, translation),
            #             uid
            #         )
            #     )
            #     file.write(f"{i+1} | Fail | Out-of-bounds | {symm_to_boss(new_pos, slab)} | N/A | {determinant:.3f} | {rot_idx} | {trans_idx} | N/A | N/A | N/A\n")
            #     continue
            
            for (mol_symmetry, symm_idx) in zip(mol_symmetries, mol_symm_idx):
                rotation_trans = rotation_from_transformed(mol=mol, slab=slab, orientation=angles, rotation=rotation, translation=translation, idx=idx)
                R_new = np.dot(rotation_trans, np.dot(R_init, mol_symmetry))
                new_orientations = rotation_to_euler_rounded(R_new)

                for j, new_orientation in enumerate(new_orientations):
                    uid = f"{i+1}_W{rot_idx}_w{trans_idx}_R{symm_idx}_O{j+1}"
                    if out_of_bounds:
                        out_of_bounds_points.append(
                            (
                                np.array(new_pos_boss), 
                                np.array(new_orientation),
                                (rotation, translation),
                                uid
                            )
                        )
                        file.write(f"{i+1} | Fail | Out-of-bounds | {new_pos_boss} | {new_orientation} | {determinant:.3f} | {rot_idx} | {trans_idx} | {symm_idx} | {j+1} | {uid}\n")                    
                        continue
                        
                    pair_is_unique = True
                    for existing_pos, existing_angle, _, _ in unique_positions_and_angles:
                        if (np.allclose(new_pos_boss, existing_pos, atol=tolerance) and
                            np.allclose(new_orientation, existing_angle, atol=tolerance)):
                            pair_is_unique = False
                            break

                    if pair_is_unique:
                        unique_positions_and_angles.append(
                            (
                                np.array(new_pos_boss), 
                                np.array(new_orientation),
                                (rotation, translation),
                                uid
                            )
                        )
                        file.write(f"{i+1} | Pass | None | {new_pos_boss} | {new_orientation} | {determinant:.3f} | {rot_idx} | {trans_idx} | {symm_idx} | {j+1} | {uid}\n")
                    
                    else:
                        duplicate_positions_and_angles.append(
                            (
                                np.array(new_pos_boss), 
                                np.array(new_orientation),
                                (rotation, translation),
                                uid
                            )
                        )
                        file.write(f"{i+1} | Fail | Non-unique | {new_pos_boss} | {new_orientation} | {determinant:.3f} | {rot_idx} | {trans_idx} | {symm_idx} | {j+1} | {uid}\n")
    
    return unique_positions_and_angles, duplicate_positions_and_angles, out_of_bounds_points