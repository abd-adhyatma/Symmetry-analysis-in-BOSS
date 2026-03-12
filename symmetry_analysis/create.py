from ase import Atoms
import numpy as np

def create_adatom(slab_atoms:Atoms, adsorbate_atoms:Atoms, x:float, y:float, z:float, z_top:float=None):
    """
    Takes in slab, adsorbent, and BOSS input parameters (location only), returns adsorption geometry)
    Note: no nx and ny, assume whole slab is search area
    """
    slab = slab_atoms.copy()
    adsorbate = adsorbate_atoms.copy()
    
    # define cell for adsorbate
    cell = slab.get_cell()
    parameters = cell.cellpar(True)
    vx = parameters[0]
    vy = parameters[1]
    theta = parameters[5]

    # set the cell for the adsorbate
    adsorbate.set_cell(cell)

    # # center the adsorbate at the origin
    # adsorbate.center()
    
    # define dz
    if z_top == None:
        z_0 = slab.positions[:, 2].max()
    else:
        z_0 = z_top
        
    dz = z_0 + z

    # define x and y considering angle between a and b
    x = x+y*np.cos(theta)
    y = y*np.sin(theta)

    # define dx and dy
    dx = x * vx
    dy = y * vy
    
    # move adsorbent
    adsorbate.translate((dx, dy, dz))

    # define adsorption geometry
    ads = slab + adsorbate
    return ads

### DEFINING FUNCTIONS ###
def create(slab_atoms:Atoms, adsorbate_atoms:Atoms, alpha:float, beta:float, gamma:float, x:float, y:float, z:float, z_top:float = None):
    """
    Takes in slab, adsorbent, and BOSS input parameters (location and orientation), returns adsorption geometry)
    z_top value is optional, provide for corrugated / rumpled surfaces.
    """ 
    slab = slab_atoms.copy()
    adsorbate = adsorbate_atoms.copy()
    
    # define cell for adsorbate
    cell = slab.get_cell()
    parameters = cell.cellpar(True)
    vx = parameters[0]
    vy = parameters[1]
    theta = parameters[5]

    # set the cell for the adsorbate
    adsorbate.set_cell(cell)

    # # center the adsorbate at the origin
    # adsorbate.center()
    
    # define dz
    if z_top == None:
        z_0 = slab.positions[:, 2].max()
    else:
        z_0 = z_top
        
    dz = z_0 + z

    # define x and y considering angle between a and b
    x = x+y*np.cos(theta)
    y = y*np.sin(theta)

    # define dx and dy
    dx = x * vx
    dy = y * vy
    
    # orient adsorbent
    adsorbate.rotate(alpha, 'x', center=(0,0,0))
    adsorbate.rotate(beta, 'y', center=(0,0,0))
    adsorbate.rotate(gamma, 'z', center=(0,0,0))

    # move adsorbent
    adsorbate.translate((dx, dy, dz))

    # define adsorption geometry
    ads = slab + adsorbate
    return ads

# Create default function
def default_molecule(mol:Atoms, cell:int=5):
    """
    Create "default" of molecule in 5x5x5 box.
    """
    mol_default = mol.copy()
    mol_default.set_cell([cell,cell,cell])

    return mol_default

# Create rotate function
def rotate_molecule(mol:Atoms, angles_list:list, cell:int=5):
    """
    Rotates molecules in the center of the cell, mainly used to visualize and return rotated molecule.
    Takes in "raw" molecules without cell and located in the origin.
    Angles are given in the order of alpha, beta, gamma for rotations about x, y, and z, respectively.
    """
    alpha = angles_list[0]
    beta = angles_list[1]
    gamma = angles_list[2]
    
    mol_rotate = mol.copy()

    mol_rotate.rotate(alpha, 'x', center=(0,0,0))
    mol_rotate.rotate(beta, 'y', center=(0,0,0))
    mol_rotate.rotate(gamma, 'z', center=(0,0,0))

    return mol_rotate

def separate_ads_symbols(atoms: Atoms, slab_symbols: list, mol_symbols: list):
    symbols = atoms.get_chemical_symbols()

    slab_indices = [i for i, symbol in enumerate(symbols) if symbol in slab_symbols]
    mol_indices = [i for i, symbol in enumerate(symbols) if symbol in mol_symbols]

    slab = atoms[slab_indices]
    molecule = atoms[mol_indices]

    return slab, molecule

def create_from_s_prime(slab:Atoms, mol:Atoms, s_prime:tuple):
    """
    Creates 
    """
    a, b, c = s_prime[1]
    x, y, z = s_prime[0]
    ads = create(slab, mol, 
                 a, b, c,   ### PRESCRIBES ANGLES FROM S_PRIME
                 x, y, z)   ### PRESCRIBES POSITIONS FROM S_PRIME
    return ads