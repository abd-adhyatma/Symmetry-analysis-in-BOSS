############################################ IMPORTING LIBRARIES ############################################
import math
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting
import matplotlib.colors as mcolors
from ase import Atoms  # ASE's Atoms class
from ase.visualize.plot import plot_atoms
from .create import *
from .transformations import *

### Creating xyz-projection visualization functions for molecule
def plot_atoms_and_plane(atoms: Atoms, idx:list = [0,1,2], cell:int = 5, offset:float = 0.125, filename: str = 'atoms_projections.png'):
    # Set up a row of subplots to show x, y, and z views
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    mol = atoms.copy()    

    # Define correct rotation views
    views = ['90x,-90y,-180z', '-90x,0y,0z', '0x,0y,0z']  # Adjusted to match each axis' positive direction
    projection_planes = [(1, 2), (0, 2), (0, 1)]  # Aligned with the views
    view_titles = ['View from x-axis', 'View from y-axis', 'View from z-axis']
    
    # Plot each view
    for ax, v, (x_idx, y_idx), title in zip(axes, views, projection_planes, view_titles):

        # Prepare atom view
        mol.set_cell([cell, cell, cell])
        mol.center()
        plot_atoms(mol, ax, radii=0.5, rotation=v)

        ax.set_title(title)
        # Set axis limits to make sure the plot fits well within the view
        ax.set_xlim(0, cell)
        ax.set_ylim(0, cell)
    
        # Add axis labels based on the rotation view
        if title == 'View from x-axis':
            ax.set_xlabel('Y-axis')
            ax.set_ylabel('Z-axis')
        elif title == 'View from y-axis':
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Z-axis')
        else:  # 'View from z-axis'
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')

        # Add a grid for reference
        ax.grid(True, linestyle='--', alpha=0.5)

        # Get the positions and species of specified atoms
        positions = mol.positions[idx]
        species = [mol.get_chemical_symbols()[i] for i in idx]
        
        # Extract the projections of each atom onto the current plane
        projections = [(pos[x_idx]+offset, pos[y_idx]+offset) for pos in positions]
        
        # Unpack the projections into x and y coordinates
        x_coords, y_coords = zip(*projections)
        
        # Plot the triangle connecting the three atoms
        ax.fill(x_coords, y_coords, color='blue', alpha=0.3, edgecolor='black', linewidth=2)
        ax.scatter(x_coords, y_coords, color='red', s=50)  # Mark the atom positions
        
        # Annotate the atomic species
        for (x, y), symbol in zip(projections, species):
            ax.text(x, y, symbol, fontsize=12, color='black', ha='center', va='center',
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='circle,pad=0.3'))
    
    plt.tight_layout()
    plt.savefig(filename)

# Creating isometric visualization function
def plot_atoms_isometric(mol: Atoms, filename: str = 'atoms_isometric.png'):
    # Visualize
    fig, ax = plt.subplots()
    plot_atoms(mol, ax, radii=0.5, rotation=('-80x,30y,5z'))  # Adjust radii and rotation if needed
    # Remove axes and ticks
    ax.axis('off')
    plt.savefig(filename)

### Creating xyz-projection visualization functions for molecule
def plot_atoms_projections(atoms: Atoms, cell:int=5, filename: str = 'atoms_projections.png'):
    # Set up a row of subplots to show x, y, and z views
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    mol = atoms.copy()    

    # Define correct rotation views
    views = ['90x,-90y,-180z', '-90x,0y,0z', '0x,0y,0z']  # Adjusted to match each axis' positive direction
    view_titles = ['View from x-axis', 'View from y-axis', 'View from z-axis']
    
    # Plot each view
    for ax, v, title in zip(axes, views, view_titles):
        mol.set_cell([cell, cell, cell])
        mol.center()
        plot_atoms(mol, ax, radii=0.5, rotation=v)
        ax.set_title(title)
    
        # Add axis labels based on the rotation view
        if title == 'View from x-axis':
            ax.set_xlabel('Y-axis')
            ax.set_ylabel('Z-axis')
        elif title == 'View from y-axis':
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Z-axis')
        else:  # 'View from z-axis'
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')

        # Set axis limits to make sure the plot fits well within the view
        ax.set_xlim(0, cell)
        ax.set_ylim(0, cell)

        # Add a grid for reference
        ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(filename)

def plot_linear_arrow_projections(atoms: Atoms, cell:int=5, filename: str = 'arrow_projections.png'):
    """
    Be wary of what order the atoms are in. The order points from index 0 atom to index 1 atom.
    Also, this view has been matched to ASE's view function.
    """
    mol = atoms.copy()
    
    # Set up a row of subplots to show x, y, and z views
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Define projection planes for each view
    view_titles = ['View from x-axis', 'View from y-axis', 'View from z-axis']
    projection_planes = [(1, 2), (0, 2), (0, 1)]  # Correctly aligned with the rotation views

    # Plot each view with an arrow representing the CO molecule
    for ax, title, (x_idx, y_idx) in zip(axes, view_titles, projection_planes):
        # Center molecule for easier viewing
        mol.set_cell([cell, cell, cell])
        mol.center()

        ax.set_title(title)
        
        # Set explicit limits to make sure the arrow fits within the view
        ax.set_xlim(0, cell)
        ax.set_ylim(0, cell)

        # Add axis labels based on the view title
        if title == 'View from x-axis':
            ax.set_xlabel('Y-axis')
            ax.set_ylabel('Z-axis')
        elif title == 'View from y-axis':
            ax.set_xlabel('X-axis')  # Correctly label for the y-axis view
            ax.set_ylabel('Z-axis')
        else:  # 'View from z-axis'
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            
            # Draw lines across the center of the cell
            center = cell/2
            ax.axhline(y=center, color='blue', linestyle='--', linewidth=1)  # Horizontal center line
            ax.axvline(x=center, color='blue', linestyle='--', linewidth=1)  # Vertical center line
            
            # Add y=x and y=-x lines
            x_vals = [0, cell]  # x range
            y_vals_pos = [0, cell]  # y = x
            y_vals_neg = [cell, 0]  # y = -x
            ax.plot(x_vals, y_vals_pos, color='green', linestyle='--', linewidth=1, label='y = x')
            ax.plot(x_vals, y_vals_neg, color='orange', linestyle='--', linewidth=1, label='y = -x')
            
            # Add a legend to clarify the added lines
            ax.legend()

        # Add a grid for reference
        ax.grid(True, linestyle='--', alpha=0.5)
    
        # Get the positions
        a1_pos = mol.positions[0]  # 1st atom position
        a2_pos = mol.positions[1]  # 2nd atom position
        
        # Project onto the appropriate plane for the current view
        a1_proj = (a1_pos[x_idx], a1_pos[y_idx])
        a2_proj = (a2_pos[x_idx], a2_pos[y_idx])
    
        # Plot an arrow from C to O in the current view
        ax.annotate(
            '', xy=a2_proj, xytext=a1_proj,
            arrowprops=dict(arrowstyle="->", color='red', lw=3)
        )
    
    plt.tight_layout()
    plt.savefig(filename)

def plot_planar_triangle_projections(mol: Atoms, cell:int=5, filename: str = 'triangle_projections.png'):
    # Set up a row of subplots to show x, y, and z views
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Define projection planes for each view
    view_titles = ['View from x-axis', 'View from y-axis', 'View from z-axis']
    projection_planes = [(1, 2), (0, 2), (0, 1)]  # Aligned with the views

    # Plot each view with a triangle representing the molecule
    for ax, title, (x_idx, y_idx) in zip(axes, view_titles, projection_planes):
        ax.set_title(title)

        # Center molecule for easier viewing
        mol.set_cell([cell, cell, cell])
        mol.center()

        # Set explicit limits to make sure the arrow fits within the view
        ax.set_xlim(0, cell)
        ax.set_ylim(0, cell)

        # Add axis labels based on the view title
        if title == 'View from x-axis':
            ax.set_xlabel('Y-axis')
            ax.set_ylabel('Z-axis')
        elif title == 'View from y-axis':
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Z-axis')
        else:  # 'View from z-axis'
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
        
        # Add a grid for reference
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Get the positions and species of all atoms
        positions = mol.positions
        species = mol.get_chemical_symbols()
        
        # Extract the projections of each atom onto the current plane
        projections = [(pos[x_idx], pos[y_idx]) for pos in positions]
        
        # Unpack the projections into x and y coordinates
        x_coords, y_coords = zip(*projections)
        
        # Plot the triangle connecting the three atoms
        ax.fill(x_coords, y_coords, color='blue', alpha=0.3, edgecolor='black', linewidth=2)
        ax.scatter(x_coords, y_coords, color='red', s=50)  # Mark the atom positions
        
        # Annotate the atomic species
        for (x, y), symbol in zip(projections, species):
            ax.text(x, y, symbol, fontsize=12, color='black', ha='center', va='center',
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='circle,pad=0.3'))
            
    plt.tight_layout()
    plt.savefig(filename)

def plot_atoms_and_vectors(atoms: Atoms, idx: list = [0, 1, 2], cell: int = 5, offset: float = 0.125, 
                           annotate:bool = False, filename: str = 'atoms_and_vectors.png'):
    """
    Creates a visualization of a molecule with its projections on the x, y, and z planes.
    It also plots vectors defined by the atoms and their respective orientations.
    
    Parameters:
        atoms (Atoms): The ASE Atoms object representing the molecule.
        idx (list): List of indices of atoms to define the plane and vectors.
        cell (int): Size of the cell for visualization.
        offset (float): Offset for the projections.
        filename (str): File name to save the output image.
    """
    # Set up a row of subplots to show x, y, and z views
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    mol = atoms.copy()

    # Define correct rotation views
    views = ['90x,-90y,-180z', '-90x,0y,0z', '0x,0y,0z']  # Adjusted to match each axis' positive direction
    projection_planes = [(1, 2), (0, 2), (0, 1)]  # Aligned with the views
    view_titles = ['View from x-axis', 'View from y-axis', 'View from z-axis']
    
    # Prepare atom view
    mol.set_cell([cell, cell, cell])
    mol.center()

    # Calculate vectors and positions for plotting
    positions = mol.positions[idx]
    # species = [mol.get_chemical_symbols()[i] for i in idx]

    pos_0 = positions[0]
    vector = triangle_vector(mol, idx)  # Normal vector to the plane
    pos_1 = pos_0 + vector
    pos_2 = positions[1] + (positions[2] - positions[1]) / 2  # Midpoint between atoms idx[1] and idx[2]

    # Plot each view
    for ax, v, (x_idx, y_idx), title in zip(axes, views, projection_planes, view_titles):
        # Plot atoms
        plot_atoms(mol, ax, radii=0.5, rotation=v)
        ax.set_title(title)
        ax.set_xlim(0, cell)
        ax.set_ylim(0, cell)
        ax.set_xlabel(f"{'XYZ'[x_idx]}-axis")
        ax.set_ylabel(f"{'XYZ'[y_idx]}-axis")
        ax.grid(True, linestyle='--', alpha=0.5)

        # Plot vectors and highlight the key points
        # Project positions to the current plane
        p0_proj = pos_0[[x_idx, y_idx]] + offset

        # Calculate the vector from pos_0 to pos_2, and then scale the vector from pos_0 to pos_1
        vector_length = np.linalg.norm(pos_2 - pos_0)  # Length of the vector from pos_0 to pos_2
        vector_direction = (pos_1 - pos_0) / np.linalg.norm(pos_1 - pos_0)  # Normalized direction of the vector from pos_0 to pos_1
        
        # Scale pos_1 to match the length of pos_2 - pos_0
        pos_1_scaled = pos_0 + vector_length * vector_direction
        
        # Apply offset to the projections
        p1_proj = pos_1_scaled[[x_idx, y_idx]] + offset

        p2_proj = pos_2[[x_idx, y_idx]] + offset

        # Plot the origin point and vectors
        # ax.scatter(*p0_proj, color='red', label=f"Atom {idx[0]} ({species[0]})", zorder=5)
        ax.quiver(*p0_proj, *(p1_proj - p0_proj), angles='xy', scale_units='xy', scale=1, color='cyan', label='Vector 1', zorder=3)
        ax.quiver(*p0_proj, *(p2_proj - p0_proj), angles='xy', scale_units='xy', scale=1, color='magenta', label='Vector 2', zorder=3)

        # Annotate positions
        if annotate:
            # ax.text(*p0_proj + offset, f"{species[0]}(0)", color='red', fontsize=9)
            ax.text(*p1_proj + offset, "Vec1", color='cyan', fontsize=9)
            ax.text(*p2_proj + offset, "Vec2", color='magenta', fontsize=9)

    # Adjust layout and save
    plt.tight_layout()
    # plt.legend(loc='upper right', fontsize=8)
    plt.savefig(filename)
    plt.show()

### PLOT ADSORPTION GEOMETRY PROJECTION VISUALIZATION FUNCTIONS
def plot_adsorption_atoms_projections(ads:Atoms, filename:str = 'ads_atoms_projections.png'):
    # Set up a row of subplots to show x, y, and z views
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Define correct rotation views
    views = ['90x,-90y,-180z', '-90x,0y,0z', '0x,0y,0z']  # Adjusted to match each axis' positive direction
    view_titles = ['View from x-axis', 'View from y-axis', 'View from z-axis']

    # Plot each view
    for ax, v, title in zip(axes, views, view_titles):
        plot_atoms(ads, ax, radii=0.5, rotation=v)
        ax.set_title(title)

        # Add axis labels based on the rotation view
        if title == 'View from x-axis':
            ax.set_xlabel('Y-axis')
            ax.set_ylabel('Z-axis')
        elif title == 'View from y-axis':
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Z-axis')
        else:  # 'View from z-axis'
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')

        # Add a grid for reference
        ax.grid(True, linestyle='--', alpha=0.5)

    # Adjust spacing to remove gaps between subplots
    plt.tight_layout()
    plt.savefig(filename)

def plot_adatom_adsorption_set(slab:Atoms, adsorbate:Atoms, processed_symmetric_set:list, 
                                  perspective:str='z', offset:float=0.775, filename:str = 'ads_arrows_projected.png',
                                  annotated:bool=False):
    # Set up a single plot for the x-axis view
    fig, ax = plt.subplots(figsize=(5, 5))

    # Define rotation view, view_title, and projection planes
    if perspective == 'z':
        view_title = 'View from z-axis'
        rotation_view = '0x,0y,0z'
        projection_plane = (0, 1)
        add_axes_lines = True

    elif perspective == 'y':
        view_title = 'View from y-axis'
        rotation_view = '-90x,0y,0z'
        projection_plane = (0, 2)
        add_axes_lines = False

    elif perspective == 'x':
        view_title = 'View from x-axis'
        rotation_view = '90x,-90y,-180z'
        projection_plane = (1, 2)  
        add_axes_lines = False              

    # Plot slab
    plot_atoms(slab, ax, radii=0.5, rotation=rotation_view)
    ax.set_title(view_title)

    # Add axis labels for the x-axis view
    axis_labels = {'z': ('X-axis', 'Y-axis'),
                'y': ('X-axis', 'Z-axis'),
                'x': ('Y-axis', 'Z-axis')}
    xlabel, ylabel = axis_labels[perspective]
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Add a grid for reference
    ax.grid(True, linestyle='--', alpha=0.5)

    slab_list = slab.get_chemical_symbols()
    mol_list = adsorbate.get_chemical_symbols()

    # Generate list of molecules
    for i, symm in enumerate(processed_symmetric_set):
        # Extract positional data
        pos = symm['position']
        det = symm['det']

        x, y, z = pos
        # alpha, beta, gamma = ori

        # Making adsorption geometry
        symm_ads = create_adatom(slab, adsorbate, x, y, z)

        _, mol = separate_ads_symbols(symm_ads, slab_list, mol_list)

        # Get the positions
        a1_pos = mol.positions[0]  # 1st atom position

        # Project onto the appropriate plane for the current view
        a1_proj = (a1_pos[projection_plane[0]] + offset, a1_pos[projection_plane[1]] + offset)

        if math.isclose(det, 1):
            c = 'red'
        else:
            c = 'blue'

        if i == 0:
            mark = 'o'
        else:
            mark = 'x'

        # Plot a big cross at a1
        ax.scatter(*a1_proj, color=c, s=100, marker=mark, linewidths=2)

    plt.tight_layout()
    plt.savefig(filename)

def plot_adsorption_arrow_projections(ads:Atoms, slab_symbols:list, mol_symbols:list, 
                                 arrow_offset:float=0.75, filename:str = 'ads_arrow_projected.png'):
    
    # Set up a row of subplots to show x, y, and z views
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Separate slab and molecule
    slab, mol = separate_ads_symbols(ads, slab_symbols, mol_symbols)

    # Define projection planes for each view
    view_titles = ['View from x-axis', 'View from y-axis', 'View from z-axis']
    views = ['90x,-90y,-180z', '-90x,0y,0z', '0x,0y,0z'] 
    projection_planes = [(1, 2), (0, 2), (0, 1)]  # Correctly aligned with the rotation views

    # Plot each view with an arrow representing the CO molecule
    for ax, title, v, (x_idx, y_idx) in zip(axes, view_titles, views, projection_planes):
        ax.set_title(title)

        # Add axis labels based on the view title
        if title == 'View from x-axis':
            ax.set_xlabel('Y-axis')
            ax.set_ylabel('Z-axis')
        elif title == 'View from y-axis':
            ax.set_xlabel('X-axis')  # Correctly label for the y-axis view
            ax.set_ylabel('Z-axis')
        else:  # 'View from z-axis'
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')

        # Add a grid for reference
        ax.grid(True, linestyle='--', alpha=0.5)

        ### PLOT THE SLAB ATOMS USING plot_atoms FUNCTION
        plot_atoms(slab, ax, radii=0.5, rotation=v)

        # Get the positions
        a1_pos = mol.positions[0]  # 1st atom position
        a2_pos = mol.positions[1]  # 2nd atom position

        # Project onto the appropriate plane for the current view
        a1_proj = (a1_pos[x_idx]+arrow_offset, a1_pos[y_idx]+arrow_offset)
        a2_proj = (a2_pos[x_idx]+arrow_offset, a2_pos[y_idx]+arrow_offset)

        # Plot an arrow from a1 to a2 in the current view
        ax.annotate(
            '', xy=a2_proj, xytext=a1_proj,
            arrowprops=dict(arrowstyle="->", color='red', lw=3)
        )

    plt.tight_layout()
    plt.savefig(filename)

def plot_linear_adsorption_set(slab:Atoms, adsorbate:Atoms, processed_symmetric_set:list, 
                                  perspective:str='z', arrow_offset:float=0.775, filename:str = 'ads_arrows_projected.png',
                                  annotated:bool=False):
    # Set up a single plot for the x-axis view
    fig, ax = plt.subplots(figsize=(5, 5))

    # Define rotation view, view_title, and projection planes
    if perspective == 'z':
        view_title = 'View from z-axis'
        rotation_view = '0x,0y,0z'
        projection_plane = (0, 1)
        add_axes_lines = True

    elif perspective == 'y':
        view_title = 'View from y-axis'
        rotation_view = '-90x,0y,0z'
        projection_plane = (0, 2)
        add_axes_lines = False

    elif perspective == 'x':
        view_title = 'View from x-axis'
        rotation_view = '90x,-90y,-180z'
        projection_plane = (1, 2)  
        add_axes_lines = False              

    # Plot slab
    plot_atoms(slab, ax, radii=0.5, rotation=rotation_view)
    ax.set_title(view_title)

    # Add axis labels for the x-axis view
    axis_labels = {'z': ('X-axis', 'Y-axis'),
                'y': ('X-axis', 'Z-axis'),
                'x': ('Y-axis', 'Z-axis')}
    xlabel, ylabel = axis_labels[perspective]
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Add a grid for reference
    ax.grid(True, linestyle='--', alpha=0.5)

    # if add_axes_lines:
    #     center_x = slab.get_cell()[0][projection_plane[0]] / 2
    #     center_y = slab.get_cell()[1][projection_plane[1]] / 2

    #     x_dashed_line = arrow_offset + center_x
    #     y_dashed_line = arrow_offset + center_y

    #     ax.axvline(x=x_dashed_line, linestyle='--', color='gray', alpha=0.7, label=f"x={x_dashed_line}")
    #     ax.axhline(y=y_dashed_line, linestyle='--', color='gray', alpha=0.7, label=f"y={y_dashed_line}")

    slab_list = slab.get_chemical_symbols()
    mol_list = adsorbate.get_chemical_symbols()

    # Generate list of molecules
    for i, symm in enumerate(processed_symmetric_set):
        # Extract positional data
        pos = symm['position']
        ori = symm['orientation']

        x, y, z = pos
        alpha, beta, gamma = ori

        # Making adsorption geometry
        symm_ads = create(slab, adsorbate, alpha, beta, gamma, x, y, z)

        _, mol = separate_ads_symbols(symm_ads, slab_list, mol_list)

        # Get the positions
        a1_pos = mol.positions[0]  # 1st atom position
        a2_pos = mol.positions[1]  # 2nd atom position

        # Project onto the appropriate plane for the current view
        a1_proj = (a1_pos[projection_plane[0]] + arrow_offset, a1_pos[projection_plane[1]] + arrow_offset)
        a2_proj = (a2_pos[projection_plane[0]] + arrow_offset, a2_pos[projection_plane[1]] + arrow_offset)

        # Plot an arrow from a1 to a2 in the current view
        ax.annotate(
            '', xy=a2_proj, xytext=a1_proj,
            arrowprops=dict(arrowstyle="->", color='red', lw=2)
        )
        # Annotate the arrow with the index from symm['idx']
        # Adjust the position slightly to place the index label at the midpoint of the arrow
        mid_point = ((a1_proj[0] + a2_proj[0]) / 2, (a1_proj[1] + a2_proj[1]) / 2)
        if annotated:
            ax.text(a1_proj[0], a1_proj[1], f'{symm["idx"]}', color='blue', fontsize=10, ha='center', va='center')

    plt.tight_layout()
    plt.savefig(filename)

def plot_linear_adsorption_energies_set(slab:Atoms, adsorbate:Atoms, processed_symmetric_set:list, 
                                  perspective:str='z', arrow_offset:float=0.775, filename:str = 'ads_arrows_energies_projected.png',
                                  annotated:bool=False):
    # Set up a single plot for the x-axis view
    fig, ax = plt.subplots(figsize=(5, 5))

    # Define rotation view, view_title, and projection planes
    if perspective == 'z':
        view_title = 'View from z-axis, with colormap'
        rotation_view = '0x,0y,0z'
        projection_plane = (0, 1)
        add_axes_lines = True

    elif perspective == 'y':
        view_title = 'View from y-axis, with colormap'
        rotation_view = '-90x,0y,0z'
        projection_plane = (0, 2)
        add_axes_lines = False

    elif perspective == 'x':
        view_title = 'View from x-axis, with colormap'
        rotation_view = '90x,-90y,-180z'
        projection_plane = (1, 2)  
        add_axes_lines = False              

    # Plot slab
    plot_atoms(slab, ax, radii=0.5, rotation=rotation_view)
    ax.set_title(view_title)

    # Add axis labels for the x-axis view
    axis_labels = {'z': ('X-axis', 'Y-axis'),
                'y': ('X-axis', 'Z-axis'),
                'x': ('Y-axis', 'Z-axis')}
    xlabel, ylabel = axis_labels[perspective]
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Add a grid for reference
    ax.grid(True, linestyle='--', alpha=0.5)

    # if add_axes_lines:
    #     center_x = slab.get_cell()[0][projection_plane[0]] / 2
    #     center_y = slab.get_cell()[1][projection_plane[1]] / 2

    #     x_dashed_line = arrow_offset + center_x
    #     y_dashed_line = arrow_offset + center_y

    #     ax.axvline(x=x_dashed_line, linestyle='--', color='gray', alpha=0.7, label=f"x={x_dashed_line}")
    #     ax.axhline(y=y_dashed_line, linestyle='--', color='gray', alpha=0.7, label=f"y={y_dashed_line}")

    e_list = [symm['e_ads'] for symm in processed_symmetric_set]
    slab_list = slab.get_chemical_symbols()
    mol_list = adsorbate.get_chemical_symbols()

    # Normalize e_ads values to map to colors
    norm = mcolors.Normalize(vmin=min(e_list), vmax=max(e_list))
    cmap = plt.cm.cool  # You can choose another colormap if desired
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    # Generate list of molecules
    for i, symm in enumerate(processed_symmetric_set):
        # Extract positional data
        pos = symm['position']
        ori = symm['orientation']
        e_ads = symm['e_ads']

        x, y, z = pos
        alpha, beta, gamma = ori

        # Making adsorption geometry
        symm_ads = create(slab, adsorbate, alpha, beta, gamma, x, y, z)

        _, mol = separate_ads_symbols(symm_ads, slab_list, mol_list)

        # Get the positions
        a1_pos = mol.positions[0]  # 1st atom position
        a2_pos = mol.positions[1]  # 2nd atom position

        # Project onto the appropriate plane for the current view
        a1_proj = (a1_pos[projection_plane[0]] + arrow_offset, a1_pos[projection_plane[1]] + arrow_offset)
        a2_proj = (a2_pos[projection_plane[0]] + arrow_offset, a2_pos[projection_plane[1]] + arrow_offset)

        # Get color for arrow based on e_ads value
        arrow_color = sm.to_rgba(e_ads)

        # Plot an arrow from a1 to a2 in the current view
        ax.annotate(
            '', xy=a2_proj, xytext=a1_proj,
            arrowprops=dict(arrowstyle="->", color=arrow_color, lw=2)
        )
        # Annotate the arrow with the index from symm['idx']
        # Adjust the position slightly to place the index label at the midpoint of the arrow
        mid_point = ((a1_proj[0] + a2_proj[0]) / 2, (a1_proj[1] + a2_proj[1]) / 2)
        if annotated:
            ax.text(a1_proj[0], a1_proj[1], f'{symm["idx"]}', color='blue', fontsize=10, ha='center', va='center')

    # Add a colorbar for reference
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical')
    cbar.set_label('Adsorption Energy (eV)')

    plt.tight_layout()
    plt.savefig(filename)

def plot_linear_adsorption_determinants_set(slab:Atoms, adsorbate:Atoms, processed_symmetric_set:list, 
                                  perspective:str='z', arrow_offset:float=0.775, filename:str = 'ads_arrows_energies_projected.png',
                                  annotated:bool=False):
    # Set up a single plot for the x-axis view
    fig, ax = plt.subplots(figsize=(5, 5))

    # Define rotation view, view_title, and projection planes
    if perspective == 'z':
        view_title = 'View from z-axis, with colormap'
        rotation_view = '0x,0y,0z'
        projection_plane = (0, 1)
        add_axes_lines = True

    elif perspective == 'y':
        view_title = 'View from y-axis, with colormap'
        rotation_view = '-90x,0y,0z'
        projection_plane = (0, 2)
        add_axes_lines = False

    elif perspective == 'x':
        view_title = 'View from x-axis, with colormap'
        rotation_view = '90x,-90y,-180z'
        projection_plane = (1, 2)  
        add_axes_lines = False              

    # Plot slab
    plot_atoms(slab, ax, radii=0.5, rotation=rotation_view)
    ax.set_title(view_title)

    # Add axis labels for the x-axis view
    axis_labels = {'z': ('X-axis', 'Y-axis'),
                'y': ('X-axis', 'Z-axis'),
                'x': ('Y-axis', 'Z-axis')}
    xlabel, ylabel = axis_labels[perspective]
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Add a grid for reference
    ax.grid(True, linestyle='--', alpha=0.5)

    # if add_axes_lines:
    #     center_x = slab.get_cell()[0][projection_plane[0]] / 2
    #     center_y = slab.get_cell()[1][projection_plane[1]] / 2

    #     x_dashed_line = arrow_offset + center_x
    #     y_dashed_line = arrow_offset + center_y

    #     ax.axvline(x=x_dashed_line, linestyle='--', color='gray', alpha=0.7, label=f"x={x_dashed_line}")
    #     ax.axhline(y=y_dashed_line, linestyle='--', color='gray', alpha=0.7, label=f"y={y_dashed_line}")

    slab_list = slab.get_chemical_symbols()
    mol_list = adsorbate.get_chemical_symbols()

    # Get the determinant values
    det_list = [symm['det'] for symm in processed_symmetric_set]

    # Normalize det values to map to colors
    norm = mcolors.Normalize(vmin=min(det_list), vmax=max(det_list))
    cmap = plt.cm.cool  # You can choose another colormap if desired
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    # Generate list of molecules
    for i, symm in enumerate(processed_symmetric_set):
        # Extract positional data
        pos = symm['position']
        ori = symm['orientation']
        det = symm['det']

        x, y, z = pos
        alpha, beta, gamma = ori

        # Making adsorption geometry
        symm_ads = create(slab, adsorbate, alpha, beta, gamma, x, y, z)

        _, mol = separate_ads_symbols(symm_ads, slab_list, mol_list)

        # Get the positions
        a1_pos = mol.positions[0]  # 1st atom position
        a2_pos = mol.positions[1]  # 2nd atom position

        # Project onto the appropriate plane for the current view
        a1_proj = (a1_pos[projection_plane[0]] + arrow_offset, a1_pos[projection_plane[1]] + arrow_offset)
        a2_proj = (a2_pos[projection_plane[0]] + arrow_offset, a2_pos[projection_plane[1]] + arrow_offset)

        # Get color for arrow based on det value
        if det == 1:
            arrow_color = 'red'
        else:
            arrow_color = 'blue'
        # arrow_color = sm.to_rgba(det)

        # Plot an arrow from a1 to a2 in the current view
        ax.annotate(
            '', xy=a2_proj, xytext=a1_proj,
            arrowprops=dict(arrowstyle="->", color=arrow_color, lw=2)
        )
        # Annotate the arrow with the index from symm['idx']
        # # Adjust the position slightly to place the index label at the midpoint of the arrow
        # mid_point = ((a1_proj[0] + a2_proj[0]) / 2, (a1_proj[1] + a2_proj[1]) / 2)
        if annotated:
            ax.text(a1_proj[0], a1_proj[1], f'{symm["idx"]}', color='blue', fontsize=10, ha='center', va='center')

    plt.tight_layout()
    plt.savefig(filename)

def plot_adsorption_triangle_projections(ads:Atoms, slab_symbols:list, mol_symbols:list, idx:list = [0,1,2], 
                                 offset:float=0.75, filename:str = 'ads_triangle_projected.png'):
    # Set up a row of subplots to show x, y, and z views
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Separate slab and molecule
    slab, mol = separate_ads_symbols(ads, slab_symbols, mol_symbols)

    # Define projection planes for each view
    view_titles = ['View from x-axis', 'View from y-axis', 'View from z-axis']
    views = ['90x,-90y,-180z', '-90x,0y,0z', '0x,0y,0z'] 
    projection_planes = [(1, 2), (0, 2), (0, 1)]  # Correctly aligned with the rotation views

    # Plot each view with an arrow representing the CO molecule
    for ax, title, v, (x_idx, y_idx) in zip(axes, view_titles, views, projection_planes):
        ax.set_title(title)

        # Add axis labels based on the view title
        if title == 'View from x-axis':
            ax.set_xlabel('Y-axis')
            ax.set_ylabel('Z-axis')
        elif title == 'View from y-axis':
            ax.set_xlabel('X-axis')  # Correctly label for the y-axis view
            ax.set_ylabel('Z-axis')
        else:  # 'View from z-axis'
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')

        # Add a grid for reference
        ax.grid(True, linestyle='--', alpha=0.5)
        
        ### PLOT THE SLAB ATOMS USING plot_atoms FUNCTION
        plot_atoms(slab, ax, radii=0.5, rotation=v)
        
        # # Get the positions and species of all atoms
        # positions = mol.positions
        # species = mol.get_chemical_symbols()

        # Get the positions and species of specified atoms
        positions = mol.positions[idx]
        species = [mol.get_chemical_symbols()[i] for i in idx]
        
        # Extract the projections of each atom onto the current plane
        projections = [(pos[x_idx]+offset, pos[y_idx]+offset) for pos in positions]
        
        # Unpack the projections into x and y coordinates
        x_coords, y_coords = zip(*projections)
        
        # Plot the triangle connecting the three atoms
        ax.fill(x_coords, y_coords, color='blue', alpha=0.3, edgecolor='black', linewidth=2)
        ax.scatter(x_coords, y_coords, color='red', s=50)  # Mark the atom positions
        
        # Annotate the atomic species
        for (x, y), symbol in zip(projections, species):
            ax.text(x, y, symbol, fontsize=11, color='black', ha='center', va='center',
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='circle,pad=0.3'))

    plt.tight_layout()
    plt.savefig(filename)

def plot_planar_adsorption_set(
        slab: Atoms, adsorbate: Atoms, processed_symmetric_set:list, idx:list = [0,1,2], perspective:str='z',
        offset:float=0.75, filename:str = 'ads_planar_set.png',
        annotated:bool=False
):
    # Set up a single plot for the x-axis view
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # Define rotation view, view_title, and projection planes
    if perspective == 'z':
        view_title = 'View from z-axis, with colormap'
        rotation_view = '0x,0y,0z'
        projection_plane = (0, 1)
        add_axes_lines = True

    elif perspective == 'y':
        view_title = 'View from y-axis, with colormap'
        rotation_view = '-90x,0y,0z'
        projection_plane = (0, 2)
        add_axes_lines = False

    elif perspective == 'x':
        view_title = 'View from x-axis, with colormap'
        rotation_view = '90x,-90y,-180z'
        projection_plane = (1, 2)  
        add_axes_lines = False                    

    # Plot slab
    plot_atoms(slab, ax, radii=0.5, rotation=rotation_view)
    ax.set_title(view_title)

    # Add axis labels for the x-axis view
    axis_labels = {'z': ('X-axis', 'Y-axis'),
                'y': ('X-axis', 'Z-axis'),
                'x': ('Y-axis', 'Z-axis')}
    xlabel, ylabel = axis_labels[perspective]
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Add a grid for reference
    ax.grid(True, linestyle='--', alpha=0.5)

    # if add_axes_lines:
    #     center_x = slab.get_cell()[0][projection_plane[0]] / 2
    #     center_y = slab.get_cell()[1][projection_plane[1]] / 2

    #     x_dashed_line = offset + center_x
    #     y_dashed_line = offset + center_y

    #     ax.axvline(x=x_dashed_line, linestyle='--', color='gray', alpha=0.7, label=f"x={x_dashed_line}")
    #     ax.axhline(y=y_dashed_line, linestyle='--', color='gray', alpha=0.7, label=f"y={y_dashed_line}")

    x_idx = projection_plane[0]
    y_idx = projection_plane[1]

    slab_list = slab.get_chemical_symbols()
    mol_list = adsorbate.get_chemical_symbols()

    # Generate list of molecules
    for i, symm in enumerate(processed_symmetric_set):
        # Extract positional data
        pos = symm['position']
        ori = symm['orientation']

        x, y, z = pos
        alpha, beta, gamma = ori

        # Making adsorption geometry
        symm_ads = create(slab, adsorbate, alpha, beta, gamma, x, y, z)

        _, mol = separate_ads_symbols(symm_ads, slab_list, mol_list)

        # # Get the positions and species of all atoms
        # positions = mol.positions
        # species = mol.get_chemical_symbols()

        # Get the positions and species of specified atoms
        positions = mol.positions[idx]
        species = [mol.get_chemical_symbols()[i] for i in idx]
        
        # Extract the projections of each atom onto the current plane
        projections = [(pos[x_idx]+offset, pos[y_idx]+offset) for pos in positions]
        
        # Unpack the projections into x and y coordinates
        x_coords, y_coords = zip(*projections)

        # Compute centroid
        centroid_x = sum(x_coords) / len(x_coords)
        centroid_y = sum(y_coords) / len(y_coords)

        # Plot the triangle connecting the three atoms
        ax.fill(x_coords, y_coords, color='blue', alpha=0.3, edgecolor='black', linewidth=2)
        ax.scatter(x_coords, y_coords, color='red', s=50)  # Mark the atom positions

        if annotated:

            # Annotate the center of the triangle with the index
            ax.text(centroid_x, centroid_y, f'{symm["idx"]}', color='blue', fontsize=10, ha='center', va='center')
        
        # Annotate the atomic species
        for (x, y), symbol in zip(projections, species):
            ax.text(x, y, symbol, fontsize=10, color='black', ha='center', va='center',
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.1'))

    plt.tight_layout()
    plt.savefig(filename)

def plot_planar_adsorption_energies_set(
        slab: Atoms, adsorbate: Atoms, processed_symmetric_set:list, idx:list = [0,1,2],
        perspective:str='z', offset:float=0.75, filename:str = 'ads_planar_set.png',
        annotated:bool=False
):

    # Set up a single plot for the x-axis view
    fig, ax = plt.subplots(figsize=(5, 5))

    # Define rotation view, view_title, and projection planes
    if perspective == 'z':
        view_title = 'View from z-axis, with colormap'
        rotation_view = '0x,0y,0z'
        projection_plane = (0, 1)
        add_axes_lines = True

    elif perspective == 'y':
        view_title = 'View from y-axis, with colormap'
        rotation_view = '-90x,0y,0z'
        projection_plane = (0, 2)
        add_axes_lines = False

    elif perspective == 'x':
        view_title = 'View from x-axis, with colormap'
        rotation_view = '90x,-90y,-180z'
        projection_plane = (1, 2)  
        add_axes_lines = False                    

    # Plot slab
    plot_atoms(slab, ax, radii=0.5, rotation=rotation_view)
    ax.set_title(view_title)

    # Add axis labels for the x-axis view
    axis_labels = {'z': ('X-axis', 'Y-axis'),
                'y': ('X-axis', 'Z-axis'),
                'x': ('Y-axis', 'Z-axis')}
    xlabel, ylabel = axis_labels[perspective]
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    x_idx = projection_plane[0]
    y_idx = projection_plane[1]

    # Add a grid for reference
    ax.grid(True, linestyle='--', alpha=0.5)

    # if add_axes_lines:
    #     center_x = slab.get_cell()[0][projection_plane[0]] / 2
    #     center_y = slab.get_cell()[1][projection_plane[1]] / 2

    #     x_dashed_line = offset + center_x
    #     y_dashed_line = offset + center_y

    #     ax.axvline(x=x_dashed_line, linestyle='--', color='gray', alpha=0.7, label=f"x={x_dashed_line}")
    #     ax.axhline(y=y_dashed_line, linestyle='--', color='gray', alpha=0.7, label=f"y={y_dashed_line}")

    x_idx = projection_plane[0]
    y_idx = projection_plane[1]

    e_list = [symm['e_ads'] for symm in processed_symmetric_set]
    slab_list = slab.get_chemical_symbols()
    mol_list = adsorbate.get_chemical_symbols()

    # Normalize e_ads values to map to colors
    norm = mcolors.Normalize(vmin=min(e_list), vmax=max(e_list))
    cmap = plt.cm.cool  # You can choose another colormap if desired
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    # Generate list of molecules
    for i, symm in enumerate(processed_symmetric_set):
        # Extract positional data
        pos = symm['position']
        ori = symm['orientation']
        e_ads = symm['e_ads']

        x, y, z = pos
        alpha, beta, gamma = ori

        # Making adsorption geometry
        symm_ads = create(slab, adsorbate, alpha, beta, gamma, x, y, z)

        _, mol = separate_ads_symbols(symm_ads, slab_list, mol_list)

        # Get the positions and species of specified atoms
        positions = mol.positions[idx]
        species = [mol.get_chemical_symbols()[i] for i in idx]
        
        # Extract the projections of each atom onto the current plane
        projections = [(pos[x_idx]+offset, pos[y_idx]+offset) for pos in positions]
        
        # Unpack the projections into x and y coordinates
        x_coords, y_coords = zip(*projections)

        # Determine color
        fill_color = sm.to_rgba(e_ads)
        # Compute centroid
        centroid_x = sum(x_coords) / len(x_coords)
        centroid_y = sum(y_coords) / len(y_coords)

        # Plot the triangle connecting the three atoms
        ax.fill(x_coords, y_coords, color=fill_color, alpha=0.3, edgecolor='black', linewidth=2)
        ax.scatter(x_coords, y_coords, color='red', s=50)  # Mark the atom positions

        if annotated:
            # Annotate the center of the triangle with the index
            ax.text(centroid_x, centroid_y, f'{symm["idx"]}', color='blue', fontsize=10, ha='center', va='center')
        
        # Annotate the atomic species
        for (x, y), symbol in zip(projections, species):
            ax.text(x, y, symbol, fontsize=10, color='black', ha='center', va='center',
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.1'))
            
    # Add a colorbar for reference
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical')
    cbar.set_label('Adsorption Energy (eV)')

    plt.tight_layout()
    plt.savefig(filename)

def plot_planar_adsorption_determinants_set(
        slab: Atoms, adsorbate: Atoms, processed_symmetric_set:list, idx:list = [0,1,2],
        perspective:str='z', offset:float=0.75, filename:str = 'ads_planar_set.png',
        annotated:bool=False
):

    # Set up a single plot for the x-axis view
    fig, ax = plt.subplots(figsize=(5, 5))

    # Define rotation view, view_title, and projection planes
    if perspective == 'z':
        view_title = 'View from z-axis, with colormap'
        rotation_view = '0x,0y,0z'
        projection_plane = (0, 1)
        add_axes_lines = True

    elif perspective == 'y':
        view_title = 'View from y-axis, with colormap'
        rotation_view = '-90x,0y,0z'
        projection_plane = (0, 2)
        add_axes_lines = False

    elif perspective == 'x':
        view_title = 'View from x-axis, with colormap'
        rotation_view = '90x,-90y,-180z'
        projection_plane = (1, 2)  
        add_axes_lines = False                    

    # Plot slab
    plot_atoms(slab, ax, radii=0.5, rotation=rotation_view)
    ax.set_title(view_title)

    # Add axis labels for the x-axis view
    axis_labels = {'z': ('X-axis', 'Y-axis'),
                'y': ('X-axis', 'Z-axis'),
                'x': ('Y-axis', 'Z-axis')}
    xlabel, ylabel = axis_labels[perspective]
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    x_idx = projection_plane[0]
    y_idx = projection_plane[1]

    # Add a grid for reference
    ax.grid(True, linestyle='--', alpha=0.5)

    # if add_axes_lines:
    #     center_x = slab.get_cell()[0][projection_plane[0]] / 2
    #     center_y = slab.get_cell()[1][projection_plane[1]] / 2

    #     x_dashed_line = offset + center_x
    #     y_dashed_line = offset + center_y

    #     ax.axvline(x=x_dashed_line, linestyle='--', color='gray', alpha=0.7, label=f"x={x_dashed_line}")
    #     ax.axhline(y=y_dashed_line, linestyle='--', color='gray', alpha=0.7, label=f"y={y_dashed_line}")

    x_idx = projection_plane[0]
    y_idx = projection_plane[1]

    slab_list = slab.get_chemical_symbols()
    mol_list = adsorbate.get_chemical_symbols()

    det_list = [symm['det'] for symm in processed_symmetric_set]
    norm = mcolors.Normalize(vmin=min(det_list), vmax=max(det_list))
    cmap = plt.cm.cool  # You can choose another colormap if desired
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    # Generate list of molecules
    for i, symm in enumerate(processed_symmetric_set):
        pos = symm['position']
        ori = symm['orientation']
        det = symm['det']

        x, y, z = pos
        alpha, beta, gamma = ori

        symm_ads = create(slab, adsorbate, alpha, beta, gamma, x, y, z)
        _, mol = separate_ads_symbols(symm_ads, slab_list, mol_list)

        positions = mol.positions[idx]
        species = [mol.get_chemical_symbols()[i] for i in idx]
        projections = [(pos[x_idx] + offset, pos[y_idx] + offset) for pos in positions]
        x_coords, y_coords = zip(*projections)

        fill_color = sm.to_rgba(det)
        centroid_x = sum(x_coords) / len(x_coords)
        centroid_y = sum(y_coords) / len(y_coords)

        ax.fill(x_coords, y_coords, color=fill_color, alpha=0.3, edgecolor='black', linewidth=2)
        ax.scatter(x_coords, y_coords, color='red', s=50)

        if annotated:
            ax.text(centroid_x, centroid_y, f'{symm["idx"]}', color='blue', fontsize=10, ha='center', va='center')

        for (x, y), symbol in zip(projections, species):
            ax.text(x, y, symbol, fontsize=10, color='black', ha='center', va='center',
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.1'))

    plt.tight_layout()
    plt.savefig(filename)

def plot_adsorption_vectors_projections(ads: Atoms, slab_symbols: list, mol_symbols: list, idx: list = [0, 1, 2], 
                                        x_offset: float = 0.75, y_offset: float = 0.75, with_atoms: bool = False, 
                                        annotate: bool = False, filename: str = 'ads_triangle_projected.png'):
    # Set up a row of subplots to show x, y, and z views
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Separate slab and molecule
    slab, mol = separate_ads_symbols(ads, slab_symbols, mol_symbols)

    # Define projection planes for each view
    view_titles = ['View from x-axis', 'View from y-axis', 'View from z-axis']
    views = ['90x,-90y,-180z', '-90x,0y,0z', '0x,0y,0z']
    projection_planes = [(1, 2), (0, 2), (0, 1)]  # Correctly aligned with the rotation views

    # Plot each view with an arrow representing the CO molecule
    for ax, title, v, (x_idx, y_idx) in zip(axes, view_titles, views, projection_planes):
        ax.set_title(title)

        # Add axis labels based on the view title
        if title == 'View from x-axis':
            ax.set_xlabel('Y-axis')
            ax.set_ylabel('Z-axis')
        elif title == 'View from y-axis':
            ax.set_xlabel('X-axis')  # Correctly label for the y-axis view
            ax.set_ylabel('Z-axis')
        else:  # 'View from z-axis'
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')

        # Add a grid for reference
        ax.grid(True, linestyle='--', alpha=0.5)

        if with_atoms:
            # Plot the ads atoms using plot_atoms function
            plot_atoms(ads, ax, radii=0.5, rotation=v)
        else:
            # Plot the slab atoms using plot_atoms function
            plot_atoms(slab, ax, radii=0.5, rotation=v)
        
        # Get the positions and species of specified atoms
        positions = mol.positions[idx]

        pos_0 = positions[0]
        
        vector = triangle_vector(mol, idx)  # Normal vector to the plane
        pos_1 = pos_0 + vector
        pos_2 = positions[1] + (positions[2] - positions[1]) / 2  # Midpoint between atoms idx[1] and idx[2]

        # ### Adjustments here for ammonia
        # v1, v2 = ammonia_vectors(mol)  # Normal vector to the plane
        # pos_1 = pos_0 + v1
        # pos_2 = pos_0 + v2

        # Project positions to the current plane
        p0_proj = pos_0[[x_idx, y_idx]] + [x_offset, y_offset]

        # Calculate the vector from pos_0 to pos_2, and then scale the vector from pos_0 to pos_1
        vector_length = np.linalg.norm(pos_2 - pos_0)  # Length of the vector from pos_0 to pos_2
        vector_direction = (pos_1 - pos_0) / np.linalg.norm(pos_1 - pos_0)  # Normalized direction of the vector from pos_0 to pos_1

        # Scale pos_1 to match the length of pos_2 - pos_0
        pos_1_scaled = pos_0 + vector_length * vector_direction

        # Apply offsets to the projections
        p1_proj = pos_1_scaled[[x_idx, y_idx]] + [x_offset, y_offset]
        p2_proj = pos_2[[x_idx, y_idx]] + [x_offset, y_offset]

        # p1_proj = pos_1[[x_idx, y_idx]] + [x_offset, y_offset]
        # p2_proj = pos_2[[x_idx, y_idx]] + [x_offset, y_offset]        

        # Plot the origin point and vectors
        ax.quiver(*p0_proj, *(p1_proj - p0_proj), angles='xy', scale_units='xy', scale=1, color='cyan', label='Vector 1', zorder=3)
        ax.quiver(*p0_proj, *(p2_proj - p0_proj), angles='xy', scale_units='xy', scale=1, color='magenta', label='Vector 2', zorder=3)

        # Annotate positions
        if annotate:
            ax.text(*(p1_proj + [x_offset, y_offset]), "Vec1", color='cyan', fontsize=9)
            ax.text(*(p2_proj + [x_offset, y_offset]), "Vec2", color='magenta', fontsize=9)

    plt.tight_layout()
    plt.savefig(filename)

def plot_adsorption_vectors_set(
        slab: Atoms, adsorbate: Atoms, processed_symmetric_set: list, idx: list = [0, 1, 2], perspective: str = 'z',
        x_offset: float = 0.75, y_offset: float = 0.75, annotate: bool = False, filename: str = 'ads_planar_set.png',
):
    # Set up a single plot for the x-axis view
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # Define rotation view, view_title, and projection planes
    if perspective == 'z':
        view_title = 'View from z-axis, with colormap'
        rotation_view = '0x,0y,0z'
        projection_plane = (0, 1)
        add_axes_lines = True

    elif perspective == 'y':
        view_title = 'View from y-axis, with colormap'
        rotation_view = '-90x,0y,0z'
        projection_plane = (0, 2)
        add_axes_lines = False

    elif perspective == 'x':
        view_title = 'View from x-axis, with colormap'
        rotation_view = '90x,-90y,-180z'
        projection_plane = (1, 2)  
        add_axes_lines = False                    

    # Plot slab
    plot_atoms(slab, ax, radii=0.5, rotation=rotation_view)
    ax.set_title(view_title)

    # Add axis labels for the x-axis view
    axis_labels = {'z': ('X-axis', 'Y-axis'),
                'y': ('X-axis', 'Z-axis'),
                'x': ('Y-axis', 'Z-axis')}
    xlabel, ylabel = axis_labels[perspective]
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Add a grid for reference
    ax.grid(True, linestyle='--', alpha=0.5)

    if add_axes_lines:
        center_x = slab.get_cell()[0][projection_plane[0]] / 2
        center_y = slab.get_cell()[1][projection_plane[1]] / 2

        x_dashed_line = x_offset + center_x
        y_dashed_line = y_offset + center_y

        ax.axvline(x=x_dashed_line, linestyle='--', color='gray', alpha=0.7, label=f"x={x_dashed_line}")
        ax.axhline(y=y_dashed_line, linestyle='--', color='gray', alpha=0.7, label=f"y={y_dashed_line}")

    x_idx = projection_plane[0]
    y_idx = projection_plane[1]

    slab_list = slab.get_chemical_symbols()
    mol_list = adsorbate.get_chemical_symbols()

    # Generate list of molecules
    for i, symm in enumerate(processed_symmetric_set):
        # Extract positional data
        pos = symm['position']
        ori = symm['orientation']
        det = symm['det']

        x, y, z = pos
        alpha, beta, gamma = ori

        # Making adsorption geometry
        symm_ads = create(slab, adsorbate, alpha, beta, gamma, x, y, z)

        _, mol = separate_ads_symbols(symm_ads, slab_list, mol_list)

        # Get the positions and species of specified atoms
        positions = mol.positions[idx]

        # Adding details
        pos_0 = positions[0]
        
        vector = triangle_vector(mol, idx)  # Normal vector to the plane
        pos_1 = pos_0 + vector
        pos_2 = positions[1] + (positions[2] - positions[1]) / 2  # Midpoint between atoms idx[1] and idx[2]

        # ### Adjustments here for ammonia
        # v1, v2 = ammonia_vectors(mol)  # Normal vector to the plane
        # pos_1 = pos_0 + v1
        # pos_2 = pos_0 + v2

        # Apply offsets to projections
        p0_proj = pos_0[[x_idx, y_idx]] + [x_offset, y_offset]

        # Calculate the vector from pos_0 to pos_2, and then scale the vector from pos_0 to pos_1
        vector_length = np.linalg.norm(pos_2 - pos_0)  # Length of the vector from pos_0 to pos_2
        vector_direction = (pos_1 - pos_0) / np.linalg.norm(pos_1 - pos_0)  # Normalized direction of the vector from pos_0 to pos_1
        
        # Scale pos_1 to match the length of pos_2 - pos_0
        pos_1_scaled = pos_0 + vector_length * vector_direction
        
        p1_proj = pos_1_scaled[[x_idx, y_idx]] + [x_offset, y_offset]
        p2_proj = pos_2[[x_idx, y_idx]] + [x_offset, y_offset]

        # p1_proj = pos_1[[x_idx, y_idx]] + [x_offset, y_offset]
        # p2_proj = pos_2[[x_idx, y_idx]] + [x_offset, y_offset]

        if math.isclose(det, -1):
            c1 = 'red'
            c2 = 'blue'
        else:
            c1 = 'cyan'
            c2 = 'magenta'

        # Plot the origin point and vectors
        ax.quiver(*p0_proj, *(p1_proj - p0_proj), angles='xy', scale_units='xy', scale=1, color=c1, label='Vector 1', zorder=3)
        ax.quiver(*p0_proj, *(p2_proj - p0_proj), angles='xy', scale_units='xy', scale=1, color=c2, label='Vector 2', zorder=3)

        # Annotate positions
        if annotate:
            ax.text(*(p1_proj + [x_offset, y_offset]), "Vec1", color=c1, fontsize=9)
            ax.text(*(p2_proj + [x_offset, y_offset]), "Vec2", color=c2, fontsize=9)

    plt.tight_layout()
    plt.savefig(filename)