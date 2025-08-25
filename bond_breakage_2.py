from collections import Counter
import numpy as np
import pandas as pd
import py3Dmol
import copy

from monty.serialization import loadfn, dumpfn
from networkx.generators.ego import ego_graph

from pymatgen.core.periodic_table import Element
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.core.structure import Molecule
from pymatgen.io.xyz import XYZ
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.local_env import OpenBabelNN

from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash

dataset_path_ec = "NaPF6_EC_june6.json"
d1 = loadfn(dataset_path_ec)

reactants = ["MolID-138417"]
products = ["MolID-137317","MolID-138423"]
#Function to visualize the molecules along with partial charges on each atom
def visualize_molecule2(xyz_data, molecule, partial_charges):
    viewer = py3Dmol.view(width=800, height=400)
    viewer.addModel(xyz_data, "xyz")

    # Add partial charge labels to each atom
    for idx, atom in enumerate(molecule):
        charge = partial_charges[idx]
        position = atom.coords
        viewer.addLabel(f"{charge:.5f}",  # format to 3 decimal places
                        {'position': {'x': position[0], 'y': position[1], 'z': position[2]},
                         'fontSize': 10,
                         'backgroundColor': 'black',
                         'borderColor': 'black'})
    
    viewer.setStyle({
        "stick": {},
        "sphere": {"scale": 0.3}
    })
    #bad_bonds=[(0,6)]
    #viewer.removeBonds([{'atom1': i, 'atom2': j} for i, j in bad_bonds])
    viewer.zoomTo()
    
    
    return viewer.show()


def add_star_hashes(mol_graph,molecule):
    """
    Adds star_hashes to a MoleculeGraph-like object by computing the 
    Weisfeiler-Lehman hash of each atom's 1-hop neighborhood,
    if the atom index is not in mol_graph.m_inds.
    
    Parameters:
        mol_graph: MoleculeGraph object with added fields:
            - mol_graph.m_inds: set of indices to skip (must be added manually)
            - mol_graph.star_hashes: dict to store star hashes (must be added manually)
            - mol_graph.covalent_graph: alias to mol_graph.graph (must be added manually)
    
    Returns:
        None (modifies mol_graph.star_hashes in place)
    """
    metals = frozenset(["Li", "K", "Mg", "Ca", "Zn", "Al"])
    mol_graph.covalent_graph = copy.deepcopy(mol_graph.graph)
    #for i, x in enumerate(molecule.species):
    #    print(x)
    #    if x=="Na":
    #        print(i,x)
    mol_graph.m_inds = [i for i, x in enumerate(molecule.species) if str(x) in metals]
    mol_graph.star_hashes = {}
    #print(mol_graph.m_inds)
    for i in range(mol_graph.molecule.num_sites):
        #print(mol_graph)
        #print(i)
        if i in mol_graph.m_inds:
            print("In m_inds")
        if i not in mol_graph.m_inds:
            neighborhood = ego_graph(
                mol_graph.covalent_graph,
                i,
                radius=1,
                undirected=True
            ).to_undirected()
            #print(neighborhood)
            mol_graph.star_hashes[i] = weisfeiler_lehman_graph_hash(
                neighborhood,
                node_attr='specie',
                iterations=3
            )


stars_reactants = []
stars_products = []
bonds_reactants = []
bonds_products = []
    
for m in d1:
    
    if m['molecule_id'] in reactants :
        print("Reactant")
        print("----------------------------------------------------------------------------------------")
        
        partial_charges = m['partial_charges']['nbo']
        species = m['species']
        coords = m['xyz']
        mol = Molecule(species, coords)
        xyz_data = mol.to(fmt='xyz')
        visualize_molecule2(xyz_data, mol, partial_charges)
        
        add_star_hashes(m['molecule_graph'],m['molecule'])
        for i, star in m['molecule_graph'].star_hashes.items():
            #atom_type = m['molecule_graph'].molecule[i].specie.symbol  # adjust based on your data structure
            #print(f"Atom {i} ({atom_type}): {star}")
            stars_reactants.append(star)
            stars_reactants.append(star)
            
        for u, v in m['molecule_graph'].graph.edges():
            a1 = m['molecule_graph'].molecule[u].specie.symbol
            a2 = m['molecule_graph'].molecule[v].specie.symbol
            bond = "-".join(sorted([a1, a2]))
            bonds_reactants.append(bond)
            bonds_reactants.append(bond)
        
        print(stars_reactants)
        print("\n\n")

    if m['molecule_id'] in products:
        print("Product")
        print("----------------------------------------------------------------------------------------")
        
        partial_charges = m['partial_charges']['nbo']
        species = m['species']
        coords = m['xyz']
        mol = Molecule(species, coords)
        xyz_data = mol.to(fmt='xyz')
        visualize_molecule2(xyz_data, mol, partial_charges)
        
        add_star_hashes(m['molecule_graph'],m['molecule'])
        for i, star in m['molecule_graph'].star_hashes.items():
            #atom_type = m['molecule_graph'].molecule[i].specie.symbol  # adjust based on your data structure
            #print(f"Atom {i} ({atom_type}): {star}")
            stars_products.append(star)
        
        for u, v in m['molecule_graph'].graph.edges():
            a1 = m['molecule_graph'].molecule[u].specie.symbol
            a2 = m['molecule_graph'].molecule[v].specie.symbol
            bond = "-".join(sorted([a1, a2]))
            bonds_products.append(bond)
        
        print(stars_products)
        
        print("\n\n")

r_counter = Counter(stars_reactants)
p_counter = Counter(stars_products)

diff_counter = (r_counter - p_counter) + (p_counter - r_counter)
total_diff = sum(diff_counter.values())

print("=================================================")
print(f"Total number of differing stars = {total_diff}")
print("=================================================")
print("Detail:", diff_counter)

print("\n")

r_bonds_counter = Counter(bonds_reactants)
p_bonds_counter = Counter(bonds_products)

print("Bonds in reactants:", r_bonds_counter)
print("Bonds in products :", p_bonds_counter)

bond_diff = (r_bonds_counter - p_bonds_counter) + (p_bonds_counter - r_bonds_counter)
total_bond_diff = sum(bond_diff.values())

print("=================================================")
print(f"Total number of differing bonds = {total_bond_diff}")
print("=================================================")
print("Bond changes detail:", bond_diff)
