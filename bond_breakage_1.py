import MapIsing as mi
import networkx as nx
import optim_wrapper as ow
from rdkit import Chem

# --------------------------
# Reaction setup
# --------------------------
reaction_str = 'C(=O)(OC[CH2])O[Na].C(=O)(OC[CH2])O[Na]>>C=C.O(C(=O)OCCOC(=O)O[Na])[Na]'
reactants_str, products_str = reaction_str.split(">>")
reactants = reactants_str.split(".")
products = products_str.split(".")

print("Reactants:")
for r in reactants:
    print("   ", r)

print("-------------------------------------------------------------")
print("Products:")
for p in products:
    print("   ", p)
print("-------------------------------------------------------------")
print("target reaction:", reaction_str)

# --------------------------
# Run atom mapping
# --------------------------
mp = mi.Mapping(reaction_str)
nodes, edges = mp.modular_product()
graph = nx.Graph(edges)

max_cliques_finder = ow.MaxCliques(graph)
cliques, run_time = max_cliques_finder.find_maximum_cliques_sa()
print("\nRuntime:", round(run_time, 4), "seconds\n")

maps1 = mp.cliques_to_mappings(nodes, cliques)
maps2 = mp.non_equivalent(maps1)
maps3 = mp.filtering(maps2, 'filter1')

print("Atom Mappings (Reactant -> Product):")
for i, mapping in enumerate(maps3):
    print(f"\nMapping {i+1}:")
    if isinstance(mapping, dict):
        for k, v in mapping.items():
            print(f"  {k} -> {v}")
    elif isinstance(mapping, list):
        for pair in mapping:
            if isinstance(pair, tuple) and len(pair) == 2:
                print(f"  {pair[0]} -> {pair[1]}")
            else:
                print("  ", pair)

# optionally show/export the first mapping
if maps3:
    mp.export_mapping(maps3[0])
    mp.show_mapping(maps3[0])

# --------------------------
# Bond extraction helper
# --------------------------
def get_bonds(smiles, offset=0):
    """
    Return set of bonds as (i,j) pairs with i<j, shifted by offset.
    Includes atom symbols for easier filtering of O-Na bonds.
    """
    mol = Chem.MolFromSmiles(smiles)
    bonds = set()
    atom_symbols = {i + 1 + offset: atom.GetSymbol() for i, atom in enumerate(mol.GetAtoms())}
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        i = i + 1 + offset
        j = j + 1 + offset
        bonds.add((min(i, j), max(i, j)))
    return bonds, atom_symbols, mol.GetNumAtoms()

# --------------------------
# Bond change analysis
# --------------------------
if maps3:
    best_mapping = dict(maps3[0])  # take first mapping
    reactant_bonds = set()
    product_bonds = set()
    atom_symbols_react = {}
    atom_symbols_prod = {}

    # continuous numbering across all reactants
    offset = 0
    for r in reactants:
        bonds, symbols, n_atoms = get_bonds(r, offset)
        reactant_bonds |= bonds
        atom_symbols_react.update(symbols)
        offset += n_atoms

    # continuous numbering across all products
    offset = 0
    for p in products:
        bonds, symbols, n_atoms = get_bonds(p, offset)
        product_bonds |= bonds
        atom_symbols_prod.update(symbols)
        offset += n_atoms

    print("\nOriginal Reactant bonds:", reactant_bonds)
    print("Original Product bonds:", product_bonds)
    print("Best mapping:", best_mapping)

    # map product bonds back to reactant indexing
    mapped_product_bonds = set()
    for i, j in product_bonds:
        ri = next((r for r, pr in best_mapping.items() if pr == i), None)
        rj = next((r for r, pr in best_mapping.items() if pr == j), None)
        if ri is not None and rj is not None:
            mapped_product_bonds.add((min(ri, rj), max(ri, rj)))

    # ---- Ignore O-Na coordinate bonds ----
    def is_coordinate_bond(i, j, atom_symbols):
        return (
            (atom_symbols.get(i) == 'O' and atom_symbols.get(j) == 'Na') or
            (atom_symbols.get(i) == 'Na' and atom_symbols.get(j) == 'O')
        )

    reactant_bonds = {b for b in reactant_bonds if not is_coordinate_bond(b[0], b[1], atom_symbols_react)}
    mapped_product_bonds = {b for b in mapped_product_bonds if not is_coordinate_bond(b[0], b[1], atom_symbols_react)}

    # find broken and formed bonds
    broken = reactant_bonds - mapped_product_bonds
    formed = mapped_product_bonds - reactant_bonds

    print("-------------------------------------------------------------")
    print("\nBond Changes:")
    print("  Broken bonds:", len(broken), broken)
    print("  Formed bonds:", len(formed), formed)

    # ---- Check for common reaction center for break-1-form-1 ----
    common_center = False
    if len(broken) == 1 and len(formed) == 1:
        b_atoms = list(broken)[0]
        f_atoms = list(formed)[0]
        if set(b_atoms) & set(f_atoms):
            common_center = True

    print("\nReaction Center Check (break-1-form-1):", common_center)
    print("-------------------------------------------------------------")
