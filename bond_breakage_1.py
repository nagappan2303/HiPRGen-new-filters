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
    """Return set of bonds as (i,j) pairs with i<j, shifted by offset."""
    mol = Chem.MolFromSmiles(smiles)
    bonds = set()
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        # shift indices by offset, also shift to 1-based
        i = i + 1 + offset
        j = j + 1 + offset
        bonds.add(tuple(sorted((i, j))))  # ensure (i,j) == (j,i)
    return bonds, mol.GetNumAtoms()

# --------------------------
# Bond change analysis
# --------------------------
if maps3:
    #print(maps3)
    best_mapping = dict(maps3[0])  # take first mapping, ensure dict
    #print(best_mapping)
    reactant_bonds = set()
    product_bonds = set()

    # continuous numbering across all reactants
    offset = 0
    for r in reactants:
        bonds, n_atoms = get_bonds(r, offset)
        reactant_bonds |= bonds
        offset += n_atoms

    # continuous numbering across all products
    offset = 0
    for p in products:
        bonds, n_atoms = get_bonds(p, offset)
        product_bonds |= bonds
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
            mapped_product_bonds.add(tuple(sorted((ri, rj))))

    # find broken and formed bonds
    broken = reactant_bonds - mapped_product_bonds
    formed = mapped_product_bonds - reactant_bonds

    print("-------------------------------------------------------------")
    print("\nBond Changes:")
    print("  Broken bonds:", len(broken), broken)
    print("  Formed bonds:", len(formed), formed)
    print("-------------------------------------------------------------")
