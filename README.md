# How to use

## 1. Generation of graph data of modular product from Reaction SMILES

Firstly, generating graph data of modular products from reaction SMILES is required.

MapIsing module controls molecular graphs and their mapping.

The constructor of the Mapping class generates an object from reaction SMILES.

In this example, reaction SMILES 'CC(C)CC(=O)O.OCCC>>CC(C)CC(=O)OCCC.O' is used.

Mapping object has a modular_product() function to generate the graph structure.

Argument edges mean an adjacent list of the modular product, which can be converted to a NetworkX graph structure.

```python
import MapIsing as mi
import networkx as nx

mp = mi.Mapping( 'CC(C)CC(=O)O.OCCC>>CC(C)CC(=O)OCCC.O' )
nodes, edges = mp.modular_product()
graph = nx.Graph( edges )
```

## 2. Enumerating all maximum cliques from the modular product

optim_wrapper module helps to execute Ising computing.

The constructor of MaxCliques requires a NetworkX graph structure.

This class has two solvers for enumerating all the maximum cliques.

One is the find_maximum_cliques_cp() function based on the CP algorithm, a conventional exact algorithm.

Another is the find_maximum_cliques_sa() function based on SA (simulated annealing), Ising computing.

Both functions return two values: a list of enumerated maximum cliques and a run time (unit is seconds) for the enumeration of all cliques.

cliques are required to generate concrete mappings after that.

```
import optim_wrapper as ow

max_cliques_finder = MaxCliques( graph )
max_cliques_finder.find_maximum_cliques_cp()
max_cliques_finder.find_maximum_cliques_sa()

cliques, run_time = max_cliques_finder.find_maximum_cliques_sa()
print( round( run_time, 4 ), 'seconds for running' )
```

## 3. Generating complete mappings from enumerated cliques

The mapping class has a cliques_to_mappings() function to convert from a list of cliques to a list of mappings.

This function needs two arguments (nodes, edges) of modular product generated from the modular_product() function.

Resulted maps1 includes many equivalent mappings.

non_equivealnt() function can remove all equivalent mappings from inputting a list of mappings.

In addition, the filtering() function can apply mappings for two filters (second argument, 'filter1' and 'filter2').

To export each mapping, You can use export_mapping() function.

To visualize each mapping with the graphical scheme of the chemical reaction, You can use the show_mapping() function.

```
maps1 = mp.cliques_to_mappings( nodes, cliques )
maps2 = mp.non_equivalent( maps1 )
maps3 = mp.filtering( maps2, 'filter1' )

res = maps3[0]

mp.export_mapping( res )
mp.show_mapping( res )
```
