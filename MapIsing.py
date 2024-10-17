import time
from itertools import combinations
from CGRtools import smiles, ReactionContainer

class MolGraph:
    
    # atom information = atom index (not limited to starting 1), atomic number, number of substituted hydrogens
    # bond information = bond index, pair of atom index, bond order (limited to integer)
    
    def __init__( self, strn ): # CGRtools dependency
        
        t = time.time()

        mol = smiles( strn )
        mol.kekule()
        mol_Hs = deepcopy( mol )
        mol_Hs.explicify_hydrogens()

        self.MOL = mol
        self.MOL.clean2d()

        an_Hs = { idx:at.atomic_number for idx,at in mol_Hs.atoms() }
        Hs = { k:sum( [ 1 for i,j,_ in mol_Hs.bonds() if (an_Hs[i]==1 or an_Hs[j]==1) and (i==k or j==k)] ) for k,_ in mol_Hs.atoms() }
        
        atom_info = [ (x,y.atomic_number,Hs[x]) for x,y in mol.atoms() ]
        bond_info = []
        for i, (x, y, z) in enumerate( mol.bonds() ):
            if x > y:
                x, y = y, x
            bond_info.append((i, (x, y), z.order))
        
        # atom index (start from 1 to match the illustration of index)
        self.atoms   = [ idx for idx,_,_ in atom_info ]
        self.atoms.sort() # important for use SMARTS
        self.an      = { idx:an for idx,an,_ in atom_info }
        
        # bond index (start form 0, list, fixed to pair (smaller atom idx, larger atom idx) )
        # bn_an = pair of two atomic numbers
        # order = integer bond multiplicity
        # bn_kind = unique bond label (sorted pair of bn_an)
        self.bonds = [ pair for _,pair,_ in bond_info ]
        self.bn_an = [ (self.an[x],self.an[y]) for x,y in self.bonds ]
        self.order = [ z for _,_,z in bond_info ]
        self.bn_kind = [ (x,y) for x,y in self.bn_an if x <= y ] + [ (y,x) for x,y in self.bn_an if x > y ]
        
        # number of substituted hydrogens
        self.dic_hyd = { idx:Hs for idx, _, Hs in atom_info }
        
        # dictionary of neighbourhood of atom
        self.dic_neib = dict()
        for n in self.atoms:
            n_neib = []
            for x, y in self.bonds:
                if n==x:
                    n_neib.append( y )
                    continue
                if n==y:
                    n_neib.append( x )
            
            self.dic_neib[n] = n_neib
        
        # dictionary of integer bond multiplicity
        self.dic_order = dict()
        for x, y in combinations( self.atoms, 2 ):
            if x > y:
                x, y = y, x
            if not ( x, y ) in self.bonds:
                self.dic_order[(x,y)] = 0
                self.dic_order[(y,x)] = 0
            else:
                idx = self.bonds.index( ( x, y ) )
                self.dic_order[(x,y)] = self.order[idx]
                self.dic_order[(y,x)] = self.order[idx]
        
    #--------------------------------------------------------------------------------------------------------------
    # reference of bond
    # Arguments: atom index, atom index, 
    # Return: None or bn_kind of a bon
    def refer_bond( self, n, m ):
        if n > m:
            n, m = m, n
        
        if (n, m) in self.bonds:
            b_num = self.bonds.index((n, m))
            return self.bn_kind[b_num]
        
        return None
    
    # reference of integer bond multiplicity
    # Arguments: atom index, atom index, 
    # Return: integer (as bond multiplicity)
    def refer_bond_order( self, n, m ):
        if n > m:
            n, m = m, n
        
        if (n,m) in self.bonds:
            ind = self.bonds.index((n, m))
            return self.order[ind]
        else:
            return 0
    
    # searching common atom between two bonds
    # Arguments: bond index, bond index
    # Return: atomic number or None
    def bond_common_atom( self, n, m ):
        n1, n2 = self.bonds[n]
        m1, m2 = self.bonds[m]
        if n1==m1 or n1==m2:
            return self.an[n1]
        if n2==m1 or n2==m2:
            return self.an[n2]
        return None
    
    # searching conneted three atoms between two bonds
    # Arguments: bond index, bond index
    # Return: tuple of three atom index or None
    def bond_mapping( self, n, m ):
        n1, n2 = self.bonds[n]
        m1, m2 = self.bonds[m]
        # A-B, A-C -> B,A,C
        if n1==m1:
            return n2, n1, m2
        # A-B, C-A -> B,A,C
        if n1==m2:
            return n2, n1, m1
        # A-B, B-C -> A,B,C
        if n2==m1:
            return n1, n2, m2
        # A-B, C-B -> A,B,C
        if n2==m2:
            return n1, n2, m1
        return None


from copy import deepcopy
#from mset import mset
#from cliques import largest_cliques, maximal_cliques, largest_cliques_limited, maximal_cliques_limited
import igraph as ig
from itertools import permutations

class Mapping:
    
    def __init__( self, str_smiles ):
        
        t = time.time()
        print( 'target reaction:', str_smiles )
        sm_rct, sm_prd = str_smiles.split('>>')
        
        self.rct = MolGraph( sm_rct )
        self.prd = MolGraph( sm_prd )
        self.RCT = self.rct.MOL
        self.PRD = self.prd.MOL
        self.RCT.clean2d()
        self.PRD.clean2d()
        
        self.RXN = ReactionContainer( [self.RCT], [self.PRD] )
        self.RXN.clean2d()
        
        print( '# initialization: ' )
        print( '\t>> time: ', str(round( time.time() - t, 6)).rjust(4), 'sec' )
        
    def unmapped_area( self, mp ):
        
        if mp == []:
            return [], []
        
        r_mapped, p_mapped = zip( *mp )
        r_unmap = [ x for x in self.rct.atoms if not x in r_mapped ]
        p_unmap = [ x for x in self.prd.atoms if not x in p_mapped ]
        
        return r_unmap, p_unmap
    
    def modular_product( self, mode='bond', premap=[] ):
        
        t = time.time()
        
        r_area, p_area = self.unmapped_area( premap )
        
        r_neib = list( self.rct.an.values() )
        p_neib = list( self.prd.an.values() )
        
        if mode == 'atom':
            
            nodes = []
            for i, x in enumerate( r_neib ):
                for j, y in enumerate( p_neib ):
                    if x == y:
                        if r_area == [] and p_area == []: 
                            nodes.append((i+1, j+1))
                        else:
                            if (i+1 in r_area) and (j+1 in p_area):
                                nodes.append((i+1, j+1))
            
            #if len( nodes ) > 500:
            #    print( len( nodes ), 'too many nodes' )
            #    return 'too large', None
            
            edges = []
            for node1, node2 in combinations( nodes, 2 ):
                r1, p1 = node1
                r2, p2 = node2
                if r1!=r2 and p1!=p2:
                    r_bond = self.rct.refer_bond( r1, r2 )
                    p_bond = self.prd.refer_bond( p1, p2 )
                    i = nodes.index( node1 )
                    j = nodes.index( node2 )
                    if r_bond == p_bond:
                        edges.append((i, j))
        
        if mode == 'bond':
            
            rb_neib = [ (r_neib[x-1], r_neib[y-1]) for x,y in self.rct.bonds ]
            pb_neib = [ (p_neib[x-1], p_neib[y-1]) for x,y in self.prd.bonds ]
            
            nodes = []
            for i, (x,y) in enumerate( rb_neib ):
                for j, (v,w) in enumerate( pb_neib ):
                    # n-neighbourhood of bond
                    if (x==v and y==w) or (x==w and y==v):
                        if r_area == [] and p_area == []:
                            nodes.append((i,j))
                        else:
                            r1, r2 = self.rct.bonds[i]
                            p1, p2 = self.prd.bonds[j]
                            #if ((r1 in r_area) or (r2 in r_area)) and ((p1 in p_area) or (p2 in p_area)):
                            if (r1 in r_area) and (r2 in r_area) and (p1 in p_area) and (p2 in p_area):
                                nodes.append((i,j))
            
            #if len( nodes ) > 500:
            #    print( len( nodes ), 'too many nodes' )
            #    return 'too large', None
            
            edges = []
            for node1, node2 in combinations( nodes, 2 ):
                r1, p1 = node1
                r2, p2 = node2
                if r1!=r2 and p1!=p2:
                    # being common atom or not
                    if self.rct.bond_common_atom( r1, r2 ) == self.prd.bond_common_atom( p1, p2 ):
                        i = nodes.index( node1 )
                        j = nodes.index( node2 )
                        edges.append( ( i, j ) )
        
        #print( '# calculation of product graph: mode =', mode, ', neib =', neib )
        #print( '\t>> '+str(mode)+'-to-'+str(mode)+' product graph:', str( len( nodes ) ).rjust(8), 'nodes,', str( len( edges ) ).rjust(8), 'edges,' )
        #print( '\t>> time: ', str( round( time() - t, 6 )).rjust(2), 'sec' )
        
        #return mode, nodes, edges
        return nodes, edges
    
    # conversion to atom-to-atom map from bond-to-bond map
    def to_atom_mapping( self, bond_map ):
        
        atom_map = []
        
        # bond consisting defferent two elements
        for n, m in bond_map:
            r1, r2 = self.rct.bonds[n]
            p1, p2 = self.prd.bonds[m]
            r_an1, r_an2 = self.rct.bn_an[n]
            p_an1, p_an2 = self.prd.bn_an[m]
            
            if r_an1!=r_an2 and p_an1!=p_an2:
                # two bonds A-B and C-D is A->C and B->D
                if r_an1==p_an1 and r_an2==p_an2:
                    atom_map += [(r1,p1), (r2,p2)]
                # two bonds A-B and C-D is A->D and B->C
                if r_an1==p_an2 and r_an2==p_an1:
                    atom_map += [(r1,p2), (r2,p1)]
        
        # bond consisting same two elements
        for (r1,p1), (r2,p2) in combinations( bond_map, 2 ):
            if self.rct.bond_common_atom( r1, r2 ):
                
                rx, ry, rz = self.rct.bond_mapping( r1, r2 )
                px, py, pz = self.prd.bond_mapping( p1, p2 )
                atom_map += [ (rx, px), (ry, py), (rz, pz)]
        
        # removing duplicate
        atom_map = list( set( atom_map ) )
        atom_map.sort()
        
        return atom_map
    
    # get atom-to-atom map from some cliques
    def maximum_cliques( self, modular, premap=[], completion=True, optimizer='aki', timeout=10.0 ):
        
        t = time.time()
        
        mode, nodes, edges = modular
        
        if optimizer == 'aki':
            #maximum_cliques = largest_cliques( mapdat.edges )
            max_cqs = largest_cliques_limited( edges, timeout )
            
        if optimizer == 'igraph':
            max_cqs = ig.Graph( edges ).largest_cliques()
        
        if max_cqs == []:
            print('! ----- empty cliques ----- !')
            total_maps = [premap]
        else:
            maps = []
            for cq in max_cqs:
                mapping = [ nodes[x] for x in cq ]
                mapping.sort()
                maps.append( mapping )
            
            if mode == 'atom':
                atom_maps = maps
            
            if mode == 'bond':
                #print( 'transformation from bond-to-bond to atom-to-atom mapping' )
                atom_maps = []
                for bond_map in maps:
                    atom_map = self.to_atom_mapping( bond_map )
                    atom_map.sort()
                    atom_maps.append( atom_map )
            
            total_maps = [ list( set( x + premap ) ) for x in atom_maps ]
        
        running_time = time.time() - t
        
        print( '# maximum cliques: mode =', completion, ', timeout =', timeout )
        if total_maps != []:
            print( '\t>> ',len(total_maps[0]), '/', len( self.rct.atoms ), 'atoms were mapping..' )
        print( '\t>> ', str(len(nodes)).rjust(8), 'atoms,', str(len(edges)).rjust(8), 'nodes,' )
        print( '\t>> time: ', str( round( running_time, 6) ).rjust(4), 'sec' )
        
        if running_time + 0.01 > timeout:
            print( '\t>> QUALITY = APPROX' )
        
        else:
            print( '\t>> QUALITY = EXACT' )
        
        return total_maps
    
    def permutation_complete( self, premaps=[] ):
        
        total_added = []
        for x in premaps: 
        
            r_mapped, p_mapped = zip( *x )
            r_rsd = [ x for x in self.rct.atoms if not x in r_mapped ]
            p_rsd = [ x for x in self.prd.atoms if not x in p_mapped ]
            
            # if len( r_rsd ) >= 7:
            #     print( 'too large permutation was detected: ', len( r_rsd ), '!' )
            #     raise Exception    
                    
            prd_an = [ self.prd.an[x] for x in p_rsd ]
            
            for perm in list( permutations( r_rsd ) ):
                rct_an = [ self.rct.an[x] for x in perm ]
                
                if rct_an == prd_an:
                    additional = list( zip( perm, p_rsd ) )
                    total_added.append( x + additional )
        
        return total_added

        print( '\t>>', len( unique_pair ), 'pairs were found..' )
        print( '\t>>', unique_pair )
        print( '\t>> time: ', str( round( time.time() - t, 6 ) ).rjust(4), 'sec' )
        
        return unique_pair
    

    def isomorphism( self, maps, mode='bond' ):
        
        t = time.time()
        
        def connected_comp( idxs, adj ):

            adj_dic = dict()
            for i, j in adj:
                if i in adj_dic.keys():
                    adj_dic[i].add( j )
                else:
                    adj_dic[i] = { j }
                if j in adj_dic.keys():
                    adj_dic[j].add( i )
                else:
                    adj_dic[j] = { i }        

            def next_adj( adj_dic, pivot ):
                nexts = set()
                for p in pivot:
                    if not p in adj_dic.keys():
                        nexts |= {p}
                    else:
                        nexts |= adj_dic[p]

                return nexts

            visited = set()
            cand = set()
            covered = set()
            connected_comp = []

            while covered != set( idxs ):
                residue = set( idxs ) - covered 
                cand = { residue.pop() }
                visited = cand

                while cand != set():
                    visited |= cand
                    cand = next_adj( adj_dic, cand ) - visited

                connected_comp.append( visited )
                for x in connected_comp:
                    covered |= x

            return connected_comp

        if len( maps ) > 10000:
            print( 'isomorphism is skipped because of too many', len( maps ), 'mappings' )
            return [ set( i for i,_ in enumerate( maps ) ) ]
            
        r_iso = []
        p_iso = []
        
        if mode == 'skelton':
            for mp in maps:
                
                r_mapped, p_mapped = zip( *mp )
                mapped_r_bonds = [ (x,y) for x, y in self.rct.bonds if x in r_mapped and y in r_mapped ]
                mapped_p_bonds = [ (x,y) for x, y in self.prd.bonds if x in p_mapped and y in p_mapped ]
                to_rct = { y:x for x, y in mp }
                to_prd = { x:y for x, y in mp }
                conv_p_bonds = []
                conv_r_bonds = []
                
                for x, y in mapped_p_bonds:
                    s, t = to_rct[x], to_rct[y]
                    if s > t:
                        s, t = t, s
                    conv_p_bonds.append(( s, t ))
                
                for x, y in mapped_r_bonds:
                    s, t = to_prd[x], to_prd[y]
                    if s > t:
                        s, t = t, s
                    conv_r_bonds.append(( s, t ))
                
                conv_r_bonds.sort()
                conv_p_bonds.sort()
                
                if mode == 'skelton':
                    rs, ps = zip( *mp )
                    r_Hs = [ ( to_prd[x], self.rct.dic_hyd[x]) for x in rs ]
                    p_Hs = [ ( to_rct[x], self.prd.dic_hyd[x]) for x in ps ]
                    r_Hs.sort()
                    p_Hs.sort()
                    r_iso.append(( conv_r_bonds, r_Hs ))
                    p_iso.append(( conv_p_bonds, p_Hs ))
        
        if mode == 'bond':
            for mp in maps:
                
                r_mapped, p_mapped = zip( *mp )
                mapped_r_bonds = [ (x, y, self.rct.dic_order[(x, y)]) for x, y in self.rct.bonds if x in r_mapped and y in r_mapped ]
                mapped_p_bonds = [ (x, y, self.prd.dic_order[(x, y)]) for x, y in self.prd.bonds if x in p_mapped and y in p_mapped ]
                to_rct = { y:x for x, y in mp }
                to_prd = { x:y for x, y in mp }
                conv_p_bonds = []
                conv_r_bonds = []
                
                for x, y, order in mapped_p_bonds:
                    s, t = to_rct[x], to_rct[y]
                    if s > t:
                        s, t = t, s
                    conv_p_bonds.append(( s, t, order ))
                
                for x, y, order in mapped_r_bonds:
                    s, t = to_prd[x], to_prd[y]
                    if s > t:
                        s, t = t, s
                    conv_r_bonds.append(( s, t, order ))
                
                rs, ps = zip( *mp )
                r_Hs = [ ( to_prd[x], self.rct.dic_hyd[x]) for x in rs ]
                p_Hs = [ ( to_rct[x], self.prd.dic_hyd[x]) for x in ps ]
                r_Hs.sort()
                p_Hs.sort()
                conv_r_bonds.sort()
                conv_p_bonds.sort()
                
                r_iso.append(( conv_r_bonds, r_Hs ))
                p_iso.append(( conv_p_bonds, p_Hs ))
        
        # analysis of symmetry network
        # making adjascent list about symmetry network
        connectivity = []
        for i, j in combinations( [ k for k, _ in enumerate( maps ) ], 2 ):
            if r_iso[i] == r_iso[j] or p_iso[i] == p_iso[j]:
                connectivity.append(( i, j ))
        
        maps = [ k for k, _ in enumerate( maps ) ]
        #print( maps, connectivity )
        
        return connected_comp( maps, connectivity )
    
    # do not detected chaning of multiple bonds 
    def change_bonds( self, mp ):
        
        print( mp )
        dic_map = dict( mp )
        cleav_bonds = []
        for x, y in self.rct.bonds:
            if (not x in dic_map.keys()) or (not y in dic_map.keys()):
                continue
            if self.prd.refer_bond( dic_map[x], dic_map[y] ) == None:
                cleav_bonds.append( ( x, y ) )
        
        dic_map_rev = dict( [ (y,x) for x,y in mp ] )
        form_bonds = []
        for x,y in self.prd.bonds:
            if (not x in dic_map_rev.keys()) or (not y in dic_map_rev.keys()):
                continue
            if self.rct.refer_bond( dic_map_rev[x], dic_map_rev[y] ) == None:
                form_bonds.append( ( x, y ) )
        
        #print( 'cleavage', cleav_bonds )
        #print( 'formation', form_bonds )
        return cleav_bonds, form_bonds
        
    def score( self, maps, mode='hydrogen' ):
        
        t = time.time()
        
        scores = []
        for mp in maps:
            sc = 0
            
            # change the number of substituted hydrogens
            if mode == 'hydrogen':
                sc += sum( [ abs( self.rct.dic_hyd[x] - self.prd.dic_hyd[y] ) for x, y in mp ]  )
            
            # change of the bond mupliplicity
            if mode == 'bond':
                for (x, y), (s, t) in combinations( mp, 2 ):
                    sc += abs( self.rct.dic_order[(x,s)] - self.prd.dic_order[(y,t)] )
            
            if mode == 'hydrogen + bond':
                sc += sum( [ abs( self.rct.dic_hyd[x] - self.prd.dic_hyd[y] ) for x, y in mp ]  )
                for (x, y), (s, t) in combinations( mp, 2 ):
                    sc += abs( self.rct.dic_order[(x,s)] - self.prd.dic_order[(y,t)] )
            
            scores.append( sc )
        
        # list of minimum score
        best_score = min( scores )
        best_maps = [ maps[i] for i, x in enumerate( scores ) if x == best_score ]
        
        #print( '# score calculation: mode =', mode )
        #print( '\t>>', len( best_maps ), 'best maps , score =', best_score, ', length =', len( best_maps[0] ) )
        #print( '\t>> time: ', str( round( time() - t, 6 ) ).rjust(4), 'sec' )
        
        return best_maps
    
    def calculate_mapping( self, mode='bond', optimizer='aki', timeout=10.0 ):
        
        t0 = time.time()
        
        #premap = self.find_unique_pair()
        premap = []
        modular = self.form_graph_product( mode='bond', neib=0, premap=premap )
        print('TEST1')
        maps = self.maximum_cliques( modular=modular, premap=premap, completion=True, optimizer='aki', timeout=timeout )
        print('TEST2')
        maps2 = self.permutation_complete( maps )
        
        print( '\t>>', len( maps2 ), 'maps were found..' )
        print( '>>>>> mapping time:', str( round( time.time()-t0, 6 ) ).rjust( 4 ) )
        
        return maps2
    
    def check_answer( self, maps ):
        
        answer = [ (x+1, x+1) for x, _ in enumerate( maps[0] ) ]
        
        for x in maps:
            gp = self.isomorphism( [ x, answer ], mode='skelton' )
            if len( gp ) == 1:
                return x
        
        return None

    def cliques_to_mappings( self, nodes, cliques ):
        
        btb_maps = [[ nodes[x] for x in cq ] for cq in cliques ]
        maps = [ self.to_atom_mapping( x ) for x in btb_maps ]
        comp_maps = self.permutation_complete( maps )

        for x in comp_maps:
            x.sort()
        
        return comp_maps

    def filtering( self, maps, mode ):

        if mode == 'filter1' or 'filter2':
            selected = self.score( maps, 'hydrogen' )

        if mode == 'filter2':
            return self.score( selected, 'bond' )
        else:
            return selected

    def show_mapping( self, mapping ):
        cp_self = deepcopy( self )
        cp_self.PRD.remap( dict( mapping ) )

        return cp_self.RXN

    def export_mapping( self, mapping ):
        cp_self = deepcopy( self )
        cp_self.PRD.remap( dict( mapping ) )
        
        return cp_self.RXN.__format__( 'm' )

    def non_equivalent( self, maps ):
        groups = self.isomorphism( maps, 'bond' )

        return [ maps[gp.pop()] for gp in groups ]