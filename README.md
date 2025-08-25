This repo has some additional filters for HiPRGen. 

Especially for now it focusses on more effectively filtering out reactions. The current star count method implemented in HiPRGen fails to work well because not all atoms in a reaction are unique. Ideally HiPRGen has to remove reactions involving the simultaneous breakage/formation of more than two bonds in total or which are break-1-form-1 but do not have a reaction center involved in both the breakage and formation, as such reactions are unlikely to occur in a single concerted step.

In this repo, we utilize the AAM-Ising repository which performs atom-to-atom mapping quickly and efficiently for the main chain of atoms (C, O, Na, S,... but not H). Then we have added some additional functions given in "additional_filters.py" that helps achieve what HiPRGen ideally wanted to. 
