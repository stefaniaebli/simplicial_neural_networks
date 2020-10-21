#!/usr/bin/env python3

import sys

import numpy as np
import gudhi
from scipy import sparse

def mask_all_deg(bipartite,values_collaboration,simplices,seen_percentage,max_dim=3):
	"""From a collaboration simplicial complex and its collaboration values to a list of masked simplices.
	Masked simplices are the result of masking real collaborations and the faces of their associated simplices.

	Parameters
	----------
	bipartite:
		Sparse matrix representing the collaboration bipartite graph.
	values_colab: ndarray
		Array of size bipartite.shape[0], containing the values of the collaboration.
	simplices:
		List of dictionaries of simplices of the collaboration complex of order k and their indices.
	mask_percentage:
		Perecentage of masked real collaboration -percentage of seen real collaborations-
	max_dim:
		maximal dimension of the simplices in the collaboration simplicial complex
	Returns
	-------
	mask
	List of dictionaries of simplices of order k that have been masked and their indices.
	"""
	Al=bipartite.tolil()
 	indices =[]
 	for j,authors in enumerate(Al.rows):
   		if len(authors)<=max_dim+1:
             		indices.append(j)
             	else:
            		continue
 	mask = [dict() for _ in range(max_dim+1)]
	l=int(np.ceil((len(indices)/100)*seen_percentage))
 	mask_maxdim=np.random.choice(indices, l)
 	for j,index in enumerate(mask_maxdim):
 		authors=frozenset(Al.rows[index])
		st=gudhi.SimplexTree()
		st.insert(authors)
		for face, _ in st.get_skeleton(st.dimension()):
			k = len(face)
			mask[k-1][frozenset(face)] = simplices[k-1][frozenset(face)]
	return mask

	##### Input: collaboration bipartite graph and the value of the collaborations.
	##### Output: collaborations simplicial complex and the values of the collaborations on each simplex.

adjacency = sys.argv[1]
citations = sys.argv[2]
simplices =  sys.argv[3]
seen_percentage = int(sys.argv[4])
max_dim = int(sys.argv[5])
output = sys.argv[6]
print('Building mask for collaboration simplical complex')
adjacency = sparse.load_npz(adjacency)
citations = np.load(citations)
simplices=np.load(simplices)
seen_percentage=int(seen_percentage)

mask_all_deg=mask_all_deg(adjacency,citations,simplices,seen_percentage,max_dim=max_dim)

np.save(output + '_mask_all_deg.npy', mask_all_deg)
