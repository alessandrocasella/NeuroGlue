import numpy as np


def SVD(H_prune):
	svd = np.zeros([6, len(H_prune)])
	sel_svd = np.zeros([6,1])
	for j in range(len(H_prune)):
		H1 = H_prune[j]

		E = (H1[0, 0] + H1[1, 1]) / 2
		F = (H1[0, 0] + H1[1, 1]) / 2
		G = (H1[1, 0] + H1[0, 1]) / 2
		H_val = (H1[1, 0] - H1[0, 1]) / 2

		Q = np.sqrt(np.power(E, 2) + np.power(H_val, 2))
		R = np.sqrt(np.power(F, 2) + np.power(G, 2))

		a1 = np.arctan2(G, F)
		a2 = np.arctan2(H_val, E)

		svd[0, j] = Q + R  # SX SY
		svd[1, j] = Q - R

		svd[2, j] = (a2 - a1) / 2 # theta gamma
		svd[3, j] = (a2 + a1) / 2

		svd[4, j] = H1[0, 2] # tx ty
		svd[5, j] = H1[1, 2]

	return svd