import numpy as np

def save_matrix(A):
    mat = np.matrix(A)
    with open('matrix.txt','wb') as f:
        for row in mat:
            np.savetxt(f, row, fmt='%.2f')
