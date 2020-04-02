from scipy.sparse import csr_matrix

def pairwise_jaccard_sparse(csr, epsilon):
    """Computes the Jaccard distance between the rows of `csr`,
    smaller than the cut-off distance `epsilon`.
    """
    assert(0 < epsilon < 1)
    csr = csr_matrix(csr).astype(bool).astype(int)

    csr_rownnz = csr.getnnz(axis=1)
    intrsct = csr.dot(csr.T)

    nnz_i = np.repeat(csr_rownnz, intrsct.getnnz(axis=1))
    unions = nnz_i + csr_rownnz[intrsct.indices] - intrsct.data
    dists = 1.0 - intrsct.data / unions

    mask = (dists > 0) & (dists <= epsilon)
    data = dists[mask]
    indices = intrsct.indices[mask]

    rownnz = np.add.reduceat(mask, intrsct.indptr[:-1])
    indptr = np.r_[0, np.cumsum(rownnz)]

    out = csr_matrix((data, indices, indptr), intrsct.shape)
    return out