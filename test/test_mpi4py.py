from mpi4py import MPI
import numpy as np
comm = MPI.COMM_WORLD
comm.Set_errhandler(MPI.ERRORS_RETURN)

print('rank: ', comm.rank)
src = np.zeros(1)
dst = np.zeros(4)
src[0] = comm.rank
for i in range(10):
    comm.Allgather(src, dst)
    print('dst: ', dst)

