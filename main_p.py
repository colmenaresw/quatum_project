"""

Test module

"""
from array import array
import mpi4py
import numpy as np
import matplotlib.pyplot as plt
import Operators as o
import time
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()  # how many processor there are
name = comm.Get_name()

def write_into_csv(arr, name):
    """
        a function to store results
        arr: array you want to store
        name: name of the file you want to save
    """
    np.savetxt(f'./data_csv/{name}.csv', arr, delimiter='|')


if __name__ == "__main__":

    

    # velocity field declaration
    def vec_funct_in_x_ans(grid_x, grid_y):
        norm = grid_x**2+grid_y**2 
        alpha = (norm + 1)**-1
        #alpha = 1/3
        return grid_y * alpha

    def vec_funct_in_y_ans(grid_x, grid_y):
        norm = grid_x**2+grid_y**2 
        alpha = (norm + 1)**-1
        #alpha = 1/3
        return -grid_x * alpha

    def vec_funct_in_x(grid_x, grid_y):
        norm = grid_x**2+grid_y**2 
        alpha = (norm + 1)**-1
        #alpha = 1/3
        beta = np.exp(-norm*0.5)
        return grid_y * alpha + grid_x * beta

    def vec_funct_in_y(grid_x, grid_y):
        norm = grid_x**2+grid_y**2 
        alpha = (norm + 1)**-1
        #alpha = 1/3
        beta = np.exp(-norm*0.5)
        return -grid_x * alpha + grid_y * beta

    def divergence_expected(grid_x, grid_y):
        norm = grid_x**2+grid_y**2 
        beta = np.exp(-norm*0.5)
        return (2-grid_x**2-grid_y**2)*beta

    ################################################# an instance for testing
    op = o.Operator(vec_funct_in_x, vec_funct_in_y)

   

    # corrected velocity
    # vel_field_x_ans_grid = vec_funct_in_x_ans(op.X, op.Y)
    # vel_field_y_ans_grid = vec_funct_in_y_ans(op.X, op.Y)

    # divergence expected
    #div_exp = divergence_expected(op.X, op.Y)

    start_time_ = mpi4py.MPI.Wtime()  # we start the measure
    phi_ = op.p_jacobi_iterator()
    finish_time_ = mpi4py.MPI.Wtime()  # we start the measure
    
    if rank == 0:  # we record the time of the execution
        with open('time_log.txt', 'a+') as f:
            f.write(f'{size};{finish_time_ - start_time_}\n')
            print(f'for {size} the time is {finish_time_ - start_time_}')
            write_into_csv(phi_, 'phi_')
    #print(f"for processor:{rank} the result is: \ntime parallel: {finish_time_ - start_time_}")
    #print(f"for processor:{rank} the result is: \ntime parallel: {finish_time_ }")


    # if rank == 0:
    #     laplace_phi = op.laplace(phi_)
    #     div_vel = op.numerical_div(op.vel_field_x, op.vel_field_y)  # compute the divergence of the velocity field
        
    #     ans = div_vel + laplace_phi  # first proof: this must be zero

    #     # plt.pcolormesh(op.X[1:-1,1:-1], op.Y[1:-1,1:-1], ans[1:-1,1:-1], vmin=-2, vmax=2)
    #     # plt.colorbar()
    #     # plt.show()
    #     print("done")