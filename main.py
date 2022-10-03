"""

Test module

"""
from array import array
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
        #alpha = (norm + 1)**-1
        alpha = 1/3
        beta = np.exp(-norm*0.5)
        return grid_y * alpha + grid_x * beta

    def vec_funct_in_y(grid_x, grid_y):
        norm = grid_x**2+grid_y**2 
        #alpha = (norm + 1)**-1
        alpha = 1/3
        beta = np.exp(-norm*0.5)
        return -grid_x * alpha + grid_y * beta

    def divergence_expected(grid_x, grid_y):
        norm = grid_x**2+grid_y**2 
        beta = np.exp(-norm*0.5)
        return (2-grid_x**2-grid_y**2)*beta


    ###--- an instance for testing
    op = o.Operator(vec_funct_in_x, vec_funct_in_y)

    # corrected velocity
    vel_field_x_ans_grid = vec_funct_in_x_ans(op.X, op.Y)
    vel_field_y_ans_grid = vec_funct_in_y_ans(op.X, op.Y)

    # divergence expected
    div_exp = divergence_expected(op.X, op.Y)

    start_time = time.time()  # we start the measure
    # solve the poisson equation
    phi = op.jacobi_iterator()
    finish_time = time.time()  # we start the measure

    print(f"time serial: {finish_time - start_time}")



    #write_into_csv(phi_[0], f'phi_p_{phi_[1]}')
    #phi = op.X**2 + op.Y**2
    laplace_phi = op.laplace(phi)
    div_vel = op.numerical_div(op.vel_field_x, op.vel_field_y)  # compute the divergence of the velocity field
    
    dif = div_vel + laplace_phi  # first proof: this must be zero
    grad_phi = op.numerical_grad(phi)

    # every time i apply the numerical derivative i lost information around the borders
    #a = op.numerical_div(grad_phi[0], grad_phi[1])
    #a_ = op.numerical_div(numpy_grad[1], numpy_grad[0])
    #c = -div_vel[2:-2,2:-2] - a
    
    #--- correction
    new_f_x = op.vel_field_x + grad_phi[0]
    new_f_y = op.vel_field_y + grad_phi[1]

    correc = op.numerical_div(new_f_x, new_f_y)

    #--- cvs writings
    # write_into_csv(ans, 'ans')
    # write_into_csv(phi, 'phi')
    # write_into_csv(laplace_phi, 'laplace')
    # write_into_csv(div_vel, 'div_vel')  # numerical divergence of the vector field
    # write_into_csv(div_exp, 'div_exp')  # mathematical divergence of the vector field
    # write_into_csv(dif, 'dif')

    #plt.quiver(op.X, op.Y, vel_field_x_ans_grid, vel_field_y_ans_grid)
    #plt.quiver(op.X, op.Y, op.vel_field_x, op.vel_field_y, color = 'blue')
    #plt.quiver(op.X, op.Y, new_f_x, new_f_y, color = 'red')
    #plt.quiver(x, y, grad_phi[0], grad_phi[1])
    #plt.show()
    #plt.quiver(x, y, f_modified,f_modified)
    #--- plotting of correction
    # plt.figure()
    # plt.pcolormesh(op.X[1:-1,1:-1], op.Y[1:-1,1:-1], correc[1:-1,1:-1], vmin=-2, vmax=2)
    # plt.colorbar()

    # #--- plotting of poisson solution
    # plt.figure()
    # plt.pcolormesh(op.X[1:-1,1:-1], op.Y[1:-1,1:-1], dif[1:-1,1:-1], vmin=-2, vmax=2)
    # plt.colorbar()
    # plt.show()
    

    print("done")