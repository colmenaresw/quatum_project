"""

Test module

"""
from array import array
import numpy as np
import matplotlib.pyplot as plt
import Operators as o

if __name__ == "__main__":
    # vec = np.array([2,2])
    # y = vec_funct(vec)
    op = o.Operator()
    # a = op.numerical_divergence(vec)
    # b = op.numerical_gradient(vec)
    x,y = np.meshgrid(np.arange(0,1,op.delta), np.arange(0,1,op.delta))
    grid = [x,y]

    # Compute the jacobi iterator
    phi = op.jacobi_iterator(grid)

    # compute the grad of phi
    grad_phi = op.numerical_gradient_phi(phi)

    # corrected velocity


    f = o.vec_funct(x, y)
    f_modified = [f[0]+grad_phi[0], f[1]+grad_phi[1]]
    # b_ = op.numerical_gradient(grid)
    c_ = op.numerical_divergence(grid)
    # d_ = op.jacobi_iterator(grid)
    #z = np.exp(-x**2 - y**2)
    #plt.quiver(x, y, b_[0], b_[1])
    #plt.quiver(x, y, grad_phi[0], grad_phi[1])
    plt.quiver(x, y, f[0], f[1])
    #plt.show()
    plt.quiver(x, y, f_modified[0],f_modified[1] )
    #plt.pcolormesh(x,y,c_)
    #plt.colorbar()
    plt.show()
    print("done")