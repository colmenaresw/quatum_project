"""

Test module

"""
from array import array
import numpy as np
import matplotlib.pyplot as plt
import Operators as o




if __name__ == "__main__":

    op = o.Operator()

    # corrected velocity
    vel_field_x_ans = o.vec_funct_in_x_ans(op.X, op.Y)
    vel_field_y_ans = o.vec_funct_in_y_ans(op.X, op.Y)

    phi = op.jacobi_iterator()  # this phi already lost the info around the borders
    #phi = op.X**2 + op.Y**2
    laplace_phi = op.laplace(-phi)
    div_vel = op.numerical_div(op.vel_field_x, op.vel_field_y)
    dif = -laplace_phi - div_vel
    grad_phi = op.numerical_grad(-phi[1:-1,1:-1])  # second lost of info

    # every time i apply the numerical derivative i lost information around the borders
    a = op.numerical_div(grad_phi[0][1:-1,1:-1], grad_phi[1][1:-1,1:-1])
    c = -div_vel[2:-2,2:-2] - a
    #ans = div_vel - a

    # correction
    new_f_x = op.vel_field_x[1:-1, 1:-1] + grad_phi[0]
    new_f_y = op.vel_field_y[1:-1, 1:-1] + grad_phi[1]

    ans = op.numerical_div(new_f_x, new_f_y)

    plt.quiver(op.X, op.Y, vel_field_x_ans, vel_field_y_ans)
    plt.quiver(op.X, op.Y, op.vel_field_x, op.vel_field_y, color = 'blue')
    plt.quiver(op.X[1:-1,1:-1], op.Y[1:-1, 1:-1], new_f_x, new_f_y, color = 'red')
    #plt.quiver(x, y, grad_phi[0], grad_phi[1])
    plt.show()
    #plt.quiver(x, y, f_modified,f_modified)
    # plt.pcolormesh(op.X, op.Y, f)
    # plt.colorbar()
    # plt.show()
    # print("done")