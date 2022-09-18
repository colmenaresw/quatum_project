"""
    class containing the implementation of the operators

"""

from random import random
import numpy as np
import copy
from mpi4py import MPI

############ constanst declarations
NUM_OF_POINTS = 20  # number of points for the discretize grid
D_SIZE = 1  # the size of the domain
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()  # how many processor there are
name = comm.Get_name()
N = 9  # how many columns will i store per processor (num of cols-2)/(num of processors)

############ class
class Operator:
    def __init__(self, vec_funct_in_x, vec_funct_in_y) -> None:
        """
            vel_funct_in_x : the velocity function for x-coordinates
            vel_funct_in_y : the velocity function for y-coordinates

        """
        self.delta = D_SIZE / (NUM_OF_POINTS - 1)  # size of the grid
        self.x_size = np.linspace(0, D_SIZE, NUM_OF_POINTS)
        self.y_size = np.linspace(0, D_SIZE, NUM_OF_POINTS)

        # two dimensional grid
        self.X, self.Y = np.meshgrid(self.x_size, self.y_size)
        self.vel_field_x = vec_funct_in_x(self.X, self.Y)
        self.vel_field_y = vec_funct_in_y(self.X, self.Y)


    # partial derivatives
    def numerical_partial_deriv_of_x(self, vector_field_fx):  # central difference squeme for x
        delta_x = np.zeros_like(vector_field_fx)
        num_of_col = vector_field_fx.shape[1]
        for j in range(NUM_OF_POINTS):
            for i in range(num_of_col):
                if i == num_of_col - 1:  # backward difference
                    delta_x[j, i] = (
                        
                    3 * vector_field_fx[j, i] -
                    4 * vector_field_fx[j, i-1] +
                    vector_field_fx[j, i-2] 
                    ) / (
                        2 * self.delta
                    )
                elif i == 0:  # foward difference
                    delta_x[j, i] = (
                    -3 * vector_field_fx[j, i] +
                    4 * vector_field_fx[j, i+1] -
                    vector_field_fx[j, i+2] 
                    ) / (
                        2*self.delta
                    )

                else:
                    delta_x[j, i] = (
                        vector_field_fx[j, i+1] -
                        vector_field_fx[j, i-1] 
                    ) / (
                        2*self.delta
                    )


        return delta_x


    def numerical_partial_deriv_of_y(self, vector_field_fy):  # central difference squeme for x
        delta_y = np.zeros_like(vector_field_fy)
        num_of_col = vector_field_fy.shape[1]
        for i in range(num_of_col):
            for j in range(NUM_OF_POINTS):
                if j == num_of_col - 1:  # backward difference
                    delta_y[j, i] = (
                    3 * vector_field_fy[j, i] -
                    4 * vector_field_fy[j-1, i] +
                    vector_field_fy[j-2,i]
                    ) / (
                        2 * self.delta
                    )
                elif j == 0:  # foward difference
                    delta_y[j, i] = (
                    -3 * vector_field_fy[j, i] +
                    4 * vector_field_fy[ j + 1, i] -
                    vector_field_fy[j + 2, i]
                    ) / (
                        2 * self.delta
                    )

                else:
                    delta_y[j, i] = (
                        vector_field_fy[j+1, i] -
                        vector_field_fy[j-1, i] 
                    ) / (
                        2*self.delta
                    )


        return delta_y
    

    # operators
    def numerical_div(self, vector_field_fx, vector_field_fy):
        """
            takes in a vector function, returns a scalar
        """
        partial_x = self.numerical_partial_deriv_of_x(vector_field_fx)
        partial_y = self.numerical_partial_deriv_of_y(vector_field_fy)

        return partial_x + partial_y


    def numerical_grad(self, scalar_field):
        """
            takes in a scalar function, return a vector
        """
        partial_x = self.numerical_partial_deriv_of_x(scalar_field)
        partial_y = self.numerical_partial_deriv_of_y(scalar_field)

        return [partial_x,  partial_y]      


    def laplace(self, f):
        diff = np.zeros_like(f)
        diff[1:-1, 1:-1] = (
            f[1:-1, 0:-2]
            +
            f[0:-2, 1:-1]
            -
            4
            *
            f[1:-1, 1:-1]
            +
            f[1:-1, 2:  ]
            +
            f[2:  , 1:-1]
        ) / (
            self.delta**2
        )
        return diff


    # jacobi iteration
    def jacobi_iterator(self):
        """
            implementation of the jacobi iteration
        """
        phi_temp = np.zeros_like(self.vel_field_x)  # allocate space for the temporary answer
        rho = self.numerical_div(self.vel_field_x, self.vel_field_y)  # source function of poisson equation
        counter = 0

        while True:
            phi = np.zeros_like(phi_temp)
            # we iterate over the internal grid
            for y_dir in range(1, NUM_OF_POINTS - 1, 1):
                for x_dir in range(1, NUM_OF_POINTS - 1, 1):
                    # store the values of the stencil
                    up = phi_temp[y_dir-1, x_dir]
                    down = phi_temp[y_dir + 1, x_dir]
                    left = phi_temp[y_dir, x_dir - 1]
                    right = phi_temp[y_dir, x_dir + 1]
                    center = rho[y_dir, x_dir]

                    # calculate answer
                    phi[y_dir, x_dir] = (center * self.delta**2  +
                                        down + 
                                        up +
                                        left +
                                        right                                    
                                        ) * 0.25
            
        

            # e1 = np.linalg.norm(phi, ord=np.inf)  # infinity norm
            # e0 = np.linalg.norm(phi_temp, ord=np.inf)
            # error = abs(e0-e1)

            error = abs(phi-phi_temp)
            error = error.max()
            if error < 2e-10 or counter > 10000:
                break

            counter += 1

            phi_temp = copy.deepcopy(phi)
        return phi


    # parallel jacobi
    def p_jacobi_iterator(self):
            """
                implementation of the jacobi iteration in parallel
            """
            print("hi, i'm: ", rank)
            counter = 0
            dimen = self.vel_field_x[:, 0:N+2].shape
            num_of_p = dimen[1]
            l_phi_temp = np.zeros(dimen)  # allocate space for the temporary answer in the local processor
            l_rho = self.numerical_div(self.vel_field_x, 
                                       self.vel_field_y)[:, rank*N : rank*N + N + 2]  # source function of poisson equation for the local processor
            

            while True:
                l_phi = np.zeros_like(l_phi_temp)
                # we iterate over the internal grid
                for y_dir in range(1, NUM_OF_POINTS - 1, 1):
                    for x_dir in range(1, num_of_p-1, 1):
                        # store the values of the stencil
                        up = l_phi_temp[y_dir-1, x_dir]
                        down = l_phi_temp[y_dir + 1, x_dir]
                        left = l_phi_temp[y_dir, x_dir - 1]
                        right = l_phi_temp[y_dir, x_dir + 1]
                        center = l_rho[y_dir, x_dir]

                        # calculate answer
                        l_phi[y_dir, x_dir] = (center * self.delta**2  +
                                            down + 
                                            up +
                                            left +
                                            right                                    
                                            ) * 0.25

                ### we compute the local error
                l_error = abs(l_phi-l_phi_temp)
                l_error = l_error.max()

                # we gather the errors to check for the stopping criteria
                # perform the reduction
                value_max = np.array(0.0,'d')
                comm.Reduce(l_error, value_max, op=MPI.MAX, root=0)
                counter += 1
                

                # we check for the stopping criteria by sending the error to every processor
                value_max = comm.bcast(value_max, root=0)
                if value_max <= 2e-10 or counter > 10000:
                    print("counter is: ", counter)
                    break
                

                # if we need to continue here we create a copy for the next iteration
                l_phi_temp = copy.deepcopy(l_phi)

                # we exchange the boundaries to compute the next iteration
                left_boundary = l_phi[:,0]
                right_boundary = l_phi[:,-2]

                ################################################## we make the sends for left and right boundaries
                # even processors send and odd ones receive
                if rank%2 == 0:
                    #print(f"sending from... {rank} and I'm sending {right_boundary}")
                    comm.send(l_phi[:,-2], dest=1)
                    
                else:
                    #print(f"I'm {rank} and my current value is {l_phi} i'm suppose to change my left column")
                    left_boundary = comm.recv(source=0)
                    l_phi_temp[:,0] = left_boundary
                    #print(f"\nI'm {rank} and my new left column is: {l_phi_temp}")

                # odd processors send and even ones receive
                if rank%2 != 0:  # odd processors
                    #print(f"sending from... {rank} and I'm sending {right_boundary}")
                    comm.send(l_phi[:,1], dest=0)  # sending left boundary
                    
                else:
                    #print(f"I'm {rank} and my current value is {l_phi} i'm suppose to change my right column")
                    right_boundary = comm.recv(source=1)
                    l_phi_temp[:,-1] = right_boundary
                    #print(f"\nI'm {rank} and my new right column is: {l_phi_temp}")


                ### if your are not the first neither the last processor
                if rank != 0 and rank != size-1:
                    #print("im here")
                    # for even processors
                    if rank%2==0:
                        comm.send(l_phi[:, 1], dest=rank-1)  # send your left column
                    else:
                        right_boundary = comm.recv(source=rank+1)
                        l_phi_temp[:,-1] = right_boundary

                    # for odd processors
                    if rank%2!= 0:
                        comm.send(l_phi[:,-1], dest=rank+1)
                    else:
                        left_boundary = comm.recv(source=rank-1)
                        l_phi_temp[:,1] = left_boundary

            ############################ we gather the result
            sendbuf = l_phi[1:-1, 1:-1]
            rows = sendbuf.shape[0]
            cols = sendbuf.shape[1]
            dim = rows * cols
            sendbuf = sendbuf.reshape(dim,1)
            all_phi = np.array([])


            if rank != 0:
                # every processor should send their data to zero
                comm.Send(sendbuf, dest=0)

            elif rank == 0:

                all_phi = np.zeros((NUM_OF_POINTS, NUM_OF_POINTS))  # this is the final matrix
                all_phi[1:-1, rank*N + 1 : rank*N + N + 1] = l_phi[1:-1, 1:-1]  # for the first processor we allocate the result


                data = np.empty(dim, dtype='d')  # allocate space to receive the array
                for each_p in range(1,size):
                    comm.Recv(data, source=each_p)
                    all_phi[1:-1, each_p*N + 1 : each_p*N + N + 1] = data.reshape((rows,cols))

                        
            return (all_phi,rank)


############ testing of the script
if __name__ == "__main__":
    o = Operator()
    print(o.X)
    vel_field_x = vec_funct_in_x(o.X, o.Y)
    vel_field_y = vec_funct_in_y(o.X, o.Y)
    o.numerical_partial_deriv_of_x(vel_field_x)
    o.numerical_partial_deriv_of_y(vel_field_y)