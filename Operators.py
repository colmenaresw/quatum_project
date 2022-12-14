"""
    class containing the implementation of the operators

"""

from random import random
import numpy as np
import copy
from mpi4py import MPI
import time

#####----- constanst declarations
NUM_OF_POINTS = 2**6 # number of points for the discretize grid
D_SIZE = 1  # the size of the domain
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()  # how many processor there are
name = comm.Get_name()
#N = (NUM_OF_POINTS-2)//size  # how many columns will i store per processor (num of cols-2)/(num of processors)

#####-----
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


    #####----- partial derivatives
    def numerical_partial_deriv_of_x(self, vector_field_fx):  # central difference squeme for x
        delta_x = np.zeros_like(vector_field_fx)
        num_of_col = vector_field_fx.shape[1]
        for j in range(NUM_OF_POINTS):
            for i in range(num_of_col):
                if i == num_of_col - 1:  # backward difference for right boundary
                    delta_x[j, i] = (
                        
                    3 * vector_field_fx[j, i] -
                    4 * vector_field_fx[j, i-1] +
                    vector_field_fx[j, i-2] 
                    ) / (
                        2 * self.delta
                    )
                elif i == 0:  # foward difference for left boundary
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
                elif j == 0:  # foward difference for top boundary
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
    

    #####----- operators
    def numerical_div(self, vector_field_fx, vector_field_fy):
        """
            takes in a vector function, returns a scalar
        """
        partial_x = self.numerical_partial_deriv_of_x(vector_field_fx)
        partial_y = self.numerical_partial_deriv_of_y(vector_field_fy)
        with open('rho.npy', 'wb') as f:
            np.save(f, partial_x + partial_y)

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


    #####----- jacobi iteration
    def jacobi_iterator(self):
        """
            implementation of the jacobi iteration
        """
        phi_temp = np.zeros_like(self.vel_field_x)  # allocate space for the temporary answer
        rho = self.source_generator()
        counter = 0
        
        while True:
            phi = np.zeros_like(phi_temp)
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
    
            error = abs(phi-phi_temp)
            error = error.max()
            
            if error < 2e-10 or counter > 10000:
                #print("counter serial is: ", counter)
                break

            counter += 1

            phi_temp = copy.deepcopy(phi)
        return phi


    #####----- parallel jacobi
    def p_jacobi_iterator(self):
            """
                implementation of the jacobi iteration in parallel
            """
            np.set_printoptions(precision=3)
            N_of_each_p = self.how_many_columns()  # how many columns each processor will store
            N = N_of_each_p[rank]  # the personal for this processor
            #print(f'this is the distribution of columns: {N_of_each_p}')
            i_m = self.index_map(N_of_each_p)  # map of indices to assing the velocity field
            counter = 0
            dimen = self.vel_field_x[:, 0:N+2].shape
            num_of_p = dimen[1]
            l_phi_temp = np.zeros(dimen)  # allocate space for the temporary answer in the local processor
            # with open('rho.npy', 'rb') as f:
            #     l_rho = np.load(f)
            l_rho = self.numerical_div(self.vel_field_x, 
                                       self.vel_field_y)[:, i_m[rank][0] : i_m[rank][1]]  # source function of poisson equation for the local processor
            #print(l_rho.shape)
            #l_rho = l_rho[:, i_m[rank][0] : i_m[rank][1]+1]
            #print(f'im {rank} and im {i_m[rank][0]};{i_m[rank][1]+1}')
            #print(l_rho.shape)
            
            while True:
                l_phi = np.zeros_like(l_phi_temp)
                # if counter == 0:
                #     print(f"hi, im {rank}, and my shape is {l_phi.shape}")
                # we iterate over the internal grid
                # if counter == 0:
                #     print(f"hi, im {rank}, and I will iterate between 1 and {NUM_OF_POINTS - 1} in rows and 1 and {num_of_p-2} in columns")
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
                l_error = abs(l_phi[1:-1,1:-1]-l_phi_temp[1:-1,1:-1])
                l_error = l_error.max()
                # if rank == 2 and counter > 260:
                #     print(f"im {rank} and my local error is\n:{l_phi_temp}")

                # we gather the errors to check for the stopping criteria
                # perform the reduction
                value_max = np.array(0.0,'d')
                comm.Reduce(l_error, value_max, op=MPI.MAX, root=0)
                
                

                # we check for the stopping criteria by sending the error to every processor
                value_max = comm.bcast(value_max, root=0)
                
                #print("counter is: ", counter)
                if value_max < 2e-10 or counter >=  1000:
                    #print("counter for parallel is: ", counter)
                    break
                counter += 1
                

                # if we need to continue here we create a copy for the next iteration
                l_phi_temp = copy.deepcopy(l_phi)




                ################################################## we make the interchange of boundaries
                last_pro = size - 1
                #print(last_pro)
                # even processors send (to the right) and odd ones receive CASE 1
                if rank%2 == 0 and last_pro != rank:
                    #print(f"__do i arrive here? {rank}")
                    #print(f"\nsending from... {rank} and I'm sending \n{l_phi[:,-2]} from \n{l_phi}")
                    comm.send(l_phi[:,-2], dest=rank+1)
                    
                elif rank%2!= 0:
                    
                    #print(f"__do i arrive here? {rank}")
                    #print(f"I'm {rank} and my current value is {l_phi} i'm suppose to change my left column")
                    left_boundary = comm.recv(source=rank-1)
                    #print(f"I'm {rank} receiving from... {rank-1} and I'm receivng \n{left_boundary} to my first column in \n{l_phi_temp}")
                    l_phi_temp[:,0] = left_boundary
                    #print(f"\nI'm {rank} and my new left column is: {l_phi_temp}\n####################################\n")


                #print(f"do i arrive here? {rank}")

                # odd processors send (to the left) and even ones receive CASE 2
                if rank%2 != 0:
                    #print(f"\nsending from... {rank} and I'm sending \n{l_phi[:,1]} \nfrom \n{l_phi} to {rank-1}")
                    comm.send(l_phi[:,1], dest=rank-1)  # sending left boundary
                    
                else:
                    
                    if rank != last_pro:
                        right_boundary = comm.recv(source=rank+1)
                        #print(f"\nI'm {rank} receiving from... {rank+1} and I'm receivng \n{right_boundary} to my last column in \n{l_phi_temp}")
                        l_phi_temp[:,-1] = right_boundary
                        #print(f"\nI'm {rank} and my new right column is: {l_phi_temp}\n####################################\n")

                
                ### even ones send (to the left) and odd ones receive CASE 3
                if rank%2==0:
                    if rank != 0:
                        #print(f"\nsending from... {rank} and I'm sending \n{l_phi[:,1]} \nfrom \n{l_phi} to {rank-1}")
                        comm.send(l_phi[:, 1], dest=rank-1)  # send your left column
                else:
                    if rank != last_pro:
                        right_boundary = comm.recv(source=rank+1)
                        #print(f"\nI'm {rank} receiving from... {rank+1} and I'm receivng \n{right_boundary} \nto my last column in \n{l_phi_temp}")
                        l_phi_temp[:,-1] = right_boundary
                        #print(f"\nI'm {rank} and my new right column is: \n{l_phi_temp}\n####################################\n")

                # odd ones send (to the left) and even ones receive CASE 4
                if rank%2!= 0:
                    if rank != last_pro:
                        #print(f"\nsending from... {rank} and I'm sending \n{l_phi[:,-2]} \nfrom \n{l_phi} to \n{rank+1}")
                        comm.send(l_phi[:,-2], dest=rank+1)
                else:
                    if rank != 0:
                        left_boundary = comm.recv(source=rank-1)
                        #print(f"\nI'm {rank} receiving from... {rank-1} and I'm receivng \n{left_boundary} \nto my first column in \n{l_phi_temp}")
                        l_phi_temp[:,0] = left_boundary
                        #print(f"\nI'm {rank} and my new left column is: \n{l_phi_temp}\n####################################\n")

                # if rank == 0:
                #     print(f"\ncounter is: {counter} Im 0 and my third column is: \n{l_phi_temp[:,2]} \nand four columns is:\n {l_phi_temp[:,3]}")
                # elif rank == 1:
                #     print(f"\ncounter is: {counter} Im 1 and my first column is:\n {l_phi_temp[:,0]} \nand second columns is: \n{l_phi_temp[:,1]}")



            ###--- we gather the result ---###
            sendbuf = l_phi[1:-1, 1:-1]
            all_phi_v = comm.gather(sendbuf, root = 0)
            
            all_phi = None

            if rank ==0:
                # we distribute the results in the matrix accordingly
                i = 0
                all_phi = np.zeros((NUM_OF_POINTS,NUM_OF_POINTS))
                for each_rank in range(size):
                    all_phi[1:-1, i_m[each_rank][0]+1:i_m[each_rank][1]] = all_phi_v[i]
                    i += 1
                        
            
            return all_phi


    #####----- helping methods
    def how_many_columns(self):
        """
            here we determine how many columns will store my processor
        """
        n = [0 for i in range(size)]
        for column in range(1,NUM_OF_POINTS - 1):
            c = column % size
            #print(f"n : {c}")
            n[c] += 1

        return n

    def index_map(self, n_of_pro):
        """
            here we determine the index of every processor in the final solution
            n_of_pro: how many columns each processor stores
        """
        map_of_indices = {}
        for i in range(len(n_of_pro)):
            if i == 0:
                map_of_indices[i] = (i, n_of_pro[i]+1)
            else:
                map_of_indices[i] = (map_of_indices[i-1][1]-1, map_of_indices[i-1][1] + n_of_pro[i])
        return map_of_indices

    def source_generator(self):
        s = self.numerical_div(self.vel_field_x, self.vel_field_y)  # source function of poisson equation
        return s




############ testing of the script
if __name__ == "__main__":
    o = Operator()
    print(o.X)
    # vel_field_x = vec_funct_in_x(o.X, o.Y)
    # vel_field_y = vec_funct_in_y(o.X, o.Y)
    # o.numerical_partial_deriv_of_x(vel_field_x)
    # o.numerical_partial_deriv_of_y(vel_field_y)