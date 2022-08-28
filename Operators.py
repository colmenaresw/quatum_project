"""
    class containing the implementation of the operators

"""

import numpy as np
import copy

############ constanst declarations
NUM_OF_POINTS = 10  # number of points for the discretize grid
D_SIZE = 1  # the size of the domain


############ class
class Operator:
    def __init__(self, vec_funct_in_x, vec_funct_in_y) -> None:
  
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

        for j in range(NUM_OF_POINTS):
            for i in range(NUM_OF_POINTS):
                if i == NUM_OF_POINTS - 1:  # backward difference
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

        for i in range(NUM_OF_POINTS):
            for j in range(NUM_OF_POINTS):
                if j == NUM_OF_POINTS - 1:  # backward difference
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
    
    # def numerical_partial_deriv_of_x(self, vector_field_fx):  # central difference squeme for x
    #     delta_x = np.zeros_like(vector_field_fx)

    #     # [1:-1, 1:-1]  all the points that are not in the boundaries
    #     delta_x[1:-1, 1:-1] = (vector_field_fx[1:-1, 2:  ] - 
    #                            vector_field_fx[1:-1, 0:-2] 
    #                       ) / (                  
    #                        2*self.delta )
    #     return delta_x

    # def numerical_partial_deriv_of_y(self, vector_field_fy):  # central difference squeme for y
    #     delta_y = np.zeros_like(vector_field_fy)

    #     # [1:-1, 1:-1]  all the points that are not in the boundaries
    #     delta_y[1:-1, 1:-1] = (vector_field_fy[2:  , 1:-1] - 
    #                            vector_field_fy[0:-2, 1:-1] 
    #                       ) / (                  
    #                        2*self.delta )
    #     return delta_y

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
        phi = np.zeros_like(self.vel_field_x)
        phi_temp = np.zeros_like(self.vel_field_x)
        rho = self.numerical_div(self.vel_field_x, self.vel_field_y)
        counter = 0

        while True:
            phi = np.zeros_like(phi_temp)
            for y_dir in range(1, NUM_OF_POINTS - 1, 1):
                for x_dir in range(1, NUM_OF_POINTS - 1, 1):
                    up = phi_temp[y_dir-1, x_dir]
                    down = phi_temp[y_dir + 1, x_dir]
                    left = phi_temp[y_dir, x_dir - 1]
                    right = phi_temp[y_dir, x_dir + 1]
                    center = rho[y_dir, x_dir]
                    phi[y_dir, x_dir] = (center * self.delta**2  +
                                        down + 
                                        up +
                                        left +
                                        right                                    
                                        ) * 0.25
            
        

            e1 = np.linalg.norm(phi, ord=np.inf)  # infinity norm
            e0 = np.linalg.norm(phi_temp, ord=np.inf)
            error = abs(e0-e1)
            if error < 2e-10 or counter > 10000:
                break

            counter += 1

            phi_temp = copy.deepcopy(phi)

        # Neumann boundary conditions
        # phi[:, -1] = phi[:, -2]  # condition at the right
        # phi[0, :] = phi[1, :]  # condition at the top
        # phi[:, 0] = phi[:, 1] # condition at the left
        # phi[-1, :] = phi[-2, :]  # condition at the bottom

        return phi




############ testing of the script
if __name__ == "__main__":
    o = Operator()
    print(o.X)
    vel_field_x = vec_funct_in_x(o.X, o.Y)
    vel_field_y = vec_funct_in_y(o.X, o.Y)
    o.numerical_partial_deriv_of_x(vel_field_x)
    o.numerical_partial_deriv_of_y(vel_field_y)