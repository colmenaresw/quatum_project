"""
    class containing the implementation of the operators

"""

import numpy as np
import copy

# constanst declarations
NUM_OF_POINTS = 20  # number of points for the discretize grid
D_SIZE = 1  # the size of the domain
NUM_ITERATIONS = 400




#vec_funct = lambda array_x, array_y :  np.array([array_y, array_x])  # test function f(x,y) = [y, x]
#vec_funct = lambda array_x, array_y  :  np.array([array_x, array_y])  # test function f(x,y) = [x, y]
#vec_funct = lambda array_x, array_y  :  np.array([array_x, array_y])  # test function [x + y, x + y]
#vec_funct_in_x = lambda grid_x, grid_y : grid_x**2

def vec_funct_in_x(grid_x, grid_y):
    norm = grid_x**2+grid_y**2 
    alpha = (norm + 1)**-1
    beta = np.exp(-norm*0.5)

    return grid_y * alpha + grid_x * beta

def vec_funct_in_y(grid_x, grid_y):
    norm = grid_x**2+grid_y**2 
    alpha = (norm + 1)**-1
    beta = np.exp(-norm*0.5)

    return -grid_x * alpha + grid_y * beta


def vec_funct_in_x_ans(grid_x, grid_y):
    norm = grid_x**2+grid_y**2 
    alpha = (norm + 1)**-1

    return grid_y * alpha

def vec_funct_in_y_ans(grid_x, grid_y):
    norm = grid_x**2+grid_y**2 
    alpha = (norm + 1)**-1
    return -grid_x * alpha

# def vec_funct_in_x(grid_x, grid_y):

#     return grid_x

# def vec_funct_in_y(grid_x, grid_y):
#     return grid_y

# def vec_funct(array_x, array_y ):
#     """ return a vector """
#     vec1 = np.array([array_y, -array_x])
#     vec2 = np.array([array_x, array_y])
#     first_term = 1/(np.linalg.norm(vec2**2) * 1)
#     second_term = np.exp((-np.linalg.norm(vec2)**2)/2)
#     ans = vec1*first_term + vec2*second_term
#     return ans


class Operator:
    def __init__(self) -> None:
        #self.delta = 0.05
        self.delta = D_SIZE / (NUM_OF_POINTS - 1)  # size of the grid
        self.x_size = np.linspace(0, D_SIZE, NUM_OF_POINTS)
        self.y_size = np.linspace(0, D_SIZE, NUM_OF_POINTS)

        # two dimensional grid
        self.X, self.Y = np.meshgrid(self.x_size, self.y_size)
        self.vel_field_x = vec_funct_in_x(self.X, self.Y)
        self.vel_field_y = vec_funct_in_y(self.X, self.Y)

    def numerical_partial_deriv_of_x(self, vector_field_fx):  # central difference squeme for x
        delta_x = np.zeros_like(vector_field_fx)

        # [1:-1, 1:-1]  all the points that are not in the boundaries
        delta_x[1:-1, 1:-1] = (vector_field_fx[1:-1, 2:  ] - 
                               vector_field_fx[1:-1, 0:-2] 
                          ) / (                  
                           2*self.delta )
        return delta_x

    def numerical_partial_deriv_of_y(self, vector_field_fy):  # central difference squeme for y
        delta_y = np.zeros_like(vector_field_fy)

        # [1:-1, 1:-1]  all the points that are not in the boundaries
        delta_y[1:-1, 1:-1] = (vector_field_fy[2:  , 1:-1] - 
                               vector_field_fy[0:-2, 1:-1] 
                          ) / (                  
                           2*self.delta )
        return delta_y


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
        
        # boundaries
        # partial_x[:,1] = scalar_field[:,1]
        # partial_x[:,-1] = scalar_field[:,-1]

        # partial_y[1,:] = scalar_field[1,:]
        # partial_y[-1,:] = scalar_field[-1,:]

        return [partial_x,  partial_y]
        

    # def jacobi_iterator(self, vector):
    #     size_of_v = vector[0][0].size
    #     grid = vector[0][0]
    #     #y_grid = vector[0][0]
    #     jacobi_grid = []
    #     # we iterate over every x,y coordinate and create the grid
    #     for i in range(size_of_v):
    #         jacobi_grid.append([])
    #         for j in range(size_of_v):
    #             jacobi_grid[i].append(0)

    #     jacobi_grid_ = copy.deepcopy(jacobi_grid)

    #     # we try to compute for every grid point the new value
    #     # i : row and j : column
    #     while True:

    #         for i in range(size_of_v):
    #             for j in range(size_of_v):
    #                 if (i != 0 and i != size_of_v - 1) and (j != 0 and j != size_of_v - 1):
    #                     vector_val = [grid[j],grid[i]]  # (x,y) coordinates

    #                     # for vertical values
    #                     if i == 0:
    #                         up = 0
    #                     else:
    #                         y_pos = i - 1
    #                         x_pos = j

    #                         up = jacobi_grid[y_pos][x_pos]

    #                     if i == size_of_v - 1:
    #                         down = 0
    #                     else:
    #                         y_pos = i + 1
    #                         x_pos = j
    #                         down = jacobi_grid[y_pos][x_pos]
                        

    #                     # for horizontal values
    #                     if j ==  size_of_v - 1:
    #                         right = 0
    #                     else:
    #                         y_pos = i
    #                         x_pos = j + 1
    #                         right = jacobi_grid[y_pos][x_pos]

    #                     if j == 0:
    #                         left = 0
    #                     else:
    #                         y_pos = i
    #                         x_pos = j - 1
    #                         left = jacobi_grid[y_pos][x_pos]

    #                     rho = self.numerical_divergence(vec_funct(vector_val[0],vector_val[1]))
                                            
    #                     jacobi_grid_[i][j] = (rho*self.delta**2 + \
    #                                           up + down + left + right)*0.25

    #         e1 = np.linalg.norm(jacobi_grid_, ord=np.inf)  # infinity norm
    #         e0 = np.linalg.norm(jacobi_grid, ord=np.inf)
    #         error = abs(e0-e1)
    #         if error < 2e-4:
    #             break
    #         jacobi_grid = copy.deepcopy(jacobi_grid_)
    #     return jacobi_grid

    def jacobi_iterator(self):
        phi = np.zeros_like(self.vel_field_x)
        rho = self.numerical_div(self.vel_field_x, self.vel_field_y)

        while True:
            phi_temp = copy.deepcopy(phi)
            for y_dir in range(1, NUM_OF_POINTS - 1, 1):
                for x_dir in range(1, NUM_OF_POINTS - 1, 1):
                    up = phi_temp[y_dir-1, x_dir]
                    down = phi_temp[y_dir + 1, x_dir]
                    left = phi_temp[y_dir, x_dir - 1]
                    right = phi_temp[y_dir, x_dir + 1]
                    center = rho[y_dir, x_dir]
                    phi[y_dir, x_dir] = (-center * self.delta**2  +
                                        down + 
                                        up +
                                        left +
                                        right                                    
                                        ) * 0.25
            e1 = np.linalg.norm(phi, ord=np.inf)  # infinity norm
            e0 = np.linalg.norm(phi_temp, ord=np.inf)
            error = abs(e0-e1)
            if error < 2e-10:
                break

        return phi

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

    # def numerical_gradient_phi(self, grid):

    #     size = len(grid)
    #     ans_grid_x  = copy.deepcopy(grid)  # we create a copy of the original grid
    #     ans_grid_y = copy.deepcopy(grid) 
        
    #     for i in range(1,size-1): # we iterate over the rows (vertical direction)
    #         for j in range(1,size-1):  # we iterate over the columns (horizontal direction)

    #             # for partial of x we check left and right
    #             # we begin with left and check boundaries
    #             left = grid[i][j-1]
    #             right = grid[i][j+1]

    #             partial_x = (right - left)/2*self.delta

    #             # now we check up and down
    #             up = grid[i+1][j]
    #             down = grid[i-1][j]

    #             partial_y = (up - down)/2*self.delta

    #             ans_grid_x[i][j] = partial_x
    #             ans_grid_y[i][j] = partial_y

    #             print("done")

    #     return [ans_grid_x,ans_grid_y]


if __name__ == "__main__":
    o = Operator()
    print(o.X)
    vel_field_x = vec_funct_in_x(o.X, o.Y)
    vel_field_y = vec_funct_in_y(o.X, o.Y)
    o.numerical_partial_deriv_of_x(vel_field_x)
    o.numerical_partial_deriv_of_y(vel_field_y)