import numpy as np
from scipy.special import binom
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import NonlinearConstraint
from scipy.signal.windows import gaussian


class Parseval:
    """
    Creates multidimensional Parseval frame or regular frame filterbanks. Allows for 
    difference filters of first and/or second order.
    """
    def __init__(self, shape, low_pass_kernel='gauss', 
                 first_order=False, second_order=False, type_of_out_frame='pframe'):
        """
        :param shape: list: the desired shape of a single filter in the filter bank
        :param low_pass_kernel: either 'gauss' or 'bspline'. Beginning with a gaussian
        kernel or a bspline kernel
        :param first_order: boolean. Whether you want first order central difference 
        filters directed at all available orientations in a grid of size "shape" in the filterbank
        :param second_order: boolean. Whether you want second order central difference 
        filters directed at all available orientations in a grid of size "shape" in the filterbank
        :param type_of_out_frame: either 'pframe' or 'frame'. 
        """

        self.shape = shape
        self.kernel = low_pass_kernel
        self.first_order = first_order
        self.second_order = second_order
        self.type_of_out_frame = type_of_out_frame
        
        
    def low_pass(self):
        """
        creates a multi-dimensional tensor product kernel 
        :param shape: list: desired kernel shape
        :param kernel str: either 'gauss' or 'bspline'
        :return: a vectorized version of the multi dimensional kernel  
        """

        def gauss(n):
            """
            creates a 1d gaussian low pass filter with unit
            standard deviation.
            :param n: number of desired filter components
            """
            kernel = gaussian(n, 1).reshape(1, n)
            return kernel/np.sum(kernel)

        def bspline(n):
            """
            creates a 1d b-spline low pass filter
            """
            all_rows = list(list(int(binom(row, k)) for k in range(row + 1)) for row in range(n))
            return np.array(all_rows[-1]).reshape(1, n)/np.sum(all_rows[-1])

        assert np.all(np.array(self.shape) > 1), "All components of the shape list must be greater than 1."

        lpf = 1
        for i in range(len(self.shape)):
            lpf = np.kron(lpf, vars()[self.kernel](self.shape[i]))

        return lpf  
    
    def order1_kernels(self):
        """
        creates a template array from all available orientation 
        vectors starting at the origin on an (rows x columns) grid. The 
        resulting row vectors are first order differences.
        """
        dim_of_vec = np.prod(self.shape)
        left_id = np.eye(int(dim_of_vec/2))
        right_id = np.fliplr(-left_id)

        return np.concatenate((left_id, np.zeros((int(dim_of_vec/2), 1)), right_id), axis=1).astype(np.int8) 
    
    def order2_kernels(self):
        """
        creates a template array from all available orientation 
        vectors starting at the origin on an (rows x columns) grid. The resulting
        row vectors are second order differences.
        """
        dim_of_vec = np.prod(self.shape)
        assert dim_of_vec % 2 != 0,  "for 2nd order difference filters make sure product of the "\
                                     "elements in shape is an odd number."

        left_id = np.eye(int(dim_of_vec/2))
        right_id = np.fliplr(left_id)

        return np.concatenate((-left_id, 2*np.ones((int(dim_of_vec/2), 1)), -right_id), axis=1).astype(np.int8)
    
    def optimization(self):
        """
        builds the final version going into pframe for the 
        parseval frame construction.
        """
        candidate_arrays = list()
        candidate_arrays.append(np.sqrt(self.low_pass()))
        
        if self.first_order:
            candidate_arrays.append(self.order1_kernels())
            
        if self.second_order:
            candidate_arrays.append(self.order2_kernels())
        
        def normalize(A):
            """
            normalizes the rows of array A. Here it's used for numerical stability
            """
            rsums = np.sqrt((A**2).sum(axis=1, keepdims=True))
            return A / rsums
        
        # get the final template array for optimization
        array = normalize(np.concatenate(candidate_arrays, axis=0))
        # get the rank of the template array
        dim = np.linalg.matrix_rank(array)
        
        def objective(x):
            scaled = np.matmul(np.diag(np.insert(x, 0, 1)), array)
            return -np.linalg.norm(scaled, ord=2)

        def constraint(x):
            scaled = np.matmul(np.diag(np.insert(x, 0, 1)), array)
            return 1 - np.linalg.norm(scaled, ord=2)
        
        ineq_constraint = {'type': 'ineq', 'fun': constraint}
        x0 = np.ones(dim - 1)
        bounds = Bounds([0]*(dim - 1), [1]*(dim - 1))
        res = minimize(objective, x0, 
                       method='SLSQP',
                       constraints=[ineq_constraint], 
                       options={'disp': False, 'maxiter': 3000},
                       bounds=bounds)
        
        return np.matmul(np.diag(np.insert(res.x, 0, 1)), array)
        
    def pframe(self):
        """
        creates a Parseval frame filterbank comprising the filters
        given by "optimizatio" but also some additional high pass filters 
        needed for the completion of the parseval frame.
        """
        # get the singular values of the optimized array
        eps = 1e-10    
        array = self.optimization()
        _, sigma1, vstar = np.linalg.svd(array)
        
        if sigma1[0] <= 1 + 1e-6:
            sigma1[0] = 1
        else:
            assert sigma1[0] <= 1 + 1e-6,  "Optimization did not converge to a numerically " \
                                           "stable solution"
        
        # prepare the array basis vactors for the high-pass filters
        sigma2 = np.ones(vstar.shape[0])
        sigma2[0:sigma1.shape[0]] = 1 - sigma1**2
        sigma2[np.abs(sigma2) < eps] = 0
        sigma2 = np.diag(np.sqrt(sigma2))
        
        # get the high-pass filters
        high_pass = np.matmul(sigma2, vstar)
        parseval = np.concatenate((array, high_pass), axis=0)
        parseval = parseval[~np.all(np.abs(parseval) < eps, axis=1)]
        parseval[np.abs(parseval) < eps] = 0
        parseval = parseval.reshape(((parseval.shape[0],) + tuple(self.shape)))
        
        # return the filterbank based on whether you want a parseval frame or a frame
        np.set_printoptions(precision=7, suppress=True)
        
        fname = ''
        for i in range(len(self.shape)):
            fname += str(self.shape[i])
            
        if self.type_of_out_frame == 'pframe':
#             np.save('pframe' + fname + '.npy', parseval)
            return parseval
        elif self.type_of_out_frame == 'frame':
#             np.save('frame' + fname + '.npy', parseval[0:np.prod(self.shape)])
            return parseval[0:np.prod(self.shape)]