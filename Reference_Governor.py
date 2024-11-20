'''
High distance= breaks
create small steps
find maximum feasible steps and as less stops as possible
The SRG algorithm adjusts the reference trajectory  by computing  κt, the maximum feasible step from the current reference 
The system iterates over time to update the reference trajectory and ensure constraint satisfaction at each step.
The SRG is formulated as a linear programming (LP) problem with constraints based on the system dynamics.

'''
from Constraints import * 

import numpy as np
class ReferenceGovernor:
    def __init__(self):
        pass
    
    def rg( Hx, Hv, h, ref, vk, current_state:xx):
        x0 = np.zeros(12,1)  #Initial state
        ts = 0.01 # Sampling time
        ref = 3 * [1, 1, 1, 0] # Desired point
        vk = 0.01 * ref # Initial feasible v0
        ell_star = 1000
        tt = np.arange(0, ts, 10) #[0:ts:10] #Time interval for the continuous-time system
        N = len(tt) #Number of time steps for Euler’s 
    
        M= Hx.shape[0]
        k= np.zeros(M)
        for j in range(M):
            alpha
            if alpha > 0:
                k[j] = beta / alpha
            elif alpha < 0:
                k[j] = 0
            else:
                k[j] = 1

        # Find the minimum value of k
        min_k = np.min(k)
        
        pass
    def Hx(self, S, Ad, l_star):
        ''''
        Computes the Hx matrix [Sx | SAx | ... | SA^ℓ*x].
        
        Parameters:
        S : Matrix S (constraint matrix)
        Ad : Matrix Ad (discrete system matrix)
        l_star (int): The number of future steps (iterations)
        
        Returns:
            The resulting Hx matrix (a vertically stacked matrix of constraints over future steps)
        '''
        # Initialize the list to store each hrow (for each l from 0 to l_star)
        hrows = []

        # Compute each term in the series S * A^l (for each l from 0 to l_star)
        for l in range(0, l_star + 1):
            # Compute the matrix A^l
            Ad_Power = np.linalg.matrix_power(Ad, l)
            
            # Compute the hrow for the current step
            hrow = S @ Ad_Power
            
            # Append the computed hrow to the list
            hrows.append(hrow)
        
        # Stack all hrows vertically to form the full Hx matrix
        Hx = np.vstack(hrows)
        
        return Hx


    def Hv(self, S, Ad, Bd, l_star):
        ''''
        Computes the Hv matrix [0 | SB | ... | S(I-A)^-1 (I-A^ℓ)*B].
        
        Parameters:
        S : Matrix S (constraint matrix)
        Ad : Matrix Ad (discrete system matrix)
        Bd : Matrix Bd (input matrix)
        l_star (int): The number of future steps (iterations)
        
        Returns:
            The resulting Hv matrix (a vertically stacked matrix of control constraints over future steps)
        '''
        # Initialize the list to store each hvrow (for each l from 0 to l_star)
        hvrows = []
        
        # Compute each term in the series for Hv (for each l from 0 to l_star)
        for l in range(0, l_star + 1):
            # Compute the identity matrix I
            I = np.eye(Ad.shape[0])
            
            # Compute the matrix A^l
            Ad_Power = np.linalg.matrix_power(Ad, l)
            
            # Compute the inverse of (I - A)
            Ad_inverse = np.linalg.inv(I - Ad)
            
            # Compute the difference (I - A^l)
            Ad_Power_Diff = I - Ad_Power
            
            # Compute the hvrow for the current step
            hvrow = S @ Ad_inverse @ Ad_Power_Diff @ Bd
            
            # Append the computed hvrow to the list
            hvrows.append(hvrow)
        
            # Stack all hvrows vertically to form the full Hv matrix
        Hv = np.vstack(hvrows)
            
        return Hv
    def H(self, s, epsilon, l_star):
        """
        Creates a matrix h where:
        - The first `l_star - 1` rows are equal to s
        - The last row is equal to s - epsilon

        Parameters:
        s : numpy array
            The base vector s.
        epsilon : numpy array
            The vector to subtract from s for the last row.
        l_star : int
            The number of rows in the resulting matrix.

        Returns:
        numpy array
            The matrix h with the specified structure.
        """
        # Create the matrix where all rows are initially s
        h = np.tile(s, (l_star - 1, 1))  # Repeat s for (l_star - 1) times

        # Subtract epsilon for the last row
        h = np.vstack([h, s - epsilon])

        return h
    
        
        

    
# Main Script
if __name__ == "__main__":
    l_star =  1000  # Number of future steps
    epsilon= 0.001
  
    #usage
    governor = ReferenceGovernor()

    # Compute Hx and Hv
    Hx_matrix = governor.Hx(S, Ad, l_star)
    Hv_matrix = governor.Hv(S, Ad, Bd, l_star)
    h_matrix= governor.H(s,epsilon, l_star)
   
    