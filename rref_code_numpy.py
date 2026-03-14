import numpy as np
import sympy as sp
#claude code
def rref(matrix, tol=1e-10):
    """
    Compute the Reduced Row Echelon Form (RREF) of a matrix.
    
    Parameters:
    -----------
    matrix : numpy.ndarray
        Input matrix (will be copied, not modified in place)
    tol : float, optional
        Tolerance for determining if a value is zero (default: 1e-10)
    
    Returns:
    --------
    numpy.ndarray
        Matrix in reduced row echelon form
    """
    # Work with a copy and convert to float
    A = matrix.astype(float).copy()
    rows, cols = A.shape
    
    current_row = 0
    
    for col in range(cols):
        # Find pivot - the maximum absolute value in current column
        # from current_row downward
        pivot_row = current_row + np.argmax(np.abs(A[current_row:, col]))
        
        # If pivot is effectively zero, move to next column
        if np.abs(A[pivot_row, col]) < tol:
            continue
        
        # Swap rows if needed
        if pivot_row != current_row:
            A[[current_row, pivot_row]] = A[[pivot_row, current_row]]
        
        # Scale pivot row to make pivot = 1
        A[current_row] = A[current_row] / A[current_row, col]
        
        # Eliminate all other entries in this column
        for row in range(rows):
            if row != current_row:
                A[row] = A[row] - A[row, col] * A[current_row]
        
        current_row += 1
        
        # If we've processed all rows, we're done
        if current_row >= rows:
            break
    
    # Clean up near-zero values
    A[np.abs(A) < tol] = 0
    
    return A


# Example usage
if __name__ == "__main__":
    # Example 1: Simple 3x3 system
    # M1 = np.array([
    #     [1, 2, 3],
    #     [4, 5, 6],
    #     [7, 8, 9]
    # ])
    
    # print("Original matrix:")
    # print(M1)
    # print("\nRREF:")
    # print(rref(M1))
    
    # # Example 2: Augmented matrix for solving system of equations
    # # System: x + 2y = 5, 3x + 4y = 11
    # M2 = np.array([
    #     [1, 2, 5],
    #     [3, 4, 11]
    # ])
    
    # print("\n\nAugmented matrix:")
    # print(M2)
    # print("\nRREF (solution in last column):")
    # print(rref(M2))
    
    # Example 3: 4x5 matrix
    M3 = np.array([1,3,1,3,0,1,1,0,-3,0,6,-1,3,4,-2,1,2,0,-4,2]).reshape(5,4)
    
    # Cubing M3 using the built-in power function
    M3_cubed_alt = np.linalg.matrix_power(M3, 3)

    print("\nM3 Cubed (matrix_power):")
    print(M3_cubed_alt)
    print("\n\nLarger matrix:")
    print(M3)
    print("\nRREF:")
    print(rref(M3))