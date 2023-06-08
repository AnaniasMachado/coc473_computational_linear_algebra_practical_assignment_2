import numpy as np

__all__ = ["NonlinearSystemSolver"]

class NonlinearSystemSolver:
    def __init__(self, F, J):
        """
        Constructor for the NonlinearSystemSolver class.

        Args:
            F (function): Function representing the system of equations F(x) = 0.
            J (function): Function that calculates the Jacobian matrix of the system J(x).
        """
        self.F = F  
        self.J = J  

    def newton(self, x0, tol=1e-4, max_iter=1000):
        """
        Solve the system of equations using Newton's method.

        Args:
            x0 (array): Initial guess for the solution.
            tol (float, optional): Tolerance for convergence. Defaults to 1e-4.
            max_iter (int, optional): Maximum number of iterations. Defaults to 1000.

        Returns:
            tuple: Tuple containing the solution of the system of equations and the number of iterations.
        """
        x = x0.copy()
        for iter in range(max_iter):
            try:
                delta_x = np.linalg.solve(self.J(x), -self.F(x))
            except np.linalg.LinAlgError:
                raise ValueError("Jacobian matrix is not invertible.")
            x += delta_x
            if np.linalg.norm(delta_x)/np.linalg.norm(x) < tol:
                break
        return x, iter+1

    def broyden(self, x0, tol=1e-4, max_iter=1000):
        """
        Solve the system of equations using Broyden's method.

        Args:
            x0 (array): Initial guess for the solution.
            tol (float, optional): Tolerance for convergence. Defaults to 1e-4.
            max_iter (int, optional): Maximum number of iterations. Defaults to 1000.

        Returns:
            tuple: Tuple containing the solution of the system of equations and the number of iterations.
        """
        x = x0.copy()
        J = self.J(x0) # Initial guess for the Jacobian matrix.
        for iter in range(max_iter):
            try:
                delta_x = np.linalg.solve(J, -self.F(x))
            except np.linalg.LinAlgError:
                raise ValueError("Jacobian matrix is not invertible.")
            x += delta_x
            if np.linalg.norm(delta_x)/np.linalg.norm(x) < tol:
                break
            else:
                delta_F = self.F(x) - self.F(x - delta_x)
                J += np.outer((delta_F - J @ delta_x), delta_x) / np.linalg.norm(delta_x)**2
        return x, iter+1

if __name__ == "__main__":
    # Example usage -> Exercise 3 from Task 01

    # System of equations F(x) = 0
    def F(x):
        f1 = 16*x[0]**4 + 16**x[1]**4 + x[2]**4 - 16
        f2 = x[0]**2 + x[1]**2 + x[2]**2 - 3
        f3 = x[0]**3 - x[1] + x[2] - 1
        return np.array([f1, f2, f3])

    # Function that calculates the Jacobian matrix of the system J(x)
    def J(x):
        j11 = 64*x[0]**3
        j12 = 64*x[1]**3
        j13 = 4*x[2]**3
        j21 = 2*x[0]
        j22 = 2*x[1]
        j23 = 2*x[2]
        j31 = 3*x[0]**2
        j32 = -1
        j33 = 1
        return np.array([[j11, j12, j13], [j21, j22, j23], [j31, j32, j33]])

    # Initial guess
    x0 = np.array([1, 1, 1], dtype=np.float64)

    solver = NonlinearSystemSolver(F, J)

    # Newton's method
    solution_newton, iter_newton = solver.newton(x0)
    print(f"\nSolution using Newton's method: {solution_newton}")
    print(f"Number of iterations: {iter_newton}")

    # Broyden's method
    solution_broyden, iter_broyden = solver.broyden(x0)
    print(f"\nSolution using Broyden's method: {solution_broyden}")
    print(f"Number of iterations: {iter_broyden}")
