import numpy as np

__all__ = ["Integrator"]

class Integrator:
    def __init__(self, F, a, b):
        """
        Constructor for the Integrator class.

        Args:
            F (function): Function to be integrated.
            a (float): Lower limit of the integration interval.
            b (float): Upper limit of the integration interval.
        """
        self.F = F
        self.a = a
        self.b = b

    def _simpson_rule(self, a, b, y):
        """
        Calculates the integral using Simpson's rule (3 integration points).

        Args:
            a (float): Lower limit of the integration interval.
            b (float): Upper limit of the integration interval.
            y (array-like): Array with function evaluation of each integration point.

        Returns:
            float: Estimated value of the integral.
        """
        L = b - a

        # Applying Simpson weights
        integral = y[0] * L/6 + y[1] * (2*L/3) + y[2] * L/6
        return integral
    
    def adaptative_integration(self, tol =1e-4, max_iter=1000):
        """
        Performs a self-adaptive numerical integration using Simpson's rule.

        Args:
            tol (float, optional): Tolerance for stopping criteria. Defaults to 1e-4.
            max_iter (int, optional): Maximum number of iterations. Defaults to 1000.

        Returns:
            float: The estimated value of the integral.
        """
        n = 3
        x = np.linspace(self.a, self.b, n)  # Defining three integration points
        y = self.F(x)

        I_previous = 0.0
        I_current = self._simpson_rule(self.a, self.b, y)  # Initial integral estimative
        iter = 1

        print("Iteration: 1")
        print("Number of integration points:", n)
        print("Integral value:", I_current)
        print("--------------------------")

        # Stores function calculations previously done, so as to avoid needless recalculations
        evaluated = dict()
        for i in range(len(x)): evaluated[x[i]] = y[i]

        while abs((I_current - I_previous) / I_current) > tol:
            I_previous = I_current
            I_current = 0.0

            n = (n-1)*2 + 1
            x = np.linspace(self.a, self.b, n)  # Defining new integration points

            # Evaluating function only at the new points (midpoints)
            for i in range(1, len(x), 2):
                new_evaluation = self.F(x[i]) # midpoint
                evaluated[x[i]] = new_evaluation
                y = [evaluated[x[i-1]], new_evaluation, evaluated[x[i+1]]]
                I_current += self._simpson_rule(x[i-1], x[i+1], y)

            iter += 1
            if iter >= max_iter: break

            print("Iteration:", iter)
            print("Number of integration points:", n)
            print("Integral value:", I_current)
            print("--------------------------")

        print("Convergence reached after", iter, "iterations.")
        print("Final number of integration points:", n)
        return I_current
    
    def gauss_legendre_quad(self):
        """
        Performs the Gauss-Legendre Quadrature integral approximation using 11 integration points.

        Returns:
            float: The estimated value of the integral.
        """
        weights = [0.2729250867779006,	
                    0.2628045445102467,
                    0.2628045445102467,	
                    0.2331937645919905,	
                    0.2331937645919905,	
                    0.1862902109277343,	
                    0.1862902109277343,	
                    0.1255803694649046,	
                    0.1255803694649046,	
                    0.0556685671161737,	
                    0.0556685671161737]

        abscissae = [0.0000000000000000,
                    -0.2695431559523450,
                    0.2695431559523450,
                    -0.5190961292068118,
                    0.5190961292068118,
                    -0.7301520055740494,
                    0.7301520055740494,
                    -0.8870625997680953,
                    0.8870625997680953,
                    -0.9782286581460570,
                    0.978228658146057]

        # Transform limits of integration
        g = lambda z: (self.b - self.a) * z / 2 + (self.b + self.a) / 2

        integral = 0.0
        for i in range(11):
            integral += weights[i] * self.F(g(abscissae[i]))

        integral *= (self.b - self.a) / 2

        return integral


if __name__ == "__main__":
    #
    # Example usage -> Exercise 6 from Task 02
    #

    def S_sigma(x):
        return 2.0 * Rao(x)**2

    def Rao(x):
        denom = (1 - x ** 2) ** 2 + (2 * 0.05 * x) ** 2
        return 1 / np.sqrt(denom) 
    
    # Solve m_0 integration
    print(f"\nSolving Exercise 6 for m_0:\n")

    solver_m0 = Integrator(F = S_sigma, a = 0, b = 10)
    result_m0_1 = solver_m0.adaptative_integration()
    result_m0_2 = solver_m0.gauss_legendre_quad()
    print(f"\nFor comparison:\nSelf-adaptative integration result: {result_m0_1}")
    print(f"Gauss-Legendre Quadrature result: {result_m0_2}\n\n")

    # Solve m_2 integration
    print(f"Solving Exercise 6 for m_2:\n")

    solver_m2 = Integrator(F = lambda x: (x ** 2) * S_sigma(x), a = 0, b = 10)
    result_m2_1 = solver_m2.adaptative_integration()
    result_m2_2 = solver_m2.gauss_legendre_quad()
    print(f"\nFor comparison:\nSelf-adaptative integration result: {result_m2_1}")
    print(f"Gauss-Legendre Quadrature result: {result_m2_2}\n\n")

    #
    # Example usage -> Exercise 7 from Task 02
    #

    def S_eta(x):
        val = (x ** 4) * (5.0 ** 4)
        return 4 * (np.pi ** 3) * (3.0 ** 2) * np.exp(-16 * (np.pi) ** 3 / val) / (x * val)

    def S_sigma_2(x):
        return S_eta(x) * Rao(x)**2
    
    # Solve m_0 integration
    print(f"Solving Exercise 7 for m_0:\n")

    solver_m0 = Integrator(F = S_sigma_2, a = 0.1, b = 10)
    result_m0_1 = solver_m0.adaptative_integration()
    result_m0_2 = solver_m0.gauss_legendre_quad()
    print(f"\nFor comparison:\nSelf-adaptative integration result: {result_m0_1}")
    print(f"Gauss-Legendre Quadrature result: {result_m0_2}\n\n")

    # Solve m_2 integration
    print(f"Solving Exercise 7 for m_2:\n")

    solver_m2 = Integrator(F = lambda x: (x ** 2) * S_sigma_2(x), a = 0.1, b = 10)
    result_m2_1 = solver_m2.adaptative_integration()
    result_m2_2 = solver_m2.gauss_legendre_quad()
    print(f"\nFor comparison:\nSelf-adaptative integration result: {result_m2_1}")
    print(f"Gauss-Legendre Quadrature result: {result_m2_2}\n\n")