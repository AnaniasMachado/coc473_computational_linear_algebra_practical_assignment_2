import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import g as acceleration_gravity

class DiffEquationSolver:
    def __init__(self, F, t0, y0, dy0):
        """
        Constructor for the DiffEquationSolver class.

        Args:
            F (function): Function representing the second order differential equation F(t) = y''(t) = f(t, y(t), y'(t)).
            t0 (scalar): Scalar value representing initial time.
            y0 (scalar): Scalar value representing initial position.
            dy0 (scalar): Scalar value representing initial velocity.
        """
        self.F = F  
        self.t0 = t0
        self.y0 = y0
        self.dy0 = dy0  
    
    def taylor(self, h, T):
        """
        Solve the second order differential equation using second order Taylor approximation.

        Args:
            h (float): Step size for iterations.
            T (int): Total integration time.

        Returns:
            tuple: Tuple containing the solution of the differential equation at time t0 + T, 
            number of interations and a nested list of generated points for plotting.
        """
        points = [[], []]
        tk = self.t0
        yk = self.y0
        dyk = self.dy0
        N = int(T / h)
        for k in range(1, N+1):
            ddyk = self.F(tk, yk, dyk)
            tk = k * h
            yk = yk + dyk * h + ddyk * h * h / 2
            dyk = dyk + ddyk * h
            points[0].append(tk)
            points[1].append(yk)
        return yk, N, points
    
    def runge_kutta_nystron(self, h, T):
        """
        Solve the second order differential equation using second order Taylor approximation.

        Args:
            h (float): Step size for iterations.
            T (int): Total integration time.

        Returns:
            tuple: Tuple containing the solution of the differential equation at time t0 + T, 
            number of interations and a nested list of generated points for plotting.
        """
        points = [[], []]
        tk = self.t0
        yk = self.y0
        dyk = self.dy0
        N = int(T / h)
        for k in range(1, N+1):
            K1 = h * self.F(tk, yk, dyk) / 2
            Q = h * (dyk + K1 / 2) / 2
            K2 = h * self.F(tk + h / 2, yk + Q, dyk + K1) / 2
            K3 = h * self.F(tk + h / 2, yk + Q, dyk + K2) / 2
            L = h * (dyk + K3)
            K4 = h * self.F(tk + h, yk + L, dyk + 2 * K3) / 2
            tk = k * h
            yk = yk + h * (dyk + (K1 + K2 + K3) / 3)
            dyk = dyk + (K1 + 2 * K2 + 2 * K3 + K4) / 3
            points[0].append(tk)
            points[1].append(yk)
        return yk, N, points

if __name__ == "__main__":
    #
    # Example usage -> Exercise 4 from Task 03
    #

    # Differential equation F(t) = y''(t) = f(t, y(t), y'(t))
    def F(t, y, dy):
        m = 1
        c = 0.2
        k = 1
        w = 0.5
        g = 2 * np.sin(w * t) + np.sin(2 * w * t) + np.cos(3 * w * t)
        G = (g - c * dy - k * y) / m
        return G

    # Initial conditions
    t0 = 0
    y0 = 0
    dy0 = 0

    # Step size and integration time
    h = 0.01
    T = 25

    solver = DiffEquationSolver(F, t0, y0, dy0)

    # Taylor's method
    solution_taylor, iter_taylor, points_taylor = solver.taylor(h, T)
    print(f"\nSolution using Taylor's method: {solution_taylor}")
    print(f"Number of iterations: {iter_taylor}")

    plt.plot(points_taylor[0], points_taylor[1])
    plt.title("Taylor's method solution plot")
    plt.xlabel("t")
    plt.ylabel("y")
    plt.show()

    # Runge-Kutta Nystron's method
    solution_runge_kutta, iter_runge_kutta, points_runge_kutta = solver.runge_kutta_nystron(h, T)
    print(f"\nSolution using Runge-Kutta Nystron's method: {solution_runge_kutta}")
    print(f"Number of iterations: {iter_runge_kutta}")

    plt.plot(points_runge_kutta[0], points_runge_kutta[1])
    plt.title("Runge-Kutta Nystron's method solution plot")
    plt.xlabel("t")
    plt.ylabel("y")
    plt.show()

    #
    # Example usage -> Exercise 5 from Task 03
    #

    # Differential equation F(t) = y''(t) = f(t, y(t), y'(t))
    def F(t, y, dy):
        kd = 1
        G = (-1) * acceleration_gravity - kd * dy * np.abs(dy)
        return G

    print(acceleration_gravity)

    # Initial conditions
    t0 = 0
    y0 = 0
    dy0 = 0

    # Step size and integration time
    h = 0.01
    T = 20

    solver = DiffEquationSolver(F, t0, y0, dy0)

    # Taylor's method
    solution_taylor, iter_taylor, points_taylor = solver.taylor(h, T)
    print(f"\nSolution using Taylor's method: {solution_taylor}")
    print(f"Number of iterations: {iter_taylor}")

    plt.plot(points_taylor[0], points_taylor[1])
    plt.title("Taylor's method solution plot")
    plt.xlabel("t")
    plt.ylabel("y")
    plt.show()

    # Runge-Kutta Nystron's method
    solution_runge_kutta, iter_runge_kutta, points_runge_kutta = solver.runge_kutta_nystron(h, T)
    print(f"\nSolution using Runge-Kutta Nystron's method: {solution_runge_kutta}")
    print(f"Number of iterations: {iter_runge_kutta}")

    plt.plot(points_runge_kutta[0], points_runge_kutta[1])
    plt.title("Runge-Kutta Nystron's method solution plot")
    plt.xlabel("t")
    plt.ylabel("y")
    plt.show()