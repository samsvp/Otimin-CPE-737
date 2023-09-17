import numpy as np
import pandas as pd
from autograd import hessian, grad


class MultivariableFunctionEvaluator:
    def __init__(self, expression, variables):
        self.expression = expression
        self.variables = variables
        self.func = self._create_function(expression, variables)

    def _create_function(self, expression, variables):
        def func(*args):
            var_dict = {var: val for var, val in zip(variables, args)}
            return eval(expression, var_dict)
        return func

    def evaluate(self, values):
        try:
            result = self.func(*values)
            return float(result)
        except Exception as e:
            raise ValueError(f"Error: {str(e)}")

    def evaluate_hessian(self, values):
        try:
            # Compute the Hessian matrix using numerical differentiation
            epsilon = 1e-6  # Small value for numerical differentiation
            n = len(values)
            hessian_matrix = np.zeros((n, n))

            for i in range(n):
                for j in range(i, n):
                    # Compute the second partial derivatives
                    x_plus_delta = np.array(values)
                    x_minus_delta = np.array(values)

                    x_plus_delta[i] += epsilon
                    x_plus_delta[j] += epsilon

                    x_minus_delta[i] -= epsilon
                    x_minus_delta[j] -= epsilon

                    f_plus_plus = self.evaluate(x_plus_delta)
                    f_plus_minus = self.evaluate(x_minus_delta)

                    x_plus_delta[i] -= 2 * epsilon
                    x_minus_delta[j] -= 2 * epsilon

                    f_minus_plus = self.evaluate(x_plus_delta)
                    f_minus_minus = self.evaluate(x_minus_delta)

                    hessian_matrix[i, j] = (f_plus_plus - f_plus_minus - f_minus_plus + f_minus_minus) / (4 * epsilon ** 2)
                    hessian_matrix[j, i] = hessian_matrix[i, j]  # The Hessian is symmetric

            return hessian_matrix
        except Exception as e:
            return f"Error computing Hessian: {str(e)}"


class GradientDescentOptimizer(MultivariableFunctionEvaluator):
    def gradient_descent(self, x0, tol=0.1, alpha=1.0, ratio=0.8, c=0.01):
        x_k = np.array(x0)
        num_steps = 0
        step_size = alpha

        while True:
            g_k = grad(self.func)(*x_k)
            if np.abs(g_k).max() < tol:
                break
            num_steps += 1

            fx = self.evaluate(x_k)
            cg = -c * (g_k**2).sum()

            while self.evaluate(x_k - step_size * g_k) > (fx + step_size * cg):
                step_size *= ratio

            x_k -= step_size * g_k

        return x_k, g_k, num_steps


class GoldenSectionSearch(MultivariableFunctionEvaluator):
    def check_pos(self, x1, x2):
        if x2 < x1:
            label = 'right'
        else:
            label = ''
        return label

    def update_interior(self, xl, xu):
        if isinstance(xl, list) and isinstance(xu, list):
            xl = xl[0]
            xu = xu[0]

        d = ((np.sqrt(5) - 1) / 2) * (xu - xl)
        x1 = xl + d
        x2 = xu - d
        return [x1, x2]


    def find_max(self, xl, xu, x1, x2, label):
        fx1 = self.evaluate([x1])
        fx2 = self.evaluate([x2])
        if fx2 > fx1 and label == "right":
            xl = xl
            xu = x1
            new_x = self.update_interior(xl, xu)
            x1 = new_x[0]
            x2 = new_x[1]
            xopt = x2
        else:
            xl = x2
            xu = xu
            new_x = self.update_interior(xl, xu)
            x1 = new_x[0]
            x2 = new_x[1]
            xopt = x1
        return xl, xu, xopt

    def find_min(self, xl, xu, x1, x2, label):
        fx1 = self.evaluate([x1])
        fx2 = self.evaluate([x2])
        if fx2 > fx1 and label == "right":
            xl = x2
            xu = xu
            new_x = self.update_interior(xl, xu)
            x1 = new_x[0]
            x2 = new_x[1]
            xopt = x1
        else:
            xl = xl
            xu = xl
            new_x = self.update_interior(xl, xu)
            x1 = new_x[0]
            x2 = new_x[1]
            xopt = x2
        return xl, xu, xopt

    def golden_search(self, xl, xu, mode, et):
        it = 0
        e = 1
        while e >= et:
            new_x = self.update_interior(xl, xu)
            x1 = new_x[0]
            x2 = new_x[1]
            fx1 = self.evaluate([x1])
            fx2 = self.evaluate([x2])
            label = self.check_pos(x1, x2)
            if mode == 'max':
                new_boundary = self.find_max(xl, xu, x1, x2, label)
            elif mode == 'min':
                new_boundary = self.find_min(xl, xu, x1, x2, label)
            else:
                print('Please define min/max mode')
                break
            xl = new_boundary[0]
            xu = new_boundary[1]
            xopt = new_boundary[2]

            it += 1
            print('Iteration: ', it)
            r = (np.sqrt(5) - 1) / 2
            e = ((1 - r) * (abs((xu[0] - xl[0]) / xopt))) * 100  # Error
            print('Error:', e)


class FibonacciSearch(MultivariableFunctionEvaluator):
    def fibonacci(self, n):
        fn = [0, 1]
        for i in range(2, n + 1):
            fn.append(fn[i - 1] + fn[i - 2])
        return fn

    def fib_search(self, xl, xr, n):
        F = self.fibonacci(n)
        L0 = xr - xl
        ini = L0
        Li = (F[n - 2] / F[n]) * L0
        R = [Li / L0]
        a = [xl]
        b = [xr]
        F1 = [self.evaluate([xl])]
        F2 = [self.evaluate([xr])]

        for i in range(2, n + 1):
            if Li > L0 / 2:
                x1 = xr - Li
                x2 = xl + Li
            else:
                x1 = xl + Li
                x2 = xr - Li

            f1, f2 = self.evaluate([x1]), self.evaluate([x2])

            if f1 < f2:
                xr = x2
                Li = (F[n - i] / F[n - (i - 2)]) * L0
            elif f1 > f2:
                xl = x1
                Li = (F[n - i] / F[n - (i - 2)]) * L0
            else:
                xl, xr = x1, x2
                Li = (F[n - i] / F[n - (i - 2)]) * (xr - xl)

            L0 = xr - xl
            R.append(Li / ini)
            a.append(xl)
            b.append(xr)
            F1.append(f1)
            F2.append(f2)

        data = {
            'n': range(0, n),
            'xl': a,
            'xr': b,
            'f(x1)': F1,
            'f(x2)': F2,
            'Reduction Ratio': R
        }

        df = pd.DataFrame(data, columns=['n', 'xl', 'xr', 'f(x1)', 'f(x2)', 'Reduction Ratio'])
        return df


class Brent(MultivariableFunctionEvaluator):
    def brent(self, x0, x1, max_iter=50, tol=1e-5):
        x_history = []
        fx_history = []

        def f(x):
            return self.evaluate(x)

        x0 = np.array(x0)
        x1 = np.array(x1)
        fx0 = f(x0)
        fx1 = f(x1)

        # Check if the initial guesses bracket the root
        if (fx0 * fx1) > 0:
            raise ValueError("Initial guesses do not bracket the root.")

        if abs(fx0) < abs(fx1):
            x0, x1 = x1, x0
            fx0, fx1 = fx1, fx0

        x2, fx2 = x0, fx0
        mflag = True
        steps_taken = 0

        while steps_taken < max_iter and abs(x1 - x0) > tol:
            x_history.append(x1)
            fx_history.append(fx1)

            if fx0 != fx2 and fx1 != fx2:
                L0 = (x0 * fx1 * fx2) / ((fx0 - fx1) * (fx0 - fx2))
                L1 = (x1 * fx0 * fx2) / ((fx1 - fx0) * (fx1 - fx2))
                L2 = (x2 * fx1 * fx0) / ((fx2 - fx0) * (fx2 - fx1))
                new = L0 + L1 + L2
            else:
                new = x1 - ((fx1 * (x1 - x0)) / (fx1 - fx0))

            if ((new < ((3 * x0 + x1) / 4) or new > x1) or
               (mflag and abs(new - x1) >= (abs(x1 - x2) / 2)) or
               (not mflag and abs(new - x1) >= (abs(x2 - d) / 2))):
                new = (x0 + x1) / 2
                mflag = True
            else:
                mflag = False

            fnew = f(new)
            d, x2 = x2, x1

            if (fx0 * fnew) < 0:
                x1 = new
            else:
                x0 = new

            if abs(fx0) < abs(fx1):
                x0, x1 = x1, x0

            fx0 = f(x0)
            fx1 = f(x1)
            fx2 = f(x2)

            steps_taken += 1

        x_history.append(x1)
        fx_history.append(fx1)

        return x1, steps_taken, x_history, fx_history


class BFGS(MultivariableFunctionEvaluator):
    def line_search(self, x, p, nabla):
        a = 1
        c1 = 1e-4
        c2 = 0.9
        fx = self.evaluate(x)
        x_new = x + a * p
        nabla_new = grad(self.func)(*x_new)
        while self.evaluate(x_new) >= fx + (c1 * a * np.dot(nabla.T, p)) or np.dot(nabla_new.T, p) <= c2 * np.dot(nabla.T, p):
            a *= 0.5
            x_new = x + a * p
            nabla_new = grad(self.func)(*x_new)
        return a
    
    def BFGS(self, x0, max_it):
        d = len(x0)
        x0 = np.array(x0)
        nabla = grad(self.func)(*x0)
        H = np.eye(d)  # Initialize H as an identity matrix
        x = x0.copy()  # Make a copy of the initial guess
        it = 0

        while np.linalg.norm(nabla) > 1e-5:
            if it > max_it:
                print('Maximum iterations reached!')
                break
            it += 1

            nabla = nabla.reshape(-1, 1)
            p = - np.dot(H,nabla)  # Matrix multiplication: H should be a square matrix, nabla should be a column vector
            a = self.line_search(x, p, nabla)
            s = a * p
            x_new = x + s
            nabla_new = grad(self.func)(*x_new)
            y = nabla_new - nabla
            y = np.reshape(y, (d, 1))  # Reshape y to be a column vector
            s = np.reshape(s, (d, 1))  # Reshape s to be a column vector
            r = 1 / (np.dot(y.T, s))
            li = np.eye(d) - r * np.dot(s, y.T)
            ri = np.eye(d) - r * np.dot(y, s.T)
            hess_inter = np.dot(np.dot(li, H), ri)
            H = hess_inter + r * np.dot(s, s.T)  # BFGS Update
            nabla = nabla_new[:]
            x = x_new[:]

        return x


class NewtonMethod(MultivariableFunctionEvaluator):
    def newtons_method(self, max_its, w, values):
        epsilon = 1e-7
        weight_history = [w]
        cost_history = [self.evaluate(w)]
        for k in range(max_its):
            grad_eval = grad(self.func)(*w)
            hess_eval = self.evaluate_hessian(w)
            A = hess_eval + epsilon * np.eye(w.size)
            b = grad_eval
            w = np.linalg.solve(A, np.dot(A, w) - b)
            weight_history.append(w)
            cost_history.append(self.evaluate(w))
        return weight_history, cost_history


class ConjugateGradient(MultivariableFunctionEvaluator):
    def line_search(self, x, p, c1=1e-4, c2=0.9):
        a = 1.0
        fx = self.evaluate(x)
        nabla = grad(self.func)(*x)
        x_new = x + a * p
        nabla_new = grad(self.func)(*x_new)
        
        while (self.evaluate(x_new) >= fx + c1 * a * np.dot(nabla, p) or
               np.dot(nabla_new, p) <= c2 * np.dot(nabla, p)):
            a *= 0.5
            x_new = x + a * p
            nabla_new = grad(self.func)(*x_new)
        
        return a, x_new

    def Conjugate_Gradient(self, initial_point, tol=1e-6, alpha_1=1e-4, alpha_2=0.9, max_iterations=100):
        x = np.array(initial_point)
        x_history = [x]
        gradient = grad(self.func)(*x)
        descent_direction = -gradient
        iteration = 0

        while np.linalg.norm(gradient) > tol and iteration < max_iterations:
            a, x_new = self.line_search(x, descent_direction, alpha_1, alpha_2)
            
            if a is None:
                # Line search failed, terminate
                break

            gradient_new = grad(self.func)(*x_new)
            beta = np.dot(gradient_new, gradient_new) / np.dot(gradient, gradient)
            descent_direction = -gradient_new + beta * descent_direction
            gradient = gradient_new
            x = x_new
            x_history.append(x)
            iteration += 1
        
        return x_history
