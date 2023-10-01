import numpy as np


X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([0.38, 0.64, 0.79, 0.85, 0.93, 0.94, 0.96, 0.98, 0.99, 0.99])

def mean_squared_error(y_true, y_predicted):
    # Calculating the loss or cost
    cost = np.sum((y_true-y_predicted)**2) / len(y_true)
    return cost


# Gradient Descent Function
# Here iterations, learning_rate, stopping_threshold
# are hyperparameters that can be tuned
def gradient_descent(x, y, iterations = 100, learning_rate = 0.001,
                     stopping_threshold = 1e-10):
     
    # Initializing weight, bias, learning rate and iterations
    current_k = 0
    current_tau = 3
    iterations = iterations
    learning_rate = learning_rate
    n = float(len(x))
     
    costs = []
    weights = []
    previous_cost = None
     
    # Estimation of optimal parameters
    for i in range(iterations):
         
        # Making predictions
        #y_predicted = (current_weight * x) + current_bias
        y_predicted =  current_k * (1 - np.exp(-x/current_tau))

        
        # Calculating the current cost
        current_cost = mean_squared_error(y, y_predicted)
 
        # If the change in cost is less than or equal to
        # stopping_threshold we stop the gradient descent
        if previous_cost and abs(previous_cost-current_cost)<=stopping_threshold:
            break
         
        previous_cost = current_cost
 
        costs.append(current_cost)
        weights.append(current_k)
         
        # Calculating the gradients
        #weight_derivative = -(2/n) * sum(x * (y-y_predicted))
        k_derivative = (1/n)*sum(2*y*np.exp(-x/current_tau) -2*y + 2*current_k*np.exp(-(2*x)/current_tau) -4*current_k*np.exp(-x/current_tau) + 2*current_k)

        
        #bias_derivative = -(2/n) * sum(y-y_predicted)
        tau_derivative = (1/n)*sum( (2*current_k*x*np.exp(-x/current_tau))/current_tau**2 + 
                                   (2*(current_k**2)*x*np.exp(-(2*x)/current_tau))/current_tau**2
                                   - (2*(current_k**2)*x*np.exp(-x/current_tau))/current_tau**2)
         
        # Updating weights and bias
        current_k = current_k - (learning_rate * k_derivative)
        current_tau = current_tau - (learning_rate * tau_derivative)
                 

    return current_k, current_tau


# generate function 
def f(K,tau,x):
    return K * (1 - np.exp(-x/tau))


def Jacobian(K,tau,x):
    eps = 1e-6
    grad_K = (f(K+eps, tau, x) - f(K-eps, tau, x))/(2*eps)
    grad_tau = (f(K, tau+eps, x) - f(K, tau-eps, x))/(2*eps)
    return np.column_stack([grad_K, grad_tau])


def Gauss_Newton(x, y, K0, tau0, tol, max_iter):
    old = new = np.array([K0, tau0])
    for itr in range (max_iter):
        old = new
        J = Jacobian(old[0], old[1], x)
        #print(J.shape)
        dy = y - f(old[0], old[1],x)
        new = old + np.linalg.inv(J.T@J)@J.T@dy
        if np.linalg.norm(old-new) < tol:
            break
    return new


def grad(f,x): 
    '''
    CENTRAL FINITE DIFFERENCE CALCULATION
    '''
    h = np.cbrt(np.finfo(float).eps)
    d = len(x)
    nabla = np.zeros(d)
    for i in range(d): 
        x_for = np.copy(x) 
        x_back = np.copy(x)
        x_for[i] += h 
        x_back[i] -= h 
        nabla[i] = (f(x_for) - f(x_back))/(2*h) 
    return nabla 


def line_search(f,x,p,nabla):
    '''
    BACKTRACK LINE SEARCH WITH WOLFE CONDITIONS
    '''
    a = 1
    c1 = 1e-4 
    c2 = 0.9 
    fx = f(x)
    x_new = x + a * p 
    nabla_new = grad(f,x_new)
    max_iters = 100
    i = 0
    while f(x_new) >= fx + (c1*a*nabla.T@p) or nabla_new.T@p <= c2*nabla.T@p : 
        a *= 0.5
        x_new = x + a * p 
        nabla_new = grad(f,x_new)
        i += 1
        if i >= max_iters:
            break
    return a


def BFGS(x0,t,y_true,max_it):
    '''
    DESCRIPTION
    BFGS Quasi-Newton Method

    INPUTS:
    f:      function to be optimised 
    x0:     intial guess
    max_it: maximum iterations 
    plot:   if the problem is 2 dimensional, returns 
            a trajectory plot of the optimisation scheme.

    OUTPUTS: 
    x:      the optimal solution of the function f 
    '''
    f = lambda x: np.sum((x[0] * (1 - np.exp(-t/x[1])) - y_true)**2)
    d = len(x0) # dimension of problem 
    nabla = grad(f,x0) # initial gradient 
    H = np.eye(d) # initial hessian
    x = x0[:]
    it = 2 
    while np.linalg.norm(nabla) > 1e-5: # while gradient is positive
        if it > max_it: 
            print('Maximum iterations reached!')
            break
        it += 1
        p = -H@nabla # search direction (Newton Method)
        a = line_search(f,x,p,nabla) # line search 
        s = a * p 
        x_new = x + a * p 
        nabla_new = grad(f,x_new)
        y = nabla_new - nabla 
        y = np.array([y])
        s = np.array([s])
        y = np.reshape(y,(d,1))
        s = np.reshape(s,(d,1))
        r = 1/(y.T@s)
        li = (np.eye(d)-(r*((s@(y.T)))))
        ri = (np.eye(d)-(r*((y@(s.T)))))
        hess_inter = li@H@ri
        H = hess_inter + (r*((s@(s.T)))) # BFGS Update
        nabla = nabla_new[:] 
        x = x_new[:]
    return x

