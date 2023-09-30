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
