import numpy as np

t = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
m = np.array([0.38, 0.64, 0.79, 0.85, 0.93, 0.94, 0.96, 0.98, 0.99, 0.99])

def gradient_descent(alpha, max_iters, k_guess=5.0, tau_guess=5.0):
    errors = []
    # Perform gradient descent
    for i in range(max_iters):
        # Calculate the model predictions
        predictions = k_guess * (1 - np.exp(-t / tau_guess))

        # Calculate the error (mean squared error)
        error = np.mean((m - predictions) ** 2)

        # Calculate the gradients
        gradient_k = -2 * np.mean((m - predictions) * (1 - np.exp(-t/tau_guess)))
        gradient_tau = -2 * np.mean((m - predictions) * k_guess * t * np.exp(-t / tau_guess) / tau_guess**2)

        # Update k and tau using gradient descent
        k_guess -= alpha * gradient_k
        tau_guess -= alpha * gradient_tau

        # Append the error to the list for plotting
        errors.append(error)

    return errors, k_guess, tau_guess




