import numpy as np
import matplotlib.pyplot as plt


def computeCost(X, y, theta):
    """
    Computes cost according to MSE.

    Args:
        X:
        y:
        theta:

    Returns:

    """
    dis = (np.dot(X, theta) - y)
    J = np.dot(dis, dis) / m / 2
    return J


def gradientDescent(X, y, theta, alpha, num_iters):
    """


    Args:
        X:
        y:
        theta:
        alpha:
        num_iters:

    Returns:

    """
    results = np.zeros(num_iters)
    new_theta = theta
    for i in range(num_iters):
        new_theta = new_theta - (alpha / m) * np.dot((np.dot(X, new_theta) - y), X)  # optimization step.
        results[i] = computeCost(X, y, new_theta)

    return new_theta, results


# Programming Exercise 1: Linear Regression

# Linear regression with one variable

if __name__ == "__main__":

    data = np.loadtxt("C:\\Users\\Galaz\\Desktop\\AI\Machine Learning\\CourseraML-master\\ex1\\data\\ex1data1.txt",
                      delimiter=',')
    iterations = 1500
    alpha = 0.01
    theta = np.zeros(2)

    x, y = np.array(data[:, 0]), np.array(data[:, 1])
    m = len(y)
    X = x.reshape(m, 1)
    X = np.insert(X, 0, 1, axis=1)

    new_theta, J_history = gradientDescent(X, y, theta, alpha, iterations)

    plt.figure(1)
    plt.plot(x, y, 'rx', markersize=10)
    plt.title('Training Data')
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.show()

    plt.figure(2)
    plt.plot(range(iterations), J_history)
    plt.xlabel('Iteration number')
    plt.ylabel('Cost Function')
    plt.show()

    plt.figure(3)
    plt.plot(x, y, 'rx', markersize=10)
    plt.plot(x, np.dot(X, new_theta))
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.legend(["T.Data", "L.Regression"])
    plt.show()

    theta0_vals = np.linspace(-10, 10, 1000)
    theta1_vals = np.linspace(-1, 4, 1000)
    J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))
    for i, value_i in enumerate(theta0_vals):
        for j, value_j in enumerate(theta1_vals):
            t = np.array([value_i, value_j])
            J_vals[i, j] = computeCost(X, y, t)

    px, py = np.meshgrid(theta0_vals, theta1_vals)

    plt.figure(4)
    ax = plt.gca(projection='3d')
    surf = ax.plot_surface(px, py, J_vals.T, cmap='jet',
                           antialiased=True)
    plt.plot(new_theta[0], new_theta[1], 'x', color='r')
    ax.set_xlabel(r'$\theta_0$')
    ax.set_ylabel(r'$\theta_1$')
    plt.show()

    plt.figure(5)
    plt.contour(px, py, J_vals.T)
    plt.plot(new_theta[0], new_theta[1], 'x', color='r', mew=2)
    plt.show()
