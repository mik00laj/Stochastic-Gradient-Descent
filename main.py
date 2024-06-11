import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def f(x, y):
    return (9 * x * y) / np.exp(x ** 2 + 0.5 * x + y ** 2)


def gradient(x, y):
    df_dx = (np.exp(-x ** 2 - y ** 2 - 0.5 * x)) * 9 * (1 - 0.5 * x - 2 * x ** 2) * y
    df_dy = (np.exp(-x ** 2 - y ** 2 - 0.5 * x)) * 9 * (x - 2 * x * y ** 2)
    return df_dx, df_dy


def gradient_descent(initial_x, initial_y, learning_rate, num_iterations, tolerance):
    x = initial_x
    y = initial_y
    x_history = [x]
    y_history = [y]

    for i in range(num_iterations):
        d = gradient(x, y)
        x = x - learning_rate * d[0]
        y = y - learning_rate * d[1]
        x_history.append(x)
        y_history.append(y)
        if i > 0 and abs(f(x_history[i], y_history[i]) - f(x_history[i - 1], y_history[i - 1])) <= tolerance:
            break

    return x, x_history, y, y_history


def plot_contour(X, Y, Z, x_history, y_history, x, y):
    plt.figure(figsize=(12, 10))
    contour = plt.contour(X, Y, Z, levels=20, cmap='viridis')
    plt.scatter(x_history, y_history, c='red', label='Gradient Descent')
    plt.scatter(x, y, c='green', marker='*', label='Final Point')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Gradient Descent in Contour Plot')
    plt.legend()
    plt.colorbar(contour)
    plt.show()


def plot_mesh_grid(X, Y, Z, x_history, y_history, x, y):
    plt.figure(figsize=(12, 10))
    plt.pcolormesh(X, Y, Z, cmap='viridis')
    plt.colorbar()
    plt.plot(x_history, y_history, 'rx-', label='Gradient Descent')
    plt.scatter([x], [y], color='green', marker='*', label='Final Point')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Gradient Descent in 2D')
    plt.legend()
    plt.show()


def plot_3d_surface(X, Y, Z, x_history, y_history, x, y):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, antialiased=True)
    ax.plot(x_history, y_history, f(np.array(x_history), np.array(y_history)), 'rx-', label='Gradient Descent')
    ax.scatter([x], [y], [f(x, y)], color='green', marker='*', label='Final Point')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('f(X, Y)')
    ax.set_title('Gradient Descent in 3D')
    ax.legend()
    plt.show()


def main():
    learning_rate = 0.1
    num_iterations = 1000
    tolerance = 1e-10

    # Uncomment the desired section to find minimum or maximum
    # Find II Minimum
    # initial_x = 0
    # initial_y = -1

    # Find IV Minimum
    # initial_x = 0
    # initial_y = 1

    # Find I Maximum
    # learning_rate = -learning_rate
    # initial_x = 0
    # initial_y = -1

    # Find III Maximum
    learning_rate = -learning_rate
    initial_x = 0
    initial_y = 1

    x, x_history, y, y_history = gradient_descent(initial_x, initial_y, learning_rate, num_iterations, tolerance)

    print("Local minimum: {:.12f}".format(f(x, y)))
    print("Final Point: x = {:.12f}, y = {:.12f}".format(x, y))
    print("Number of Iterations:", len(x_history))

    x_vals = np.linspace(-3.0, 3, 100)
    y_vals = np.linspace(-3.0, 3, 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = f(X, Y)

    plot_contour(X, Y, Z, x_history, y_history, x, y)
    plot_mesh_grid(X, Y, Z, x_history, y_history, x, y)
    plot_3d_surface(X, Y, Z, x_history, y_history, x, y)


if __name__ == "__main__":
    main()
