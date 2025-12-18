import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp


def make_data(N=100, noise_std=0.3, seed=1):
    """
    generating simple noisy line data y = 2x - 1 + noise.
    """

    rng = np.random.default_rng(seed)

    # creating x values
    x = np.linspace(-2, 2, N)

    # creating noisy y values
    y = 2 * x - 1 + rng.normal(0, noise_std, size=N)

    return x, y


def model(params, x):
    # computing y = a x + b
    a, b = params
    return a * x + b


def loss_fn(params, x, y):
    # computing mean squared error
    y_pred = model(params, x)
    return jnp.mean((y_pred - y) ** 2)


@jax.jit
def step(params, x, y, lr):
    """
    taking one gradient descent step.
    """

    # computing gradients of the loss
    grads = jax.grad(loss_fn)(params, x, y)

    # updating parameters
    new_params = params - lr * grads

    # computing current loss
    current_loss = loss_fn(params, x, y)

    return new_params, current_loss


# running the script when executed directly
if __name__ == "__main__":
    # generating data
    x_np, y_np = make_data(N=100, noise_std=0.3, seed=1)

    # converting data to jax arrays
    x = jnp.array(x_np)
    y = jnp.array(y_np)

    # setting starting guess
    params = jnp.array([0, 0], dtype=float)

    # setting optimization settings
    lr = 0.1
    n_steps = 200

    # storing loss history
    loss_history = []

    # running gradient descent
    for _ in range(n_steps):
        params, current_loss = step(params, x, y, lr)
        loss_history.append(float(current_loss))

    # extracting final parameters
    a_fit, b_fit = float(params[0]), float(params[1])

    print("\njax optimization result")
    print(f"a ≈ {a_fit}")
    print(f"b ≈ {b_fit}\n")

    # plotting loss vs iteration
    plt.figure()
    plt.plot(loss_history)
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.title("loss vs iteration (jax gradient descent)")
    plt.show()

    # plotting data and fitted line
    y_fit = a_fit * x_np + b_fit

    plt.figure()
    plt.plot(x_np, y_np, "o", label="data")
    plt.plot(x_np, y_fit, label="fit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("linear fit using jax")
    plt.legend()
    plt.show()
