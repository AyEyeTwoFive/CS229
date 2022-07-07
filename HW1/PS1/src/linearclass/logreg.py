import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***

    # Train a logistic regression classifier
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def grad_l(theta, x, y):
        z = y * x.dot(theta)
        g = -np.mean((1 - sigmoid(z)) * y * x.T, axis=1)
        return g

    def hess_l(theta, x, y):
        hess = np.zeros((x.shape[1], x.shape[1]))
        z = y * x.dot(theta)
        for i in range(hess.shape[0]):
            for j in range(hess.shape[1]):
                if i <= j:
                    hess[i][j] = np.mean(sigmoid(z) * (1 - sigmoid(z)) * x[:, i] * x[:, j])
                    if i != j:
                        hess[j][i] = hess[i][j]
        return hess

    def newton(theta0, x, y, G, H, eps):
        theta = theta0
        delta = 1
        while delta > eps:
            theta_prev = theta.copy()
            theta -= np.linalg.inv(H(theta, x, y)).dot(G(theta, x, y))
            delta = np.linalg.norm(theta - theta_prev, ord=1)
        return theta

    # Initialize theta0
    theta0 = np.zeros(x_train.shape[1])

    # Run Newton's method
    theta_final = newton(theta0, x_train, y_train, grad_l, hess_l, 1e-6)
    print(theta_final)

    # Plot decision boundary on top of validation set set
    x_values = np.linspace(x_train.min(),x_train.max(),2)
    plt.scatter()

    # Use np.savetxt to save predictions on eval set to save_path

    # *** END CODE HERE ***


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***

        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***

        # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='logreg_pred_2.txt')
