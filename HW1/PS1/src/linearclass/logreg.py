import numpy as np
import util
import matplotlib.pyplot as plt


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
    model = LogisticRegression(eps=1e-5)
    model.fit(x_train, y_train)

    # Plot decision boundary on top of validation set set
    x_val, y_val = util.load_dataset(valid_path, add_intercept=True)
    y_pred = model.predict(x_val)
    util.plot(x_val, y_val, model.theta, '{}.png'.format(save_path))

    # Use np.savetxt to save predictions on eval set to save_path
    np.savetxt(save_path, y_pred)

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
        g = lambda x: 1 / (1 + np.exp(-x))
        m, n = x.shape

        # initialize theta
        if self.theta is None:
            self.theta = np.zeros(n)

        for i in range(self.max_iter):
            theta = self.theta
            J = - (y - g(x.dot(theta))).dot(x) * (1 / m)

            x_dot_theta = x.dot(theta)
            H = g(x_dot_theta).dot(g(1 - x_dot_theta)) * (1 / m) * (x.T).dot(x)
            Hinv = np.linalg.inv(H)

            self.theta = theta - Hinv.dot(J)

            if np.linalg.norm(self.theta - theta, ord=1) < self.eps:
                break

        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        g = lambda x: 1 / (1 + np.exp(-x))
        preds = g(x.dot(self.theta))
        return preds
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='logreg_pred_2.txt')
