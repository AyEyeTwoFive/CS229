import numpy as np
import util
from numpy.linalg import inv


def main(train_path, valid_path, save_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***

    # Train a GDA classifier
    gda = GDA()
    gda.fit(x_train, y_train)

    # Plot decision boundary on validation set
    x_val, y_val = util.load_dataset(valid_path,add_intercept=False)
    y_pred = gda.predict(x_val)
    util.plot(x_val,y_val, gda.theta, '{}.png'.format((save_path)
                                                      ))
    # Use np.savetxt to save outputs from validation set to save_path
    np.savetxt(save_path,y_pred)

    # *** END CODE HERE ***


class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
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
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        m,n = x.shape
        phi = (1/m) * (y==1).sum()
        mu0 = (1/(y==0).sum()) * x[y == 0].sum(axis=0)
        mu1 = (1/(y==1).sum()) * x[y == 1].sum(axis=0)
        D = x.copy()
        D[y==0] -= mu0
        D[y==1] -= mu1
        sig = D.T.dot(D) / m
        sig_inv = inv(sig)
        theta = sig_inv.dot(mu1-mu0)
        theta0 = .5 * (mu0.T.dot(sig_inv).dot(mu0) - mu1.T.dot(sig_inv).dot(mu1)) - np.log((1 - phi) / phi)
        theta = np.hstack([np.array([theta0]), theta])
        self.theta = theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        sigmoid = lambda z: 1 / (1 + np.exp(-z))
        x = util.add_intercept(x)
        probs = sigmoid(x.dot(self.theta))
        preds = (probs >= 0.5).astype(int)
        return preds
        # *** END CODE HERE

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='gda_pred_2.txt')
