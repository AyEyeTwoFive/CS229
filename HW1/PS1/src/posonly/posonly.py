import numpy as np
import util
import sys

sys.path.append('../linearclass')

### NOTE : You need to complete logreg implementation first!

from logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/save_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, save_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on t-labels,
        2. on y-labels,
        3. on y-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        save_path: Path to save predictions.
    """
    output_path_true = save_path.replace(WILDCARD, 'true')
    output_path_naive = save_path.replace(WILDCARD, 'naive')
    output_path_adjusted = save_path.replace(WILDCARD, 'adjusted')

    # *** START CODE HERE ***

    # Part (a): Train and test on true labels
    # Make sure to save predicted probabilities to output_path_true using np.savetxt()
    x_train, t_train = util.load_dataset(train_path, label_col='t', add_intercept=True)
    log = LogisticRegression(max_iter=1000)
    log.fit(x_train, t_train)
    x_test, t_test = util.load_dataset(test_path, label_col='t', add_intercept=True)
    t_pred = log.predict(x_test)
    util.plot(x_test, t_test, log.theta, '{}.png'.format(output_path_true))
    np.savetxt(output_path_true, t_pred)

    # Part (b): Train on y-labels and test on true labels
    # Make sure to save predicted probabilities to output_path_naive using np.savetxt()
    x_train, y_train = util.load_dataset(train_path, label_col='y', add_intercept=True)
    log = LogisticRegression()
    log.fit(x_train, y_train)
    t_pred = log.predict(x_test)
    util.plot(x_test, t_test, log.theta, '{}.png'.format(output_path_naive))
    np.savetxt(output_path_naive, t_pred)

    # Part (f): Apply correction factor using validation set and test on true labels
    x_val, y_val = util.load_dataset(valid_path, label_col='y', add_intercept=True)
    y_pred = log.predict(x_val)
    alpha = y_pred[y_val == 1].sum() / (y_val == 1).sum()
    correction = (np.log(2 / alpha - 1) / log.theta[0]) + 1
    util.plot(x_test, t_test, log.theta, '{}.png'.format(output_path_adjusted), correction=correction)
    np.savetxt(output_path_adjusted, t_pred)

    # Plot and use np.savetxt to save outputs to output_path_adjusted

    # *** END CODER HERE

if __name__ == '__main__':
    main(train_path='train.csv',
        valid_path='valid.csv',
        test_path='test.csv',
        save_path='posonly_X_pred.txt')
