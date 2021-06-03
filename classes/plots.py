import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shap

from sklearn.metrics import plot_confusion_matrix
from sklearn.calibration import calibration_curve
from sklearn.model_selection import learning_curve

def show_loss_plot(loss_values):
    plt.plot(loss_values)
    plt.show()

def show_learning_curve(model, X, y):
    _, axes = plt.subplots(1, 3, figsize=(20, 5))
    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(model, X, y, cv=3, n_jobs=-1, return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    plt.show()


def show_heatmap(model, X, y):
    """
    regression visual.
    """
    y_pred = model.predict(X)

    heatmap, xedges, yedges = np.histogram2d(y, y_pred, bins=50)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plt.clf()
    plt.ylabel('predicted values')
    plt.xlabel('true values')
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.show()


def show_regplot(model, X, y):
    """
    regression visual.
    """
    y_pred = model.predict(X)

    sns.set_theme(color_codes=True)

    # sns.regplot(x=y, y=y_pred)
    sns.regplot(x=y, y=y_pred, order=2)

    plt.show()


def show_confusion_matrix(model, X, y):
    """
    classification visual
    """
    disp = plot_confusion_matrix(model, X, y, cmap=plt.cm.Blues)
    print(disp.confusion_matrix)
    plt.show()


def show_calibration_curve(models, X, y):
    """
    models in format: [(model, 'model_name'), ..]
    """
    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    for clf, name in models:
        # clf.fit(X_input_train, y_train)
        prob_pos = clf.predict_proba(X)[:, 1]
        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y, prob_pos, n_bins=10)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label=name)

        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
            histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=3)

    plt.tight_layout()
    plt.show()
