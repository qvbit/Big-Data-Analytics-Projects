import matplotlib.pyplot as plt
# TODO: You can use other packages if you want, e.g., Numpy, Scikit-learn, etc.
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies):
    # TODO: Make plots for loss curves and accuracy curves.
    # TODO: You do not have to return the plots.
    # TODO: You can save plots as files by codes here or an interactive way according to your preference.
    
     # Loss curve
    plt.plot(train_losses, label='Train')
    plt.plot(valid_losses, label='Validation')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title('Loss Curve')
    plt.legend()
    plt.show()
    
     # Accuracy Curve
    plt.plot(train_accuracies, label='Train')
    plt.plot(valid_accuracies, label='Validation')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.show()
    pass

# Credit: Most of my code for the confusion matrix plot was based on code from sklearn documentation.

def plot_confusion_matrix(results, class_names):
	
    # # TODO: Make a confusion matrix plot.
    # # TODO: You do not have to return the plots.
    # # TODO: You can save plots as files by codes here or an interactive way according to your preference.
    y_true = list(zip(*results))[0]
    y_pred = list(zip(*results))[1]
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    cmap=plt.cm.Blues
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title="Confusion Matrix",
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()
    pass