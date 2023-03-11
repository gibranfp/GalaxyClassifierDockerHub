from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def get_confusion_matrix(y_true, y_pred, save_img = False, img_name = 'confusion_mat.jpg'):
    
    # Get conf mat
    labels = np.unique(y_true)
    confusion_mat = confusion_matrix(y_true, y_pred, labels = labels)
    # Normalize matrix
    cm_normalized = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis]
    # Create plot
    plt.figure(figsize = (10,7))
    ax = sns.heatmap(cm_normalized, annot=True, cmap="BuPu") #notation: "annot" not "annote"
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    # Save img
    if save_img:
        plt.savefig(img_name , bbox_inches='tight')
    # Show matrix
    plt.show()
    
    return None