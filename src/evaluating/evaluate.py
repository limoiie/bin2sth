import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def evaluate_auc(Y_test, pred):
    fpr, tpr, _ = roc_curve(Y_test, pred, pos_label=1)
    roc_auc = auc(fpr, tpr) * 100

    plt.figure()
    plt.plot(fpr, tpr, color='red', linewidth=1.2,
             label='Siamese Model (AUC = %0.2f%%)' % roc_auc)

    plt.plot([0, 1], [0, 1], color='silver', linestyle=':', linewidth=1.2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()