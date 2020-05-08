import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_curve, auc

from src.training.pvdm_training import normalize


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


def doc_eval_transform(output):
    doc_ids, (_, base_doc_embedding), (_, doc_embedding) = output
    true_embedding_w = normalize(base_doc_embedding.idx2vec.weight)
    pred_embedding_w = normalize(doc_embedding(doc_ids))

    y_pred = torch.matmul(pred_embedding_w, true_embedding_w.T)
    y = doc_ids

    return y_pred, y.T


def doc_eval_flatten_transform(output):
    doc_ids, (_, base_doc_embedding), (_, doc_embedding) = output
    true_embedding_w = normalize(base_doc_embedding.idx2vec.weight)
    pred_embedding_w = normalize(doc_embedding(doc_ids))

    y_pred = torch.matmul(true_embedding_w, pred_embedding_w.T)
    y = torch.zeros_like(y_pred, dtype=torch.int32)
    y[doc_ids] = torch.eye(len(doc_ids), dtype=torch.int32)

    return y_pred.reshape(-1), y.reshape(-1)