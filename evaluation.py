import torch
import torchmetrics
from torchmetrics.classification import MulticlassConfusionMatrix

def mean_avg_precision(y_true, y_pred, k=10) -> float:
    return y_true


if __name__ == "__main__":
    gold_labels = torch.IntTensor([1, 6, 3, 0, 6, 9, 3, 5, 2, 0])
    pred_labels = torch.IntTensor([1, 6, 3, 0, 6, 9, 3, 5, 2, 0])

    #mean_avg_prec = mean_avg_precision(gold_labels, pred_labels)
    #print(mean_avg_prec)

    precision = torchmetrics.Precision(task="multiclass", average='macro', num_classes=10)
    precision_result = precision(pred_labels, gold_labels)
    print(precision_result)

    recall = torchmetrics.Recall(task="multiclass", average='macro', num_classes=10)
    recall_result = recall(pred_labels, gold_labels)
    print(recall_result)

    confmat = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=10)
    print(confmat(pred_labels, gold_labels))
