import torch


def evaluate(model, test_data):
    """
    Evaluates model performance on test data.
    :param test_data: Test data consisting of (input, label) pairs.
    :return: Dictionary containing the number of correct and incorrect predictions by category.
    """
    x, y = test_data.get_data()
    pred = model(x)

    evaluation = {i: {True: 0, False: 0} for i in range(y.shape[1])}
    for (x, y) in zip(torch.argmax(pred, dim=1), torch.argmax(y, dim=1)):
        evaluation[x.item()][x.item() == y.item()] += 1
    return evaluation,pred.shape[0]


def accuracy_score(model, test_data):
    evaluation, size = evaluate(model, test_data)
    true_positives = sum(v[True] for v in evaluation.values())
    return true_positives / size


def recall_score(model, test_data):
    evaluation, _ = evaluate(model, test_data)
    recalls = []
    for class_label, counts in evaluation.items():
        true_positives = counts[True]
        false_negatives = sum(v[True] for k, v in evaluation.items() if k != class_label)
        if true_positives + false_negatives > 0:
            recalls.append(true_positives / (true_positives + false_negatives))
    return sum(recalls) / len(recalls) if recalls else 0.0


def f1_score(model, test_data):
    # Calculate precision and recall first
    evaluation, _ = evaluate(model, test_data)
    precisions, recalls = [], []

    for class_label, counts in evaluation.items():
        true_positives = counts[True]
        false_positives = sum(v[False] for k, v in evaluation.items() if k == class_label)
        false_negatives = sum(v[True] for k, v in evaluation.items() if k != class_label)

        # Calculate precision and recall for each class
        if true_positives + false_positives > 0:
            precisions.append(true_positives / (true_positives + false_positives))
        if true_positives + false_negatives > 0:
            recalls.append(true_positives / (true_positives + false_negatives))

    # Compute F1 score for each class and take the average
    f1_scores = [2 * (p * r) / (p + r) for p, r in zip(precisions, recalls) if p + r > 0]
    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0