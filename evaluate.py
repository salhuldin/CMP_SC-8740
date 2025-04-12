from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, matthews_corrcoef
import numpy as np

def evaluate_model(trainer, test_dataset):
    predictions = trainer.predict(test_dataset)
    valid = predictions.label_ids != -100
    labels = predictions.label_ids[valid]
    preds = np.argmax(predictions.predictions[valid], axis=1)

    conf_matrix = confusion_matrix(labels, preds)
    report = classification_report(labels, preds)
    acc = accuracy_score(labels, preds)
    mcc = matthews_corrcoef(labels, preds)

    print("Confusion Matrix:\n", conf_matrix)
    print("\nClassification Report:\n", report)
    print("\nAccuracy:", acc)
    print("Matthews Correlation Coefficient (MCC):", mcc)
