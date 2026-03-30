from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from collections import Counter
import pickle
import matplotlib.pyplot as plt
import numpy as np


# Load data
data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])


# Debug info
print("\n--- DATA INSPECTION ---")
print(f"Total samples: {len(data)}")
print(f"Feature size: {data.shape[1]}")
print(f"Sample features: {data[0][:10]}")
print(f"Class distribution: {Counter(labels)}")
print("-----------------------\n")

from sklearn.utils import shuffle
data, labels = shuffle(data, labels, random_state=42)
# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    data, labels,
    test_size=0.2,
    shuffle=True,
    stratify=labels,
    random_state=42
)

# Model comparison
models = {
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(kernel='linear')   # faster + stable
}

best_model = None
best_score = 0
best_model_name = ""

for name, model in models.items():
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    print(f"{name} Accuracy: {score}")

    if score > best_score:
        best_score = score
        best_model = model
        best_model_name = name


print("\nBest Model:", best_model_name)
print("Best Accuracy:", best_score)


# Evaluation
y_pred = best_model.predict(x_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Accuracy (correct)
final_accuracy = accuracy_score(y_test, y_pred)
print(f"\nFinal Accuracy: {final_accuracy * 100:.2f}%")


# Feature importance (only RF)
if best_model_name == "Random Forest":
    importances = best_model.feature_importances_

    plt.figure()
    plt.plot(importances)
    plt.title("Feature Importance")
    plt.xlabel("Feature Index")
    plt.ylabel("Importance")
    plt.show()


# Save ONLY best model
with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("\nModel saved successfully!")
f.close()
