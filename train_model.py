import pandas as pd
import numpy as np
import joblib

print("Loading dataset...")

df = pd.read_csv("diabetes_prediction_dataset.csv")
df = df.drop_duplicates()

print("Dataset loaded!")
print("Shape:", df.shape)

# Convert categorical columns to numbers
categorical_columns = ["gender", "smoking_history"]

for col in categorical_columns:
    df[col] = df[col].astype("category").cat.codes

X = df.drop("diabetes", axis=1).values.astype(float)
y = df["diabetes"].values.astype(int)

# Fill missing values
X = np.nan_to_num(X, nan=0.0)

# Shuffle dataset
np.random.seed(42)
indices = np.random.permutation(len(X))

X = X[indices]
y = y[indices]

# Use 80% for training and 20% for testing
split = int(len(X) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# Simple decision tree
class Tree:
    def __init__(self, feature=None, threshold=None, left=None, right=None, prediction=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.prediction = prediction


def gini(y):
    if len(y) == 0:
        return 0

    p = np.mean(y)
    return 1 - p * p - (1 - p) * (1 - p)


def build_tree(X, y, depth=0, max_depth=6):

    # Stop condition
    if len(y) < 20 or depth >= max_depth or len(np.unique(y)) == 1:
        return Tree(prediction=int(np.mean(y) >= 0.5))

    best_feature = None
    best_threshold = None
    best_score = float("inf")

    # Randomly select features
    num_features = X.shape[1]
    features = np.random.choice(
        num_features,
        size=min(5, num_features),
        replace=False
    )

    for feature in features:

        values = np.unique(X[:, feature])

        if len(values) > 20:
            values = np.percentile(values, np.linspace(10, 90, 10))

        for threshold in values:

            left_mask = X[:, feature] <= threshold
            right_mask = ~left_mask

            if left_mask.sum() < 5 or right_mask.sum() < 5:
                continue

            score = (
                left_mask.sum() * gini(y[left_mask])
                + right_mask.sum() * gini(y[right_mask])
            ) / len(y)

            if score < best_score:
                best_score = score
                best_feature = feature
                best_threshold = threshold

    if best_feature is None:
        return Tree(prediction=int(np.mean(y) >= 0.5))

    left_mask = X[:, best_feature] <= best_threshold
    right_mask = ~left_mask

    left_tree = build_tree(
        X[left_mask],
        y[left_mask],
        depth + 1,
        max_depth
    )

    right_tree = build_tree(
        X[right_mask],
        y[right_mask],
        depth + 1,
        max_depth
    )

    return Tree(
        feature=best_feature,
        threshold=best_threshold,
        left=left_tree,
        right=right_tree
    )


def predict_tree(tree, row):

    if tree.prediction is not None:
        return tree.prediction

    if row[tree.feature] <= tree.threshold:
        return predict_tree(tree.left, row)

    return predict_tree(tree.right, row)


# Random Forest
print("\nTraining Random Forest...")

trees = []

number_of_trees = 15

for i in range(number_of_trees):

    sample_indices = np.random.choice(
        len(X_train),
        size=len(X_train),
        replace=True
    )

    sample_X = X_train[sample_indices]
    sample_y = y_train[sample_indices]

    tree = build_tree(
        sample_X,
        sample_y,
        max_depth=6
    )

    trees.append(tree)

    print("Tree", i + 1, "completed")

print("\nTraining completed!")


# Prediction
def predict_forest(trees, X):

    predictions = []

    for row in X:

        votes = [
            predict_tree(tree, row)
            for tree in trees
        ]

        predictions.append(int(np.mean(votes) >= 0.5))

    return np.array(predictions)


y_pred = predict_forest(trees, X_test)


# Metrics
tp = np.sum((y_test == 1) & (y_pred == 1))
tn = np.sum((y_test == 0) & (y_pred == 0))
fp = np.sum((y_test == 0) & (y_pred == 1))
fn = np.sum((y_test == 1) & (y_pred == 0))

accuracy = (tp + tn) / len(y_test)

precision = tp / (tp + fp) if (tp + fp) > 0 else 0

recall = tp / (tp + fn) if (tp + fn) > 0 else 0

f1 = (
    2 * precision * recall / (precision + recall)
    if (precision + recall) > 0
    else 0
)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))

print("\nConfusion Matrix:")
print("[[TN FP]")
print(" [FN TP]]")

print("[[", tn, fp, "]")
print(" [", fn, tp, "]]")


# Save model
model_data = {
    "trees": trees,
    "categorical_columns": categorical_columns,
    "feature_names": df.drop("diabetes", axis=1).columns.tolist()
}

joblib.dump(model_data, "diabetes_model.pkl")

print("\nModel saved as diabetes_model.pkl")

# Save results
with open("model_results.txt", "w") as file:
    file.write("AI-BASED DIABETES PREDICTION SYSTEM\n")
    file.write("====================================\n\n")
    file.write(f"Accuracy : {accuracy:.4f}\n")
    file.write(f"Precision: {precision:.4f}\n")
    file.write(f"Recall   : {recall:.4f}\n")
    file.write(f"F1 Score : {f1:.4f}\n")

print("Results saved as model_results.txt")
