

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("/content/drive/MyDrive/ML (1)/spam_or_not_spam.csv")
df.head()

print("Columns found:", df.columns.tolist())

df["email"] = df["email"].fillna("")
df["label"] = df["label"].astype(int)

X_text = df["email"]
y = df["label"]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X_text)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

train_count = X_train.shape[0]
test_count = X_test.shape[0]

print("\nTraining samples :", train_count)
print("Testing samples  :", test_count)

plt.figure(figsize=(6,4))
plt.bar(["Train", "Test"], [train_count, test_count])
plt.title("Train vs Test Sample Distribution")
plt.ylabel("Number of Samples")
plt.show()

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_cont = lr.predict(X_test)
y_pred = (y_pred_cont >= 0.5).astype(int)

mse = mean_squared_error(y_test, y_pred_cont)
print("MSE (Mean Squared Error):", mse)

rmse = np.sqrt(mse)
print("RMSE (Root Mean Squared Error):", rmse)

mae = mean_absolute_error(y_test, y_pred_cont)
print("MAE (Mean Absolute Error):", mae)

r2 = r2_score(y_test, y_pred_cont)
print("R² Score:", r2)

accuracy = accuracy_score(y_test, y_pred)
print("\n📌 Model Accuracy:", accuracy)

plt.figure(figsize=(4,4))
plt.bar(["Accuracy"], [accuracy])
plt.title("Model Accuracy (Linear Regression)")
plt.ylim(0, 1)  # accuracy max = 1
plt.show()

sample = ["Congratulations you won a prize claim now"]
sample_vec = vectorizer.transform(sample)
sample_pred = lr.predict(sample_vec)

print("\nCustom sample prediction:",
      "SPAM ✓" if sample_pred >= 0.5 else "NOT SPAM ✗")

actual = np.array(y_test[:200])
predicted = np.array(y_pred_cont[:200])
plt.figure(figsize=(12,5))
plt.plot(actual, label="Actual (0 = not spam, 1 = spam)")
plt.plot(predicted, label="Predicted (Linear Regression)", linestyle='--')
plt.title("Linear Graph: Actual vs Predicted Spam Values")
plt.xlabel("Sample Index")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()
