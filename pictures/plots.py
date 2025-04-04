import matplotlib.pyplot as plt
import pandas as pd

# Prepare the data
data = {
    "Coreset Size": [1, 5, 10, 20],
    "Accuracy FE DB2 Wasserstein": [73.35, 71.05, 78.44, 79.86],
    "Accuracy DD": [78.75, 76.06, 78.70, None],
    "Accuracy Random": [77.43, 75.71, 67.23, 76.76]
}

# Macro F1 score
# data = {
#     "Coreset Size": [1, 5, 10, 20],
#     "Macro F1 Score FE DB2 Wasserstein": [33.71, 44.61, 44.54, 41.67],
#     "Macro F1 Score DD": [37.24, 44.98, 43.22, None],
#     "Macro F1 Score Random": [34.27, 45.62, 44.55, 41.09]
# }

df = pd.DataFrame(data)

# Plot accuracy
plt.figure(figsize=(10, 6))
plt.plot(df["Coreset Size"], df["Accuracy FE DB2 Wasserstein"], marker='o', label="Accuracy FE DB2 Wasserstein")
plt.plot(df["Coreset Size"], df["Accuracy DD"], marker='o', label="Accuracy DD")
plt.plot(df["Coreset Size"], df["Accuracy Random"], marker='o', label="Accuracy Random")

# plot macro F1
# plt.figure(figsize=(10, 6))
# plt.plot(df["Coreset Size"], df["Macro F1 Score FE DB2 Wasserstein"], marker='o', label="Macro F1 Score FE DB2 Wasserstein")
# plt.plot(df["Coreset Size"], df["Macro F1 Score DD"], marker='o', label="Macro F1 Score DD")
# plt.plot(df["Coreset Size"], df["Macro F1 Score Random"], marker='o', label="Macro F1 Score Random")

# Customize the chart
plt.xlabel("Coreset Size (Fraction of all Frames)", fontsize=16)
plt.ylabel("Accuracy", fontsize=16)
# plt.ylabel("Macro F1 Score", fontsize=16)
plt.xticks(df["Coreset Size"], labels=["1%", "5%", "10%", "20%"], fontsize=16)

# accuracy
plt.yticks(range(65, 82, 2), [f"{v},00%" for v in range(65, 82, 2)], fontsize=16)
plt.grid(True, linestyle='--', alpha=0.6)

# Dynamically set y-axis range based on data
# min_y = int(min(df.drop(columns="Coreset Size").min()) // 5 * 5)
# max_y = int(max(df.drop(columns="Coreset Size").max()) // 5 * 5 + 5)
# plt.yticks(range(min_y, max_y + 1, 2), [f"{v},00%" for v in range(min_y, max_y + 1, 2)], fontsize=16)

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=16)
plt.tight_layout()

# Show or save
plt.savefig("/vol/ideadata/ep56inew/myCode/pictures/coreset_accuracy_matplotlib.png", dpi=300)