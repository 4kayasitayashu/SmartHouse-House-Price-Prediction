
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Select numerical features for the initial analysis
columns = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "YearBuilt",
    "SalePrice",
]

df = pd.read_csv("data/raw/train.csv")
correlation = df[columns].corr()

# Plot the correlation matrix
fig, ax = plt.subplots(figsize=(9, 7))

image = ax.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)

ax.set_xticks(range(len(columns)))
ax.set_yticks(range(len(columns)))
ax.set_xticklabels(columns, rotation=45, ha="right")
ax.set_yticklabels(columns)

# Display correlation values inside the cells
for i in range(len(columns)):
    for j in range(len(columns)):
        ax.text(
            j, i, f"{correlation.iloc[i, j]:.2f}",
            ha="center", va="center", color="black"
        )

fig.colorbar(image, ax=ax, label="Correlation")
ax.set_title("SmartHouse: Feature Correlation Heatmap")
plt.tight_layout()

output = Path("reports/correlation_heatmap.png")
output.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output, dpi=150)
plt.show()

print(f"Heatmap saved to {output}")