
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("data/raw/train.csv")

fig, axes = plt.subplots(1, 3, figsize=(19, 6))

# Sale price by overall quality
df.boxplot(column="SalePrice", by="OverallQual", ax=axes[0])
axes[0].set_title("Sale Price by Overall Quality")
axes[0].set_xlabel("Overall Quality")
axes[0].set_ylabel("Sale Price ($)")

# Sale price by kitchen quality
kitchen_order = ["Fa", "TA", "Gd", "Ex"]
df.boxplot(
    column="SalePrice",
    by="KitchenQual",
    ax=axes[1],
    positions=range(1, len(kitchen_order) + 1),
)
axes[1].set_xticks(range(1, len(kitchen_order) + 1))
axes[1].set_xticklabels(kitchen_order)
axes[1].set_title("Sale Price by Kitchen Quality")
axes[1].set_xlabel("Kitchen Quality")

# Median price by neighborhood, sorted for readability
medians = df.groupby("Neighborhood")["SalePrice"].median().sort_values()
medians.plot(kind="barh", ax=axes[2])
axes[2].set_title("Median Sale Price by Neighborhood")
axes[2].set_xlabel("Median Sale Price ($)")
axes[2].set_ylabel("Neighborhood")

fig.suptitle("")
plt.tight_layout()

output = Path("reports/price_group_comparisons.png")
output.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output, dpi=150, bbox_inches="tight")
plt.show()

print(f"Saved chart to {output}")