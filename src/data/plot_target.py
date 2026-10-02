
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load the training dataset
data_path = Path("data/raw/train.csv")
df = pd.read_csv(data_path)

# Create a figure with two plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(df["SalePrice"], bins=30, edgecolor="black")
axes[0].set_title("Distribution of House Sale Prices")
axes[0].set_xlabel("Sale Price ($)")
axes[0].set_ylabel("Number of Houses")
axes[0].ticklabel_format(style="plain", axis="x")

# Box plot
axes[1].boxplot(df["SalePrice"], vert=True)
axes[1].set_title("Box Plot of Sale Prices")
axes[1].set_ylabel("Sale Price ($)")
axes[1].ticklabel_format(style="plain", axis="y")

plt.tight_layout()

# Save the chart
output_dir = Path("reports")
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir / "saleprice_distribution.png", dpi=150)

# Display the chart
plt.show()

print("Chart saved to reports/saleprice_distribution.png")