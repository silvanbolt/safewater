import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, PowerNorm

# 1. DATA
data = [
    [16694, 416, 3791, 1814, 25, 130, 7, 254, 26],
    [48, 1472, 16, 89, 9, 91, 0, 119, 13],
    [237, 11, 454, 28, 0, 12, 0, 6, 3],
    [848, 182, 327, 5041, 9, 54, 0, 194, 75],
    [4, 10, 0, 8, 92, 1, 1, 5, 0],
    [8, 65, 10, 11, 0, 340, 0, 18, 0],
    [1, 1, 0, 0, 0, 0, 3, 2, 0],
    [13, 25, 13, 41, 1, 11, 0, 413, 17],
    [0, 9, 4, 14, 0, 11, 0, 14, 1366]
]

classes = [
    'Borehole\nTubewell', 'Piped\nWater', 'Protected\nSpring', 
    'Protected\nWell', 'Rainwater\nHarvesting', 'Sand or Sub-\nsurface Dam', 
    'Delivered\nWater', 'Surface\nWater', 'Unprotected\nWell'
]

df = pd.DataFrame(data, index=classes, columns=classes)

# 2. COLOR REPLICATION
colors = ["#ffffd4", "#fed98e", "#fe9929", "#d95f0e", "#993404", "#700000"]
custom_cmap = LinearSegmentedColormap.from_list("original_style", colors)

# Figsize width is the priority for the poster column
plt.figure(figsize=(10, 12)) 
sns.set_theme(style="white")

# 3. POWER NORM
norm = PowerNorm(gamma=0.2, vmin=0, vmax=df.max().max())

# --- KEY CHANGE: square=True ---
ax = sns.heatmap(df, 
                 annot=True, 
                 fmt='g', 
                 annot_kws={"size": 16, "weight": "bold"},
                 cmap=custom_cmap, 
                 norm=norm,
                 cbar_kws={'label': 'Color Intensity', 'shrink': 0.5}, # Shrunk to match square height
                 linewidths=1, 
                 linecolor='white',
                 square=True) # Forces 1:1 aspect ratio for cells

# 4. QUADRANT DIVIDERS
plt.axhline(y=6, color='black', linewidth=3)
plt.axvline(x=6, color='black', linewidth=3)

# 5. TITLES AND LABELS
plt.title('YOLO Multiclass Classification\nTop-1: 73.87% | Binary: 97.53%', 
          fontsize=18, pad=20, fontweight='bold')

plt.xlabel('Predicted Labels', fontsize=14, fontweight='bold', labelpad=10)
plt.ylabel('True Labels', fontsize=14, fontweight='bold', labelpad=10)

plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(rotation=0, fontsize=11)

plt.tight_layout()
plt.savefig('replicated_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()
