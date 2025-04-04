import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set up the figure and axes
fig, ax = plt.subplots(figsize=(4,2))

# Draw the grid
# vertical lines
# ax.plot([0, 0], [0, 1.5], color='black')
# ax.plot([1, 1], [0, 1.5], color='black')
# ax.plot([2, 2], [0, 1.5], color='black')
# # horizontal lines
# ax.plot([0, 2], [0, 0], color='black')
# ax.plot([0, 2], [0.75, 0.75], color='black')
# ax.plot([0, 2], [1.5, 1.5], color='black')

# Add colored rectangles to the four cells
colors = {
    "TP": "#a1d99b",  # light green
    "FN": "#fc9272",  # light red
    "FP": "#fdd0a2",  # light orange
    "TN": "#9ecae1"   # light blue
}
# Coordinates: (x, y) from bottom-left
# bottom-left corner, width, height
# Draw colored rectangles (2 wide × 1 tall)
cell_width = 2
cell_height = 1
ax.add_patch(patches.Rectangle((0, 1), cell_width, cell_height, facecolor=colors["TP"]))  # TP (top-left)
ax.add_patch(patches.Rectangle((2, 1), cell_width, cell_height, facecolor=colors["FN"]))  # FN (top-right)
ax.add_patch(patches.Rectangle((0, 0), cell_width, cell_height, facecolor=colors["FP"]))  # FP (bottom-left)
ax.add_patch(patches.Rectangle((2, 0), cell_width, cell_height, facecolor=colors["TN"]))  # TN (bottom-right)


# # Add cell labels
# ax.text(0.5, 1.5, "True\nPositives\n(TP)", ha='center', va='center', fontsize=20)
# ax.text(1.5, 1.5, "False\nNegatives\n(FN)", ha='center', va='center', fontsize=20)
# ax.text(0.5, 0.5, "False\nPositives\n(FP)", ha='center', va='center', fontsize=20)
# ax.text(1.5, 0.5, "True\nNegatives\n(TN)", ha='center', va='center', fontsize=20)
# # Add axis labels
# ax.text(-0.1, 1.5, "Actual:\nPositive", va='center', ha='right', fontsize=20)
# ax.text(-0.1, 0.5, "Actual:\nNegative", va='center', ha='right', fontsize=20)
# ax.text(0.5, 2.1, "Predicted:\nPositive", ha='center', va='bottom', fontsize=20)
# ax.text(1.5, 2.1, "Predicted:\nNegative", ha='center', va='bottom', fontsize=20)

# Updated cell labels (centers shifted downward due to smaller height)
# ax.text(0.5, 1.125, "True\nPositives\n(TP)", ha='center', va='center', fontsize=20)
# ax.text(1.5, 1.125, "False\nNegatives\n(FN)", ha='center', va='center', fontsize=20)
# ax.text(0.5, 0.375, "False\nPositives\n(FP)", ha='center', va='center', fontsize=20)
# ax.text(1.5, 0.375, "True\nNegatives\n(TN)", ha='center', va='center', fontsize=20)

# Text labels centered in each cell
fontsize = 10
ax.text(1, 1.5, "True\nPositives\n(TP)", ha='center', va='center', fontsize=fontsize)
ax.text(3, 1.5, "False\nNegatives\n(FN)", ha='center', va='center', fontsize=fontsize)
ax.text(1, 0.5, "False\nPositives\n(FP)", ha='center', va='center', fontsize=fontsize)
ax.text(3, 0.5, "True\nNegatives\n(TN)", ha='center', va='center', fontsize=fontsize)

# Axis labels for "Actual"
ax.text(-0.2, 1.5, "Actual:\nPositive", va='center', ha='right', fontsize=fontsize)
ax.text(-0.2, 0.5, "Actual:\nNegative", va='center', ha='right', fontsize=fontsize)

# Axis labels for "Predicted"
ax.text(1, 2.1, "Predicted:\nPositive", ha='center', va='bottom', fontsize=fontsize)
ax.text(3, 2.1, "Predicted:\nNegative", ha='center', va='bottom', fontsize=fontsize)

# Clean up axes
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(0, 4)
ax.set_ylim(0, 2)
ax.set_aspect('equal', adjustable='box')  # Enforce 2:1 cell shape
ax.axis('off')

# plt.title("Confusion Matrix Layout", fontsize=14)
plt.tight_layout()
plt.savefig("/vol/ideadata/ep56inew/myCode/pictures/confusion.png", dpi=300)