import pandas as pd
import matplotlib.pyplot as plt

#################
# ImageNet
df = pd.read_csv("/vol/ideadata/ep56inew/ImageNet_VidVRD/labels/official_split/one_class_labels_final_split_removed_classes_VAL_num.csv",
    header=None,
    names=['filename', 'class', 'set'],
    usecols=[1, 2]
    )
# Only keep TRAIN and VAL sets
# df = df[df['set'].isin(['TRAIN', 'VAL'])]
# TEST set
df = df[df['set'].isin(['TEST'])]
# Count occurrences by class and set
counts = df.groupby(['class', 'set']).size().unstack(fill_value=0).sort_index()
# Filter out classes with no data
# counts = counts[(counts['TRAIN'] > 0) | (counts['VAL'] > 0)]
counts = counts[(counts['TEST'] > 0)]
#################
# # EchoNet-Dynamic
# data = {'Test': [160, 125, 992]}
# index = ['Reduced EF', 'Mid-Range EF', 'Normal EF']
# counts = pd.DataFrame(data, index=index)
#################
# # EchoNet-Pediatric
# data = {'Test': [27, 34, 307]}
# index = ['Reduced EF', 'Mid-Range EF', 'Normal EF']
# counts = pd.DataFrame(data, index=index)
#################
# EchoNet-LVH
# data = {'Test': [19, 336]}
# index = ['Non-Hypertrophic', 'Hypertrophic']
# counts = pd.DataFrame(data, index=index)
#################

# Plotting
ax = counts.plot(kind='bar', figsize=(10, 6), width=0.8)
ax.legend().remove() # Remove legend

plt.xlabel('Class', fontsize=25)
plt.ylabel('Count', fontsize=25)
# plt.title('Class Distribution by Set')
plt.xticks(rotation=0, fontsize=20)
plt.yticks(fontsize=20)
# plt.legend(fontsize=14)

plt.tight_layout()
plt.savefig("/vol/ideadata/ep56inew/myCode/pictures/IN_distribution_test_videos.png", dpi=300)
plt.close()