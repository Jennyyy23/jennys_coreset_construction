import pandas as pd

# IMAGENET-VIDVRD:

# Read the CSV file (no header)
# df = pd.read_csv(
#     "/vol/ideadata/ep56inew/ImageNet_VidVRD/labels/official_split/one_class_labels_final_split_removed_classes_VAL_num.csv",
#     header=None,
#     names=['filename', 'category', 'set'],
#     usecols=[1, 2]
# )

# # Group by 'category' and 'set', then count how often each appears
# counts = df.groupby(['category', 'set']).size().unstack(fill_value=0).sort_index()

# # Show the result
# print("Category frequency per set:")
# print(counts)

# ECHONET:
# Read the CSV file
# EchoNet-Dynamic
# df = pd.read_csv("/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/FileList_EF_categories.csv")  # Replace with your file path
# EchoNet-Pediatric:
# df = pd.read_csv("/vol/ideadata/ep56inew/EchoNet/EchoNet-Pediatric/A4C/FileList_EF_categories.csv")
# EchoNet-LVH
df = pd.read_csv("/vol/ideadata/ep56inew/EchoNet/EchoNet-LVH/MeasurementsList_new_clean_ASH_official_split_more_test.csv")

# Group by category (new_column_name) and split, then count
# EchoNet-Dynamic
# group_columns = ['new_column_name', 'Split']
# EchoNet-Pediatric
# group_columns = ['EFCategory', 'Split']
# EchoNet-LVH
group_columns = ['ASH', 'split']
# # #
counts = df.groupby(group_columns).size().unstack(fill_value=0).sort_index()

# Show the result
print("Category frequency per split:")
print(counts)