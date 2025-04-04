import csv

# Add EF categories to csv (reduced, borderline, normal)
def add_categories(original_csv_file, output_csv_file):

    with open(original_csv_file, mode="r") as infile, open(output_csv_file, mode="w", newline="") as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # Add your new column header here
        headers = next(reader)
        headers.append('new_column_name')
    
        # Write the header to the new file
        writer.writerow(headers)

        for row in reader:
            # Convert row[1] to a float because 
            value = float(row[1])

            # reduced EF
            if value <= 40:
                row.append(0)

            # borderline EF (cutoff value 50 for EchoNet-Dynamic (adults))
            # elif 40 < value < 50:
            #     row.append(1)

            # mildly reduced EF (cutoff value 55 for EchoNet-Pediatric (children))
            elif 40 < value < 55:
                row.append(1)

            # normal EF (50 for EchoNet-Dynamic, 55 for EchoNet-Pediatric)
            elif 55 <= value:
                row.append(2)

            writer.writerow(row)

# Execute
# EchoNet-Dynamic
# original_csv_file = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/FileList.csv"
# output_csv_file = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/FileList_EF_categories.csv"
# EchoNet-Pediatric
# original_csv_file = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Pediatric/A4C/FileList.csv"
# output_csv_file = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Pediatric/A4C/FileList_EF_categories.csv"
# add_categories(original_csv_file, output_csv_file)

# Count categories
def histogram(imput_csv, column_idx):

    categories = {}

    with open(imput_csv, mode="r") as infile:

        reader = csv.reader(infile)
        rows = list(reader)

        # Skip the header
        rows = rows[1:]

        for row in rows:

            if row[6] == "TRAIN":

                value = row[column_idx]

                if value not in categories:
                    categories[value] = 1
                else:
                    categories[value] += 1

        print(categories)

# Execute
# histogram("/vol/ideadata/ep56inew/EchoNet/EchoNet-Pediatric/A4C/FileList_EF_categories.csv", 12)


# plot 
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the CSV file
# EchoNet-Dynamic
# df = pd.read_csv('/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/FileList_EF_categories.csv')
# EchoNet-Pediatric
# df = pd.read_csv('/vol/ideadata/ep56inew/EchoNet/EchoNet-Pediatric/A4C/FileList_EF_categories.csv')
# ImageNet
# df = pd.read_csv('/vol/ideadata/ep56inew/ImageNet_VidVRD/labels/official_split/one_class_labels_final_split_removed_classes_VAL_num.csv')
# EchoNet-LVH
df = pd.read_csv('/vol/ideadata/ep56inew/EchoNet/EchoNet-LVH/MeasurementsList_new_clean_ASH_official_split_more_test.csv')

# Step 2: Count occurrences in the second column
counts = df[df.iloc[:, 5] == "val"].iloc[:, 4].value_counts()
# ImageNet: 1, ED: 9, LVH: 4
# counts = df.iloc[:, 4].value_counts()

# Sort the Series by index (x-axis) in ascending order
sorted_counts = counts.sort_index()

# Step 3: Plot the histogram
plt.figure(figsize=(8, 5))
sorted_counts.plot(kind='bar', color='skyblue')
# plt.title('Distribution of Object Categories in ImageNet-VidVRD')
plt.title('Distribution of Classes across Validation Set')
plt.xlabel('')
plt.ylabel('Frequency')
# plt.xticks(rotation=0)  # Rotate x-axis labels if needed

# Set custom x-axis tick labels for values 0, 1, 2
x_labels = ['non-hypertrophic', 'hypertrophic']
# x_labels = ['Reduced', 'Mildly Reduced', 'Preserved']  # Replace with your strings
plt.xticks(ticks=range(len(x_labels)), labels=x_labels, rotation=0)  # Align labels with bars

plt.savefig('/vol/ideadata/ep56inew/EchoNet/EchoNet-LVH/distribution_val.png')