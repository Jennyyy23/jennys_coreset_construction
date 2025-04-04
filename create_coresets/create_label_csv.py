import os
import csv

class CreateLabelCSV:
    """
    Creates a CSV file containing frame names and their corresponding labels.

    Parameters:
        folder_path (str): Path to the folder containing extracted frames.
        source_csv (str): Path to the CSV file containing video names and labels.
        destination_csv (str): Path where the output label CSV will be saved.
        main_dataset (str): Name of the dataset. Options: "ImageNet" or "EchoNet".
        label_column (str): Name of the column in `source_csv` that contains the labels.
    """
    
    def __init__(self,folder_path, source_csv, destination_csv, main_dataset, label_column):

        dataset = "EchoNet"

        if main_dataset == "IN":
            dataset = "ImageNet"

        self.folder_path = folder_path
        self.source_csv = source_csv
        self.destination_csv = destination_csv
        self.main_dataset = dataset
        self.label_column = label_column

        # Get the directory part of the destination path
        destination_dir = os.path.dirname(self.destination_csv)
        # Check if the directory part exists; if not, create it
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)

        self.match_and_copy_csv()

        # count rows of final csv
        self.count_csv_rows(self.destination_csv)


    def match_and_copy_csv(self):
        """
        This method takes a folder with frames of videos as first argument.
        The video names need to be provided in the source csv file and have to have the same name as the frames (first part).
        The split condition needs to be set, such that the first part of the frame names can be compared to the video names.
        The destination csv will contain two columns: first one with frame name, and second one with the corresponding label
        from the source csv. The column with the correstponding label needs to be set. 
        The values need to be set at "!!!!!".
        """

        # Read the source CSV into a list of rows
        with open(self.source_csv, mode='r', newline='', encoding='utf-8') as src_file:
            reader = csv.reader(src_file)
            if self.main_dataset == "ImageNet":
            # Read rows and remove '.mp4' from the first column
                csv_rows = [[row[0].replace('.mp4', ''), *row[1:]] for row in reader]
            else:
                csv_rows = [row for row in reader]

        # Open the destination CSV for writing
        with open(self.destination_csv, mode='w', newline='', encoding='utf-8') as dest_file:
            writer = csv.writer(dest_file)

            # Loop through the directory and check for matching file names in the CSV
            for file_name in os.listdir(self.folder_path):
                file_full_path = os.path.join(self.folder_path, file_name)

                # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
                # Frame name split condition
                # EchoNet
                if self.main_dataset == "EchoNet":
                    filename_no_ext = file_name.split("_")[0]

                elif self.main_dataset == "ImageNet":
                # VidVRD
                    parts = file_name.split("_")
                    filename_no_ext = "_".join(parts[:-1])
                else:
                    raise ValueError(f"Specified dataset not available. Either choose EchoNet or ImageNet.")
                # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

                # Check if it's a file (instead of a folder)
                if os.path.isfile(file_full_path):
                    # Find if the file name matches any in the first column of the CSV rows
                    for row in csv_rows:
                        if filename_no_ext == row[0].strip():  # Match found

                            # Write the file name and the corresponding column value to the destination CSV
                            writer.writerow([file_name, row[self.label_column]])
                            print(f"Found and wrote: {file_name} -> {row[self.label_column]}")
                            break  # No need to check further rows once a match is found
                    else:
                        print(f"File '{file_name}' not found in source CSV.")
                else:
                    print(f"'{file_name}' is not a file.")

        print(f"Process completed. Matching file names and data have been written to {self.destination_csv}")


    def count_csv_rows(self, filename, skip_header=False):
        """This method counts the rows in the csv file"""
        with open(filename, mode='r') as file:
            reader = csv.reader(file)
            if skip_header:
                next(reader, None)  # Skip the header row
            row_count = sum(1 for row in reader)
        print(f"number of rows: {row_count}")

if __name__ == "__main__":

    print("Main block in create_label_csv.py executed!")

    # execute (EchoNet Dynamic)
    # frame_subset_path = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/Train_Videos/ED_train_spherical"
    # source_csv = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/ef_categories_1.csv" # do not change that !!!
    # destination_csv = os.path.join("/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/frame_labels/EF_regression_labels", "ED_EF_regression_train_spherical.csv")
    # main_dataset = "EchoNet",
    # label_column = 1

    # execute (EchoNet Pediatric A4C)
    frame_subset_path = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Pediatric/A4C/val_frames"
    source_csv = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Pediatric/A4C/FileList.csv"
    destination_csv = os.path.join("/vol/ideadata/ep56inew/EchoNet/EchoNet-Pediatric/A4C/labels/frame_labels_age", "EP_train_labels_age_val.csv")

    # execute (ImageNet VidVRD)
    # folder_path = "/vol/ideadata/ep56inew/ImageNet_VidVRD/train/train_coreset_spherical"
    # source_csv = "/vol/ideadata/ep56inew/ImageNet_VidVRD/labels/labels_range_from_0/num_one_class_labels_final_no_mp4.csv"
    # destination_csv = os.path.join("/vol/ideadata/ep56inew/ImageNet_VidVRD/labels/labels_range_from_0", "IN_labels_train_spherical.csv")

    # CreateLabelCSV(frame_subset_path, source_csv, destination_csv, main_dataset, label_column)