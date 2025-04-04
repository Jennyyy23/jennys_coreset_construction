import shutil
import os
import csv
import time 


class CopyFiles:
    """
    Copies files listed in the first column of a CSV file to a specified destination directory.

    Parameters:
        csv_file (str): Path to the CSV file containing the list of filenames (one per row, in the first column).
        file_folder (str): Path to the folder containing the source files.
        destination_folder (str): Path where the listed files will be copied.
    """

    def __init__(self, csv_file, file_folder, destination_folder):

        self.csv_file = csv_file
        self.file_folder = file_folder
        self.destination_folder = destination_folder

        # execute
        start_time = time.time()
        print(start_time)

        self.copy_files_from_csv()

        end_time = time.time()
        print(end_time)
        elapsed_time = end_time - start_time
        print(f"elapsed time for copying random files: {elapsed_time} seconds")

    def copy_files_from_csv(self):

        # Ensure the destination folder exists
        if not os.path.exists(self.destination_folder):
            os.makedirs(self.destination_folder)
        
        with open(self.csv_file, mode='r') as infile:
            reader = csv.reader(infile)
            
            for row in reader:
                source_file = row[0]  # First column should contain the file paths
                if os.path.exists(os.path.join(self.file_folder, source_file)):  # Check if the file exists
                    # Copy the file to the destination folder
                    shutil.copy(os.path.join(self.file_folder, source_file), self.destination_folder)
                    print(f"Copied {source_file} to {self.destination_folder}")
                else:
                    print(f"File not found: {source_file}")

# execute
if __name__ == "__main__":

    csv_file = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/jennys_coresets_smaller/labels/random_1_percent.csv"
    file_folder = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/official_split/training_sets/train_all_frames"
    destination_folder = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/jennys_coresets_smaller/random/random_1_percent"

    CopyFiles(csv_file, file_folder, destination_folder)