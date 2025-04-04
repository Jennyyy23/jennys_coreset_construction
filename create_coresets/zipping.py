import os
import zipfile

class Zipping:
    """
    Creates a ZIP archive containing all files from a specified folder.

    Parameters:
        folder_path (str): Path to the folder whose contents should be zipped.
        output_zip (str): Path where the ZIP file will be saved.
    """

    def __init__(self, folder_path, output_zip):

        self.folder_path = folder_path
        self.output_zip = output_zip

        # Get the directory part of the destination path
        destination_dir = os.path.dirname(self.output_zip)
        # Check if the directory part exists; if not, create it
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)

        self.zip_without_folders()

    def zip_without_folders(self):
        count = 0
        # Create a zip file object
        with zipfile.ZipFile(self.output_zip, 'w') as zipf:
            # Iterate over all the files in the directory
            for root, dirs, files in os.walk(self.folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Add file to the zip without including the directory structure
                    zipf.write(file_path, arcname=file)
                    # Print the file being zipped
                    print(f"Adding {file} to zip.")
                    count += 1

        print(f"Zipping completed. {count} files are saved to {self.output_zip}")

if __name__ == "__main__":

    print("Main block in zipping.py executed!")

    # Echonet Dynamic
    folder_to_zip = '/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/distillation_frames_smaller/1_percent'
    output_zip_file = '/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/distillation_frames_smaller/dd_1_percent.zip'
    Zipping(folder_to_zip, output_zip_file)

    # Echonet Pediatric
    # folder_to_zip = '/vol/ideadata/ep56inew/EchoNet/EchoNet-Pediatric/A4C/train_js'
    # output_zip_file = '/vol/ideadata/ep56inew/EchoNet/EchoNet-Pediatric/A4C/zip_files/EP_train_js.zip'
    # Zipping(folder_to_zip, output_zip_file)

    # ImageNet VidVRD
    # folder_to_zip = '/vol/ideadata/ep56inew/ImageNet_VidVRD/train/train_coreset_spherical'
    # output_zip_file = '/vol/ideadata/ep56inew/ImageNet_VidVRD/zip_files/IN_train_spherical.zip'
    # zip_without_folders(folder_to_zip, output_zip_file)