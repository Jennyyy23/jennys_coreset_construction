# when you import a class from a script (module) in Python,
# you automatically import any packages that were imported in that script. 
# This is because Python loads and executes the entire script when it's imported, 
# which includes all of its imports and definitions.

from jennys_coreset_construction import JennysCoreset
from save_all_frames import SaveAllFrames
from random_row_sampling import RandomRowSelector
from create_label_csv import CreateLabelCSV
from zipping import Zipping
from regular_frame_sampling import SampleFrames
from copy_files_from_csv import CopyFiles

# ************************************
# DIMENSIONALITY REDUCTION METHODS
dimred1 = "POD"
dimred2 = "PCA" 
dimred3 = "FE"
# CLUSTERING
clustering1 = "DBSCAN1"
clustering2 = "DBSCAN2"
clustering3 = "KMedoids"
# METRICS
metric1 = "wasserstein"
metric2 = "euclid"
metric3 = "spherical"
metric4 = "js"
metric5 = "tvd"

# !!! SET !!!
dimred = "FE"
clustering = "DBSCAN2"
# ************************************

# Execute all
 
# ********* ALL FRAMES **********
# 0. Save all Frames

# ********* SUBSETS *********************************************************************************************************
# 1.a Construct coreset
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
dataset = "EchoNet" # ImageNet, EchoNet
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

if dataset == "EchoNet":

    destination_path1 = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/official_split/training_sets/ED_FE_DBSCAN2_reduced"

#     destination_path1="/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/official_split/training_sets/ED_" + dimred + "_" + clustering + "/ED_" + metric1
#     destination_path2="/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/official_split/training_sets/ED_" + dimred + "_" + clustering + "/ED_" + metric2
#     destination_path3="/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/official_split/training_sets/ED_" + dimred + "_" + clustering + "/ED_" + metric3
#     destination_path4="/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/official_split/training_sets/ED_" + dimred + "_" + clustering + "/ED_" + metric4
#     destination_path5="/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/official_split/training_sets/ED_" + dimred + "_" + clustering + "/ED_" + metric5
    # # #
    log_csv = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/official_split/training_sets/ED_logs.csv"
    set_name = "TRAIN"
    set_csv = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/FileList.csv"
    set_column = 8
    source_path="/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/Videos"
    all_frames_path = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/official_split/training_sets/train_all_frames"

elif dataset == "ImageNet":
    # *** TESTING ***
    # source_path="/vol/ideadata/ep56inew/ImageNet_VidVRD/all_videos/one_video_for_testing"
    # destination_path="/vol/ideadata/ep56inew/ImageNet_VidVRD/all_videos/one_video_for_testing/frames"
    # * * *
    destination_path1="/vol/ideadata/ep56inew/ImageNet_VidVRD/subsets/training_coresets/" + dimred + "_" + clustering + "/IN_" + metric1
    destination_path2="/vol/ideadata/ep56inew/ImageNet_VidVRD/subsets/training_coresets/" + dimred + "_" + clustering + "/IN_" + metric2
    destination_path3="/vol/ideadata/ep56inew/ImageNet_VidVRD/subsets/training_coresets/" + dimred + "_" + clustering + "/IN_" + metric3
    destination_path4="/vol/ideadata/ep56inew/ImageNet_VidVRD/subsets/training_coresets/" + dimred + "_" + clustering + "/IN_" + metric4
    destination_path5="/vol/ideadata/ep56inew/ImageNet_VidVRD/subsets/training_coresets/" + dimred + "_" + clustering + "/IN_" + metric5
    # # #
    log_csv = "/vol/ideadata/ep56inew/ImageNet_VidVRD/subsets/training_coresets/IN_logs.csv"
    set_name = "TRAIN"
    set_csv = "/vol/ideadata/ep56inew/ImageNet_VidVRD/labels/official_split/one_class_labels_final_split_removed_classes_VAL_num.csv"
    set_column = 2
    source_path="/vol/ideadata/ep56inew/ImageNet_VidVRD/videos"
    all_frames_path = "/vol/ideadata/ep56inew/ImageNet_VidVRD/subsets/train_all_frames"

# EXECUTE !!!! 
# JennysCoreset(source_path, destination_path1, metric1, set_csv, set_name, set_column, dataset, dimred, all_frames_path, clustering, log_csv)
# JennysCoreset(source_path, destination_path2, metric2, set_csv, set_name, set_column, dataset, dimred, all_frames_path, clustering, log_csv)
# JennysCoreset(source_path, destination_path3, metric3, set_csv, set_name, set_column, dataset, dimred, all_frames_path, clustering, log_csv)
# JennysCoreset(source_path, destination_path4, metric4, set_csv, set_name, set_column, dataset, dimred, all_frames_path, clustering, log_csv)
# JennysCoreset(source_path, destination_path5, metric5, set_csv, set_name, set_column, dataset, dimred, all_frames_path, clustering, log_csv)

# 1.b Construct random subset (only as csv)
# the input csv has to be the label csv of train_all_frames
# EchoNet-Dynamic:
# input_csv = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/official_split/labels/ED_EF_labels_train_all_frames.csv"
# output_csv = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/official_split/labels/ED_EF_labels_random.csv"
# EchoNet-Pediatric:
# all_frames_csv = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Pediatric/A4C/labels/train_all_frames_labels.csv"
# random_csv = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Pediatric/A4C/labels/train_random_labels_distillationsize.csv"
# ImageNet:
# all_frames_csv = "/vol/ideadata/ep56inew/ImageNet_VidVRD/labels/official_split/frame_set_label_files/IN_train_all_frames.csv"
# random_csv = "/vol/ideadata/ep56inew/ImageNet_VidVRD/labels/official_split/frame_set_label_files/IN_train_random.csv"
# # # 
# num_rows = 27427
# RandomRowSelector(all_frames_csv, random_csv, num_rows)
#
# Now the random subset will be saved separately
# csv_file = random_csv
# file_folder = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Pediatric/A4C/train_all_frames"
# destination_folder = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Pediatric/A4C/random/train_random_distillationsize"
# CopyFiles(csv_file, file_folder, destination_folder)

# 1.c Construct regular subset

# **********************************************************************************************************************
# ****** LABELLING *****************************************************************************************************
# **********************************************************************************************************************

# 2. Create Label csv (not needed for random subset)

# choose between ImageNet or EchoNet
# !!!!!!!!!!!!!!!!!!!!!!!!
main_dataset = "ED" # IN, ED, EP, LVH
# !!!!!!!!!!!!!!!!!!!!!!!!

# EchoNet-Pediatric
if main_dataset == "EP":

        frame_subset_path = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Pediatric/A4C/random/train_random_frames_DB2reduced"
        destination_csv = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Pediatric/A4C/labels/train_random_frames_DB2reduced.csv"
        source_csv = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Pediatric/A4C/FileList_EF_categories.csv"
        label_column = 12

# EchoNet-Dynamic
elif main_dataset == "ED":

    frame_subset_path = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/jennys_coresets_smaller/1_percent_again_wasserstein"
    destination_csv = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/jennys_coresets_smaller/labels/1_percent_again_wasserstein.csv"
    
#     frame_subset_path1 = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/official_split/training_sets/ED_" + dimred + "_" + clustering + "/ED_" + metric1
#     destination_csv1 = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/official_split/labels_EF_categories/labels_" + dimred + "_" + clustering + "/" + "ED_" + metric1 + ".csv"
#     frame_subset_path2 = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/official_split/training_sets/ED_" + dimred + "_" + clustering + "/ED_" + metric2
#     destination_csv2 = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/official_split/labels_EF_categories/labels_" + dimred + "_" + clustering + "/" + "ED_" + metric2 + ".csv"
#     frame_subset_path3 = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/official_split/training_sets/ED_" + dimred + "_" + clustering + "/ED_" + metric3
#     destination_csv3 = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/official_split/labels_EF_categories/labels_" + dimred + "_" + clustering + "/" + "ED_" + metric3 + ".csv"    
#     frame_subset_path4 = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/official_split/training_sets/ED_" + dimred + "_" + clustering + "/ED_" + metric4
#     destination_csv4 = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/official_split/labels_EF_categories/labels_" + dimred + "_" + clustering + "/" + "ED_" + metric4 + ".csv"    
#     frame_subset_path5 = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/official_split/training_sets/ED_" + dimred + "_" + clustering + "/ED_" + metric5
#     destination_csv5 = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/official_split/labels_EF_categories/labels_" + dimred + "_" + clustering + "/" + "ED_" + metric5 + ".csv"    
#     # # #
    source_csv = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/FileList_EF_categories.csv"
    label_column = 9

# EchoNet-LVH
elif main_dataset == "LVH":

        frame_subset_path = "/vol/ideadata/ep56inew/EchoNet/EchoNet-LVH/frame_sets/LVH_random_jennys_cs"
        destination_csv = "/vol/ideadata/ep56inew/EchoNet/EchoNet-LVH/labels/LVH_random_jenny.csv"
        # # #
        source_csv = "/vol/ideadata/ep56inew/EchoNet/EchoNet-LVH/MeasurementsList_new_clean_ASH_official_split_more_test.csv"
        label_column = 4

elif main_dataset == "IN":

    frame_subset_path1 = "/vol/ideadata/ep56inew/ImageNet_VidVRD/subsets/training_coresets/" + dimred + "_" + clustering + "/IN_" + metric1
    destination_csv1 = "/vol/ideadata/ep56inew/ImageNet_VidVRD/labels/official_split/" + dimred + "_" + clustering + "/" + main_dataset + "_" + metric1 + ".csv"
    frame_subset_path2 = "/vol/ideadata/ep56inew/ImageNet_VidVRD/subsets/training_coresets/" + dimred + "_" + clustering + "/IN_" + metric2
    destination_csv2 = "/vol/ideadata/ep56inew/ImageNet_VidVRD/labels/official_split/" + dimred + "_" + clustering + "/" + main_dataset + "_" + metric2 + ".csv"
    frame_subset_path3 = "/vol/ideadata/ep56inew/ImageNet_VidVRD/subsets/training_coresets/" + dimred + "_" + clustering + "/IN_" + metric3
    destination_csv3 = "/vol/ideadata/ep56inew/ImageNet_VidVRD/labels/official_split/" + dimred + "_" + clustering + "/" + main_dataset + "_" + metric3 + ".csv"
    frame_subset_path4 = "/vol/ideadata/ep56inew/ImageNet_VidVRD/subsets/training_coresets/" + dimred + "_" + clustering + "/IN_" + metric4
    destination_csv4 = "/vol/ideadata/ep56inew/ImageNet_VidVRD/labels/official_split/" + dimred + "_" + clustering + "/" + main_dataset + "_" + metric4 + ".csv"
    frame_subset_path5 = "/vol/ideadata/ep56inew/ImageNet_VidVRD/subsets/training_coresets/" + dimred + "_" + clustering + "/IN_" + metric5
    destination_csv5 = "/vol/ideadata/ep56inew/ImageNet_VidVRD/labels/official_split/" + dimred + "_" + clustering + "/" + main_dataset + "_" + metric5 + ".csv"
    # # # 
    source_csv = "/vol/ideadata/ep56inew/ImageNet_VidVRD/labels/official_split/one_class_labels_final_split_removed_classes_VAL_num.csv"
    label_column = 1
    # # #

else:
    raise ValueError("Dataset not defined.")

# EXECUTE !!!
# old:
CreateLabelCSV(frame_subset_path, source_csv, destination_csv, main_dataset, label_column)

# new: 
# frame_subset_paths = [frame_subset_path1, frame_subset_path2, frame_subset_path3, frame_subset_path4, frame_subset_path5]
# destination_csvs = [destination_csv1, destination_csv2, destination_csv3, destination_csv4, destination_csv5]
# for frame_path, dest_csv in zip(frame_subset_paths, destination_csvs):
#     CreateLabelCSV(frame_path, source_csv, dest_csv, main_dataset, label_column)
 
# ******************** ZIPPING ******************************************************
# 3. Zip all frames (not needed for random subset)

if main_dataset == "EP":

        # frame_subset_path = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Pediatric/A4C/train_random_frames"
        # EP
        output_zip = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Pediatric/A4C/zip_files/EP_random_DB2reduced.zip"

if main_dataset == "LVH":

        frame_subset_path = "/vol/ideadata/ep56inew/EchoNet/EchoNet-LVH/frame_sets/LVH_random_dd"
        output_zip = "/vol/ideadata/ep56inew/EchoNet/EchoNet-LVH/zip_files/LVH_random_dd.zip"

elif main_dataset == "ED":

    output_zip = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/jennys_coresets_smaller/zip_files/ED_FEDB2_1_percent_again.zip"

#     output_zip1 = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/official_split/zip_files/zip_files_ED_" \
#             + dimred + "_" + clustering + "/ED_" + metric1 + ".zip"
#     output_zip2 = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/official_split/zip_files/zip_files_ED_" \
#             + dimred + "_" + clustering + "/ED_" + metric2 + ".zip"
#     output_zip3 = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/official_split/zip_files/zip_files_ED_" \
#             + dimred + "_" + clustering + "/ED_" + metric3 + ".zip"
#     output_zip4 = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/official_split/zip_files/zip_files_ED_" \
#             + dimred + "_" + clustering + "/ED_" + metric4 + ".zip"
#     output_zip5 = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/official_split/zip_files/zip_files_ED_" \
#             + dimred + "_" + clustering + "/ED_" + metric5 + ".zip"

elif main_dataset == "IN":

    output_zip1 = "/vol/ideadata/ep56inew/ImageNet_VidVRD/subsets/training_coresets/" + dimred + "_" + clustering + "/zip_files_IN_" \
            + dimred + "_" + clustering + "/" + main_dataset + "_" + metric1 + ".zip"
    output_zip2 = "/vol/ideadata/ep56inew/ImageNet_VidVRD/subsets/training_coresets/" + dimred + "_" + clustering + "/zip_files_IN_" \
            + dimred + "_" + clustering + "/" + main_dataset + "_" + metric2 + ".zip"      
    output_zip3 = "/vol/ideadata/ep56inew/ImageNet_VidVRD/subsets/training_coresets/" + dimred + "_" + clustering + "/zip_files_IN_" \
            + dimred + "_" + clustering + "/" + main_dataset + "_" + metric3 + ".zip"      
    output_zip4 = "/vol/ideadata/ep56inew/ImageNet_VidVRD/subsets/training_coresets/" + dimred + "_" + clustering + "/zip_files_IN_" \
            + dimred + "_" + clustering + "/" + main_dataset + "_" + metric4 + ".zip"
    output_zip5 = "/vol/ideadata/ep56inew/ImageNet_VidVRD/subsets/training_coresets/" + dimred + "_" + clustering + "/zip_files_IN_" \
            + dimred + "_" + clustering + "/" + main_dataset + "_" + metric5 + ".zip"

# EXECUTE !!!

# old
Zipping(frame_subset_path, output_zip)

# new
# output_zip_files = [output_zip1, output_zip2, output_zip3, output_zip4, output_zip5]
# for frame_folder, output_zip in zip(frame_subset_paths, output_zip_files):
#     Zipping(frame_folder, output_zip)