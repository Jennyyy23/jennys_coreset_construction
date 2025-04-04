import numpy as np
import cupy as cp
import cv2
from scipy import linalg as la
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from scipy.stats import entropy
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import jensenshannon
from scipy.special import softmax
from sklearn.cluster import DBSCAN
from sklearn_extra.cluster import KMedoids
import random
import math 
import time
import os
import csv
from ResNet_Feature_Embedding import ResNet_Feature_Embedding
from PIL import Image
import shutil
class JennysCoreset():
    """
    This class constructs a frame coreset of videos based on Jenny's clustering and subset selection approach.
    Metrics can be implemented in the method "construct_distance_matrix".
    Initializer takes source path of all videos, frame coreset destination path and metric.
    Currently implemented metrics: "wasserstein", "euclid", "js", "spherical", "total variation distance".
    """

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #                     CONSTRUCTOR                       #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    
    def __init__(self, source_path, destination_path, metric, set_csv, set_name, set_column, dataset, dimred, all_frames_path, clustering, log_csv, count): 
        # initialize time to start measuring
        start_time = time.time()

        self.source = source_path
        self.destination = destination_path
        self.metric = metric
        self.set_csv = set_csv
        self.set_name = set_name # either "TRAIN", "TEST", or "VAL"
        self.set_column = set_column
        self.dataset = dataset
        self.dimred = dimred # either "POD", "PCA", "FE"
        self.all_frames_path = all_frames_path
        self.clustering = clustering
        self.log_csv = log_csv
        self.count = count

        # new: for tracking average amount of clusters (05.11.2024)
        self.total_amount_of_clusters = 0
        self.cluster_frame_fraction = 0
        self.cluster_frame_fraction_sum = 0
        self.processed_videos = 0

        # Write header to log CSV
        with open(self.log_csv, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['batch: ', self.count])
            writer.writerow(['coreset:', dataset, dimred, clustering, metric])

        # Check if destination path exists; if not, create it
        if not os.path.exists(self.destination):
            # Create the directory
            os.makedirs(self.destination)  

        # count processed videos
        self.count = 0

        # count all videos:
        # Count only the .mp4 and .avi files
        num_video_files = sum(1 for item in os.listdir(source_path) if item.lower().endswith('.mp4') or item.lower().endswith('.avi'))
        print(f"number of videos: {num_video_files}")

        # initialize model for ResNet Embedding
        if self.dimred == "FE":
            self.embedding = ResNet_Feature_Embedding()

        # loop through all videos in source location:
        for item in os.listdir(source_path):

            if not item.lower().endswith('.mp4') and not item.lower().endswith('.avi'):
                # Skip non-MP4 and non-avi files and move to the next file
                print(f'The file {item} is not an mp4 or avi video.')
                continue

            # check if you only use videos from the desired set (TRAIN, VAL or TEST, specified as set_name)
            if self.check_value(item, self.set_csv, self.set_name) == False:
                # print(f'The file {item} is not a video belonging to the current split.')
                continue

            # for debugging: check only one video (requires most memory)
            # if item == "ILSVRC2015_train_00082000.mp4":
            #     continue

            item_path = os.path.join(source_path, item)

            # loop through all videos
            if os.path.isfile(item_path):
                start_time_one_video = time.time()
                self.processed_videos += 1
                self.total_amount_of_clusters = 0

                self.count+=1
                print(f'Video {self.count} out of {num_video_files}: {item}')

                # split the string for the image name later
                split_string = item.split('.')
                self.video_name = split_string[0]
                print(f"the current video name is {self.video_name}")

                # # # # # # # # # # # # # # # # # # # # # # # # # #
                # # # # # FEATURE EMBEDDING OF IMAGES # # # # # # #
                # # # # # # # # # # # # # # # # # # # # # # # # # #

                # POD or PCA feature extraction
                if self.dimred in ["POD", "PCA"]:
                    # import movie
                    movie, _ , bgr_frames = self.ImportMovie(source_path + '/' + item)

                    # compress into matrix
                    flat_images_matrix = self.compress_video(movie)

                    # perform frame extraction 
                    if self.dimred == "POD":
                        C = self.pod(flat_images_matrix)
                    elif self.dimred == "PCA":
                        # Step 1: Compute the mean of each row (feature mean across all samples)
                        mean_vector = np.mean(flat_images_matrix, axis=1, keepdims=True)
                        # Step 2: Center the matrix by subtracting the mean vector from each column
                        centered_matrix = flat_images_matrix - mean_vector
                        # PCA is the same as POD but on centered matrix (just the variance is considered)
                        C = self.pod(centered_matrix)

                # feature extraction with ResNet18
                elif self.dimred == "FE":

                    self.image_name_list, C = self.cnn_feature_embedding(self.embedding)
                    # Suppose `matrix` is your matrix
                    has_nan = np.isnan(C).any()
                    if has_nan:
                        print("The dim-reduced matrix contains NaN values.")
                    else:
                        print("The dim-reduced matrix does not contain any NaN values.")

                    # if self.clustering in ["DBSCAN1", "DBSCAN2"]: # keine Ahnung was ich mir dabei gedacht habe?
                    self.frames = len(self.image_name_list) # nur bei FE

                else:
                    raise ValueError("Dimensionality reduction method not defined.")

                # # # # # # # # # # # # # # # # # # # # # # # # # #
                # # # # # #         Clustering          # # # # # #
                # # # # # # # # # # # # # # # # # # # # # # # # # #         

                # create distance matrix
                distance_matrix = self.construct_distance_matrix(C)
                # Suppose `matrix` is your matrix
                has_nan = np.isnan(distance_matrix).any()
                if has_nan:
                    print("The distance matrix contains NaN values.")
                else:
                    print("The distance matrix does not contain any NaN values.")

                if self.clustering in ["DBSCAN1", "DBSCAN2"]:

                    # return closest pairs mean
                    pairs_mean = self.closest_pairs_mean(distance_matrix)

                    # perform first-round dbscan
                    labels, noise_amount = self.dbscan(pairs_mean, distance_matrix)
                
                elif self.clustering == "KMedoids":

                    labels = self.k_medoids(distance_matrix)

                # group cluster indices into dictionaries
                self.sample_groups, self.category_counts = self.give_me_clusters(labels)

                # perform second round of DBSCAN clustering if desired 
                if self.clustering == "DBSCAN2":

                    # second round dbscan
                    self.second_round_dbscan(distance_matrix, labels, noise_amount)

                # return index subset
                self.sample_from_clusters()

                # select frames (not needed anymore because directly done in method save_images)
                # X_subset_direct = self.select_frames(flat_images_matrix)

                # **************************************************
                # **************    SAVE IMAGES     ****************
                # **************************************************

                # save images
                if self.dimred != "FE":
                    self.save_images(destination_path, self.video_name, bgr_frames)

                else: # if FE as dimensionality reduction
                    image_subset_list = self.select_frames_from_img_list()
                    self.save_images_1(image_subset_list)

                # **************************************************

                # print time for video processing
                end_time_one_video = time.time()
                elapsed_time_one_video = end_time_one_video - start_time_one_video
                print(f"Elapsed time per video processing: {elapsed_time_one_video} seconds")

                # calculate cluster fraction
                self.cluster_frame_fraction = self.total_amount_of_clusters/self.frames
                self.cluster_frame_fraction_sum += self.cluster_frame_fraction

            elif os.path.isdir(item_path):
                print(f'Directory: {item}')

        # print(f"average fraction of n_clusters/n_frames: {self.cluster_frame_fraction_sum/self.processed_videos}")
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time for whole process: {elapsed_time} seconds")
        # count how many files were constructed
        files = sum(1 for entry in os.listdir(destination_path) if os.path.isfile(os.path.join(destination_path, entry)))
        print(f"amount of frames: {files}")

        # Write to CSV
        with open(self.log_csv, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['elapsed time for whole process (seconds)', elapsed_time])
            writer.writerow(['amount of frames:', files])
            writer.writerow([])

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #           JUST SELECT TRAIN, VAL OR TEST              #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # based on train, test and val split in csv for EchoNet Pediatric
    def check_value(self, item, csv_filename, target_value):
        """This function checks for the target value TRAIN, TEST or VAL"""

        # Remove the ".avi" extension from the item 
        # but .mp4 is allowed to stay because it is also in the csv file
        item_base = item.replace('.avi', '')
        
        # Open and read the CSV file
        with open(csv_filename, mode='r') as file:
            reader = csv.reader(file)

            if self.dataset == "EchoNet" or self.dataset == "ENPed" or self.dataset == "LVH":
                # Skip the header row for EchoNet csv
                next(reader)  
            
            # Loop through the rows in the CSV file
            for row in reader:
                # Check if the first column (or relevant column) matches the item_base
                if row[0] == item_base:  # Assuming the item is in the first column
                    # Check if the set column contains the target value
                    if row[self.set_column] == target_value:
                        return True  # Condition met
                    else:
                        return False
        
        return False  # Condition not met
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #                    VIDEO METHODS                      #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def rgb2gray(self, rgb):
        """This function converts RGB images to grayscale"""
        # we need greyscale images to have only one channel
        # such that a video can be represented in a 2D matrix istead of 3 x 2D
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    

    def ImportMovie(self, fname):
        """Main movie import function."""
        # Open video
        vidcap = cv2.VideoCapture(fname)
        # ATTENTION! OpenCV uses BGR format. Needs to be converted back later before saving image.
        FrameRate = vidcap.get(cv2.CAP_PROP_FPS)
        # Import first video frame
        success,image = vidcap.read()
        count = 0
        success = True
        movie = []
        bgr_frames = []
        # Import other frames until end of video
        while success:
            bgr_frames.append(image)
            movie.append(self.rgb2gray(image))
            # still save rgb values for later
            success, image = vidcap.read()
            count += 1
        # Convert to array
        movie = np.array(movie)
        # Display some summary information
        # print("==========================================")
        # print("           Video Import Summary           ")
        # print("------------------------------------------")
        # print("   Imported movie: ", fname)
        # print(" Frames extracted: ", count)
        # print("Frames per second: ", FrameRate)
        # print("      data shape = ", movie.shape)
        # print("==========================================")
        return movie, FrameRate, bgr_frames
    

    def mat2gray(self, image):
        """This function takes in a matrix (two-dimensional array) `image` and returns a matrix that can be plotted as a grayscale image."""
        out_min = np.min(image[:])
        out_max = np.max(image[:])
        idx = np.logical_and(image > out_min, image < out_max)
        image[image <= out_min] = 0;
        image[image >= out_max] = 255;
        image[idx] = ( 255/(out_max - out_min) ) * (image[idx] - out_min)
        return image


    def compress_video(self, movie):
        """This function compresses a video into a matrix with columns as frames"""
        movie_shape = movie.shape
        self.frames = movie_shape[0]
        self.height = movie_shape[1]
        self.width = movie_shape[2]
        flat_images_matrix_pre = np.reshape(movie, (self.frames, self.height * self.width))
        # transpose because we want the observations as columns (observations are our time component)
        flat_images_matrix = np.transpose(flat_images_matrix_pre)
        return flat_images_matrix
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #             FEATURE EXTRACTION METHODS                #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def pod(self, flat_images_matrix):
        """
        Performs proper orthogonal decomposition.

        Parameters:
        flat_images_matrix (numpy.ndarray): The input data matrix with shape (n_features, n_samples).
        
        Returns:
        C (numpy.ndarray): The data projected onto the modes.
        """
        # for many frames compute SVD on CPU (because more memory is available there)
        if self.frames > 0:
            # compute U_big, Sigma, V through SVD
            U_big, _ , _ = la.svd(flat_images_matrix, full_matrices=False)

        else:
            # compute on GPU for less than 1000 frames:
            cp_flat_images_matrix = cp.asarray(flat_images_matrix)
            U_big, _ , _ = cp.linalg.svd(cp_flat_images_matrix, full_matrices=False)
            del cp_flat_images_matrix  # Free GPU memory after SVD

        # determine the number of modes
        # set modes as the first 10% or at most to 10
        # modes = math.ceil(0.1 * self.frames)
        # if modes > 10:
        #     modes = 10
        # elif modes < 5:
        #     modes = 5

        # Determine number of modes (between 5 and 10)
        modes = min(max(math.ceil(0.1 * self.frames), 5), 10)

        # Find out our coefficients for the sum of coefficients and POD-modes.
        # The matrix of C-values is our projection into lower dimension. 
        # We know: X(i) = U_r C(i) => C(i) = U_r* X(i)
        U_r = U_big[:, :modes]

        # conjugate transpose of U
        U_conjugate = U_r.conj().T

        # Compute projection matrix C
        if self.frames > 0:
            C = U_conjugate @ flat_images_matrix

        else:
            # Calculate C on GPU, then transfer to CPU
            # lower-dimensional matrix C
            C_cp = U_conjugate @ cp.asarray(flat_images_matrix)
            # Convert the result back to a NumPy array
            C = cp.asnumpy(C_cp)
            del C_cp  # Free GPU memory

        return C
    
    # This approach is highly inefficient due to computing the covariance matrix 
    def pca_eigendecomposition(self, flat_images_matrix):
        """
        Performs Principal Component Analysis (PCA) using eigendecomposition on a matrix of vectors.

        Parameters:
        flat_images_matrix (numpy.ndarray): The input data matrix with shape (n_features, n_samples).

        Returns:
        projected_data (numpy.ndarray): The data projected onto the principal components.
        """
        # Step 1: Center the data (subtract the mean of each feature)
        data_mean = np.mean(flat_images_matrix, axis=1, keepdims=True)
        centered_data = flat_images_matrix - data_mean

        # Step 2: Compute the covariance matrix (for (n_features, n_samples) shape)
        covariance_matrix = np.cov(centered_data, rowvar=True)

        # Step 3: Perform eigendecomposition on the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Step 4: Sort eigenvectors by eigenvalues in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1] # last part for descending instead of ascending
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Step 5: Select the top components (if specified)
        components = math.ceil(0.1 * self.frames)
        if components > 10:
            components = 10
        elif components < 5:
            components = 5
        eigenvalues = eigenvalues[:components]
        eigenvectors = eigenvectors[:, :components]

        # Step 6: Project the data onto the selected principal components
        projected_data = eigenvectors.T @ centered_data

        return projected_data
    

    def cnn_feature_embedding(self, embedding):
        """
        This method uses ResNet18 feature embedding for dimensionality reduction.
        Returns a vector matrix with features of dimension 512.
        """
        print("cnn feature embedding started!")

        # for selecting subset of images later
        image_name_list = []
        # for concatenating into a dimensionality-reduced matrix
        embedded_array_list = []

        # loop through frames in all_frames folder which correspond to current video name
        for item in os.listdir(self.all_frames_path):

            img_path = os.path.join(self.all_frames_path, item)

            if self.dataset == "EchoNet" or self.dataset == "ENPed" or self.dataset == "LVH":
                image = item.split("_")[0]
                # print(image)

            elif self.dataset == "ImageNet":
                parts = item.split("_")
                image = "_".join(parts[:-1])

            if image == self.video_name:

                # print("image found!")
                image_name_list.append(item)
                embedded_vector = embedding.get_feature_embedding(img_path)
                embedded_array_list.append(embedded_vector)
        
        # print(embedded_array_list) # list is empty :(
        # create matrix out of all embedded vectors
        embedded_images_matrix = np.concatenate(embedded_array_list, axis=1)

        print("cnn feature embedding finished!")

        return image_name_list, embedded_images_matrix

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #            METHODS FOR DISTANCE MEASURES              #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def radial_projection(self, vector):
        """This function projects a vector onto the unit sphere"""
        # Normalize the data to lie on the unit sphere
        norms = np.linalg.norm(vector, axis=0, keepdims=True)
        projected_vector = vector / norms
        return projected_vector
    
    def spherical_distance(self, point1, point2):
        """This function computes the spherical distance between two points"""
        # Clipping the dot product ensures that any value outside the valid range is constrained back into the range [-1,1]
        # numerical computations could produce values like 1.00000000001 and that is invalid for arccos
        dot_product = np.clip(np.dot(point1, point2), -1.0, 1.0)
        return np.arccos(dot_product)

    def total_variation_distance(self, P, Q):
        return 0.5 * np.sum(np.abs(P - Q))
    # Example usage
    # P = np.array([0.1, 0.4, 0.5])
    # Q = np.array([0.2, 0.3, 0.5])
    # tvd = total_variation_distance(P, Q)

    def construct_distance_matrix(self, C):
        """This function returns the distance matrix based on the set metric"""
        distance_matrix = np.zeros((self.frames, self.frames))
        epsilon = 1e-10  # Small value for numerical stability (avoid division by zero)

        count_nan = 0

        for i in range(self.frames):
            for j in range(i, self.frames): # Compute only for upper triangular matrix

                # enforce diagonal entries to be zero (rounding errors could potentially cause non-zero values)
                if i == j:
                    distance_matrix[i, j] = 0
                else:
                    # # # # # DISTANCE MEASURE # # # # # 
                    # 1. wasserstein distance
                    if self.metric == "wasserstein":
                        dist = wasserstein_distance(C[:,i],C[:,j])

                    # 2. euclidean distance
                    elif self.metric == "euclid":
                        dist = euclidean(C[:,i],C[:,j])

                    # 3.a Jensen-Shannon-divergence and 3.b total variation distance 
                    elif self.metric in ["js", "tvd"]:
                        P = (C[:,i] - np.min(C[:,i]))/(np.max(C[:,i]) - np.min(C[:,i]) + epsilon)
                        Q = (C[:,j] - np.min(C[:,j]))/(np.max(C[:,j]) - np.min(C[:,j]) + epsilon)
                        if np.isnan(P).any() or np.isnan(Q).any():
                            print("NaN detected in P or Q after normalization.")
                            count_nan += 1
                        # if distribution is the same, the Jensen-Shannon-divergence and total variation distance is zero
                        if np.allclose(P, Q, rtol=1e-10, atol=1e-13):
                            # print("P and Q are approximately equal.")
                            dist = 0
                        else:
                            # Ensure both vectors sum to 1 (i.e., they are valid probability distributions)
                            # BUT we also want non-negative values --> use softmax to convert values
                            P_prob = softmax(P)
                            Q_prob = softmax(Q)
                            # print(f"P_prob: {P_prob}, Q_prob: {Q_prob}")
                            if np.isnan(P_prob).any() or np.isnan(Q_prob).any():
                                print("NaN detected in P or Q after normalization.")
                                # print(f"P_prob: {P_prob}, Q_prob: {Q_prob}")
                                exit()

                            if self.metric == "js":
                                # Calculate the Jensen-Shannon divergence
                                dist = jensenshannon(P_prob, Q_prob)
                                # handle nan cases 
                                if np.isnan(dist).any():
                                    print("NaN detected in distance")
                                    count_nan += 1
                                    dist = 0

                            elif self.metric == "tvd":
                                # Calculate the total variation distance 
                                dist = self.total_variation_distance(P_prob, Q_prob)

                    # 4. spherical distance with radial projection
                    elif self.metric == "spherical":
                        P = self.radial_projection(np.array(C[:,i]))
                        Q = self.radial_projection(np.array(C[:,j]))
                        dist = self.spherical_distance(P,Q)

                    else:
                        raise ValueError("Metric not defined")
                    # # # # # # # # # # # # # # # # # # # #

                    # Assign to both [i,j] and [j,i] for symmetry
                    distance_matrix[i, j] = dist
                    distance_matrix[j, i] = dist

        print(f"amount of nan values: {count_nan}")

        return distance_matrix
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #                METHODS FOR CLUSTERING                 #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    
    def closest_pairs_mean(self, distance_matrix):
        """This function returns the closest-pairs mean (serving as neighborhood parameter for DBSCAN)"""
        # Extract all pairs and their distances
        pairs = []
        n = distance_matrix.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((i, j, distance_matrix[i, j]))

        # Sort pairs by distance
        pairs.sort(key=lambda x: x[2])

        # Initialize clusters and a set to keep track of used points
        clusters = []
        used_points = set()

        # Form clusters by iterating over sorted pairs
        for i, j, distance in pairs:
            if i not in used_points and j not in used_points:
                clusters.append((i, j, distance))
                used_points.add(i)
                used_points.add(j)

        # get mean of closest pairs
        clusters_array = np.asarray(clusters)
        pairs_mean = np.mean(clusters_array[:,2])

        # print(f"Closest pairs mean: {pairs_mean}")

        return pairs_mean
    

    def dbscan(self, eps, distance_matrix):
        """This function applies dbscan and returns the labels and amount of noise"""
        db = DBSCAN(eps=eps, min_samples=5, metric='precomputed')
        labels = db.fit_predict(distance_matrix)

        # count noise points
        noise_amount = list(labels).count(-1)
        # print(" number of noise points: ", noise_amount)

        return labels, noise_amount


    def k_medoids(self, distance_matrix):
        # Initialize and fit KMedoids with a distance matrix
        # take as number of clusters 10% of the frames (at least 1 cluster)
        number_of_clusters = max(1, math.floor(0.1 * self.frames)) # maximum value: max(1, min(math.floor(0.1 * self.frames), 10))
        kmedoids = KMedoids(n_clusters=number_of_clusters, metric="precomputed", random_state=0)
        kmedoids.fit(distance_matrix)
        labels = kmedoids.labels_ # cluster labels
        medoid_indices = kmedoids.medoid_indices_ # medoid indices

        return labels
    

    def give_me_clusters(self, labels):
        """This functions groups the sample indices by their cluster assignment into 2 dictionarys"""
        sample_groups = {}

        for index, value in enumerate(labels): # The code iterates through each element in the ndarray using enumerate(arr), 
            # which provides both the index and the value of each element.

            if value not in sample_groups: # For each value, it checks if the value is already a key in the dictionary.
                sample_groups[value] = [] # If the value is not present, it initializes a new list for that value.
            sample_groups[value].append(index) # The index is then appended to the list corresponding to its value.

        # Attention: Noise points are at dictionary value == -1!

        # Calculate the number of samples in each category
        category_counts = {}
        for category, samples in sample_groups.items():
            category_counts[category] = len(samples)

        print("clusters: ", len(category_counts))
        self.total_amount_of_clusters += len(category_counts)

        # self.sample_groups = sample_groups
        # self.category_counts = category_counts

        return sample_groups, category_counts
    

    def second_round_dbscan(self, distance_matrix, labels, noise_amount):
        """This function checks if we have more than 40% noise points and runs dbscan again in that case"""
        if (noise_amount/self.frames) > 0.4:

            # get indices of noise points 
            noise_indices = np.where(labels == -1)
            noise_indices = np.vstack(noise_indices)[0].tolist()

            # determine closest pairs of noise points again
            sub_matrix = distance_matrix[np.ix_(noise_indices, noise_indices)]
            closest_mean_sub = self.closest_pairs_mean(sub_matrix)

            # apply DBSCAN again
            new_labels, new_noise = self.dbscan(eps=closest_mean_sub, distance_matrix=sub_matrix)

            # found a new cluster? Check how many
            sample_groups_1, category_counts_1 = self.give_me_clusters(new_labels)

            # now we have to extract original indices and form new clusters
            # ATTENTION: THIS SHOULD ONLY BE RUN ONCE!!! (or else run the previous cell again)
            for category in sample_groups_1.keys():

                if category == -1:
                    # get noise indices
                    noise1 = sample_groups_1[-1]
                    # print(f"noise: {noise1}")

                else:
                    # get indices of cluster
                    new_cluster = sample_groups_1[category]
                    new_cluster_length = len(new_cluster)
                    # print(f"new cluster: {new_cluster}")
                    # get original indices
                    original_cluster_indices = [self.sample_groups[-1][i] for i in new_cluster]
                    # append list of original indices to original dictionary (sample_groups)
                    key = len(self.sample_groups)-1
                    self.sample_groups[key] = original_cluster_indices
                    self.category_counts[key] = new_cluster_length

            # replace noise in original dictionary (but only if we really no NOT run DBSCAN again!)
            if -1 in sample_groups_1.keys(): # check if we have noise at all
                self.sample_groups[-1] = noise1
                self.category_counts[-1] = len(noise1)

    # rename category → cluster_id
    # rename count → cluster_size

    def sample_from_clusters(self, percentage = 0.01): # with old approach: n=100):
        """This function takes the fraction of how many samples need to be chosen and returns the subset"""
        # Random Sampling
        # first determine amount of selected samples depending on the size of each cluster
        # select 1 over n points (default = 5)
        # also select outliers

        # better: first determine amount of total frames and select percentage
        total_frames = sum(self.category_counts.values())  # Total number of frames
        print(f'the total amount of frames for this video: {total_frames}')
        target_samples_amount = int(total_frames * percentage)   # How many frames we need
        print(f'desired amount of frames for this video: {target_samples_amount}')

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # old approach (for 20% coresets)
        # select_sample_amount = {}
        # for category, count in self.category_counts.items():
        #     if category not in select_sample_amount:
        #         select_sample_amount[category] = []
        #     select_sample_amount[category].append(int(count * percentage)) # this is problematic because if cluster is small, this is zero

        # new approach (for 10% coresets and smaller)
        # Sort by value (biggest first)
        sorted_clusters_descending = dict(sorted(self.category_counts.items(), key=lambda x: x[1], reverse=True))
        sorted_clusters_ascending = dict(sorted(self.category_counts.items(), key=lambda x: x[1]))

        select_sample_amount = {}
        remaining_samples = target_samples_amount

        for category, count in sorted_clusters_descending.items():
            allocated = max(1, int((count / total_frames) * target_samples_amount))  # Ensure at least 1 sample
            select_sample_amount[category] = min(allocated, count)  # Avoid exceeding cluster size
            remaining_samples -= select_sample_amount[category]

        # Adjust if too many or too few samples were selected

        print(f'select sample amount: {select_sample_amount}')

        # add samples from largest clusters (to represent larger clusters more)
        while remaining_samples > 0:
            # print(f'remaining samples: {remaining_samples}')
            for category, count in sorted_clusters_descending.items():
                if remaining_samples == 0:
                    break
                if select_sample_amount[category] < count:  # Can we take more from this cluster?
                    select_sample_amount[category] += 1
                    remaining_samples -= 1

        # take away samples from smallest clusters (if too many samples were chosen)
        while remaining_samples < 0:
            if all(value <= 1 for value in select_sample_amount.values()): # wenn alle cluster nur ein sample geben sollen, wollen wir lieber von den kleineren entfernen
                for category, count in sorted_clusters_ascending.items(): # ascending instead of descending
                    if remaining_samples == 0:
                        break
                    if select_sample_amount[category] >= 1:  # Reduce from smallest clusters
                        select_sample_amount[category] -= 1
                        remaining_samples += 1
            else:
                for category, count in sorted_clusters_descending.items():
                    if remaining_samples == 0:
                        break
                    if select_sample_amount[category] >= 1:  # Reduce from largest clusters (only when they have more than one sample)
                        select_sample_amount[category] -= 1
                        remaining_samples += 1
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # log amount of selected frames per cluster
        print(f"All samples in clusters: {sorted_clusters_descending}")
        for key, value in select_sample_amount.items():
            print(f"For Cluster {key} I save {value} frames.")

        # randoly draw samples from each cluster (number determined in dictionary "select_sample_amount")
        selected_samples = {}
        for cluster, needed in select_sample_amount.items():
            selected_samples[cluster] = random.sample(self.sample_groups[cluster], needed) 

        self.subset = []
        for cluster, needed in selected_samples.items():
            self.subset += needed

        # print(f"The subset is this: {self.subset}")
        # self.subset represents the indices of the selected frames 

    # not needed anymore because this is done directly in save_images method
    # def select_frames(self, flat_images_matrix):
    #     """This function selects frames based on subset indices from original movie matrix. Only needed for greyscale images."""
    #     X_subset_direct = flat_images_matrix[:, self.subset]
    #     return X_subset_direct

    # for ResNet feature embedding (29.10.2024)
    def select_frames_from_img_list(self):
        """This function returns a sublist of images based on the subset indices"""
        image_subset_list = [self.image_name_list[i] for i in self.subset]
        return image_subset_list

    def save_images(self, destination_path, video_name, bgr_frames):
        """This function saves the subset images into the destination folder"""
        for i in range(len(self.subset)):
            count = "{:02d}".format(i)
            # greyscale
            # k = np.reshape(X_subset_direct[:,i], (self.height, self.width))
            # image = self.mat2gray(k)
            # bgr
            bgr_image = bgr_frames[self.subset[i]]
            # convert to rgb
            image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            matplotlib.image.imsave(destination_path + '/' + video_name + '_' + count + '.png', image) #, cmap='gray')
        print("all images have been saved!")

    # for ResNet feature embedding (29.10.2024)
    def save_images_1(self, image_subset_list):
        """This function is needed when method select_frames_from_img_list was used"""
        for img in image_subset_list:
            image_path = os.path.join(self.all_frames_path, img)
            # the command "shutil.copy(src, dst)" copies the file from the src path to the dst path
            shutil.copy(image_path, self.destination)
        print("all images have been saved!")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# metrics: "wasserstein", "euclid", "js", "spherical", "tvd"
# dimred: "POD", "PCA", "FE"

if __name__ == "__main__":

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    dataset = "EchoNet" # ImageNet or EchoNet or ENPed or LVH
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    if dataset == "ENPed":
        dimred = "FE" # POD, PCA or FE
        clustering = "DBSCAN2" # DBSCAN2, DBSCAN1, KMedoids
        metric = "wasserstein"
        # # #
        destination_path = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Pediatric/A4C/best_coreset_method/FE_DB2_wasserstein_reduced"
        # # #
        log_csv = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Pediatric/A4C/best_coreset_method/EP_logs.csv"
        set_name = "TRAIN"
        set_csv = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Pediatric/A4C/FileList_EF_categories.csv"
        set_column = 6
        source_path="/vol/ideadata/ep56inew/EchoNet/EchoNet-Pediatric/A4C/videos"
        all_frames_path = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Pediatric/A4C/train_all_frames"

    if dataset == "LVH":
        dimred = "FE"
        clustering = "DBSCAN2"
        metric = "wasserstein"
        # # #
        destination_path = "/vol/ideadata/ep56inew/EchoNet/EchoNet-LVH/frame_sets/jennys_little_coreset_Batch4"
        # # #
        log_csv = "/vol/ideadata/ep56inew/EchoNet/EchoNet-LVH/frame_sets/LVH_logs.csv"
        set_name = "train"
        set_csv = "/vol/ideadata/ep56inew/EchoNet/EchoNet-LVH/MeasurementsList_new_clean_ASH_official_split_more_test.csv"
        set_column = 5
        source_path = "/vol/ideadata/ep56inew/EchoNet/EchoNet-LVH/Batch4"
        all_frames_path = "/vol/ideadata/ep56inew/EchoNet/EchoNet-LVH/frame_sets/all_frames"

    if dataset == "EchoNet":
        dimred = "FE" # POD, PCA or FE
        metric="wasserstein"
        clustering = "DBSCAN2" # DBSCAN2, DBSCAN1, KMedoids
        log_csv = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/jennys_coresets_smaller/log_1_percent_again_wasserstein"
        # # #
        destination_path="/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/jennys_coresets_smaller/1_percent_again_wasserstein"
        source_path="/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/videos"
        set_csv = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/FileList.csv"
        set_name = "TRAIN"
        set_column = 8
        all_frames_path = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/official_split/training_sets/train_all_frames"

    elif dataset == "ImageNet":
        # *** TESTING ***
        # source_path="/vol/ideadata/ep56inew/ImageNet_VidVRD/all_videos/one_video_for_testing"
        # destination_path="/vol/ideadata/ep56inew/ImageNet_VidVRD/all_videos/one_video_for_testing/frames"
        # *** *** *** ***
        dimred = "PCA" # POD, PCA or FE
        metric1 = "js"
        metric2 = "tvd"
        clustering = "KMedoids" # DBSCAN2, DBSCAN1, KMedoids
        log_csv = "/vol/ideadata/ep56inew/ImageNet_VidVRD/subsets/training_coresets/IN_logs.csv"
        # # #
        destination_path1="/vol/ideadata/ep56inew/ImageNet_VidVRD/subsets/training_coresets/" + dimred + "_" + clustering + "/IN_" + metric1
        set_name = "train"
        set_csv = "/vol/ideadata/ep56inew/ImageNet_VidVRD/labels/official_split/one_class_labels_final_split_removed_classes_VAL_num.csv"
        set_column = 2
        source_path="/vol/ideadata/ep56inew/ImageNet_VidVRD/videos"
        all_frames_path = "/vol/ideadata/ep56inew/ImageNet_VidVRD/subsets/train_all_frames"

    # *******************************************************************************************************************
    # EXECUTE
    # only one source path
    count = 1
    JennysCoreset(source_path, destination_path, metric, set_csv, set_name, set_column, dataset, dimred, all_frames_path, clustering, log_csv, count)
    # JennysCoreset(source_path, destination_path, metric2, set_csv, set_name, set_column, dataset, dimred, all_frames_path, clustering, log_csv)
    
    # several source paths
    # count = 4
    # JennysCoreset(source_path, destination_path, metric, set_csv, set_name, set_column, dataset, dimred, all_frames_path, clustering, log_csv, count)
    # print(f"completed for {source_path}")
    
    print(f"Coreset constructed: {dataset}, {dimred}, {clustering}, {metric}")
    # count how many files were constructed
    files = sum(1 for entry in os.listdir(destination_path) if os.path.isfile(os.path.join(destination_path, entry)))
    print(f"amount of frames: {files}")
    
    # *******************************************************************************************************************

    # execute (EchoNet-LVH)
    # JennysCoreset(source_path="/vol/ideadata/ep56inew/EchoNet/EchoNet-LVH/Batch4", \
    #               destination_path="/vol/ideadata/ep56inew/EchoNet/EchoNet-LVH/train_spherical", metric="spherical")

    # execute (EchoNet-LVH)
    # alles der Reihe nach
    # source_paths = ["/vol/ideadata/ep56inew/EchoNet/EchoNet-LVH/Batch1"]
    
    # destination_path = "/vol/ideadata/ep56inew/EchoNet/EchoNet-LVH/frame_sets/train_tvd"
    # metric = "tvd"

    # set_csv = "/vol/ideadata/ep56inew/EchoNet/EchoNet-LVH/MeasurementsList_new_clean_ASH_sorted_set.csv"
    # set_name = "TRAIN"
    # set_column = 5

    # execute (EchoNet-Pediatric-A4C)
    # source_path="/vol/ideadata/ep56inew/EchoNet/EchoNet-Pediatric/A4C/Videos"
    # destination_path="/vol/ideadata/ep56inew/EchoNet/EchoNet-Pediatric/A4C/train_tvd"
    # metric="tvd"
    # set_csv="/vol/ideadata/ep56inew/EchoNet/EchoNet-Pediatric/A4C/FileList.csv"
    # set_name="TRAIN"
    # set_column=6
    # JennysCoreset(source_path, destination_path, metric, set_csv, set_name, set_column)