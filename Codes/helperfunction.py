import cv2
import os
import numpy as np
from scipy.spatial import distance
import copyreg
import matplotlib.pyplot as plt


def save_data(ndarray, path, mode):
    if mode == 'txt':
        f = open(path, "a")
        string_matrix = str(ndarray)
        f.write(string_matrix + "\n\n")
        f.close()

    elif mode == 'npy':
        np.save(path, ndarray)


def draw_kp_img(img, kp, index):
    output_path = os.path.join(os.getcwd(), "kp_imgs")
    img = cv2.drawKeypoints(img, kp, img)
    cv2.imwrite(os.path.join(output_path, "sift_keypoints{}.jpg".format(index)), img)


def read_images(path, mode='gray'):
    alist = sorted(os.listdir(path))
    image_sequence = []
    for img in alist:
        frame = cv2.imread(os.path.join(path, img))
        if mode == 'gray':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image_sequence.append(frame)

    return image_sequence


def extract_features(images, feature_type, nfeatures=3000):
    feature_sequence = []
    if feature_type == "SURF":
        operation = cv2.xfeatures2d.SURF_create(nfeatures)

    elif feature_type == "SIFT":
        operation = cv2.xfeatures2d.SIFT_create(nfeatures)

    elif feature_type == "ORB":
        operation = cv2.ORB_create(nfeatures=5000)

    for i in range(len(images)):
        kp, des = operation.detectAndCompute(images[i], None)

        feature_sequence.append((kp, des))
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
    # img1 = images[0]
    # img2 = images[1]
    # kp, des = operation.detectAndCompute(img1, None)
    # kp2, des2 = operation.detectAndCompute(img2, None)
    #
    # FLANN_INDEX_KDTREE = 1
    # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    # search_params = dict(checks=100)
    # flann = cv2.FlannBasedMatcher(index_params, search_params)
    # matches = flann.knnMatch(des, des2, k=2)
    # good = []
    # for m, n in matches:
    #     if m.distance < 0.8 * n.distance:
    #         good.append(m)
    #
    #
    # matched_image = cv2.drawMatches(img1, kp, img2, kp2, good, None, flags=2)
    # print(len(kp))
    # plt.axis('off')
    # plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
    # plt.show()
    # plt.imsave("test.png",matched_image)
    # exit()
    return feature_sequence


def Extract_labels(feature_sequence, ground_truth_frames):
    ground_truth_labels = {}
    index = 0
    background = 0
    foreground = 0
    for frame_index in feature_sequence:
        kps = frame_index[0]
        each_frame_labels = []
        for kp in kps:
            coords = np.flip(np.asarray(kp.pt).astype(int))
            if ground_truth_frames[index][coords[0]][coords[1]].all() == 0:
                each_frame_labels.append(0)
                background += 1
            else:
                each_frame_labels.append(1)
                foreground += 1
        ground_truth_labels[index] = each_frame_labels
        index += 1
    print(f"foreground sum = {foreground}  and background sum = {background}")
    return ground_truth_labels


def duplicate_data(matrices, ground_truth_sift_label):
    """
    Duplicate data for training, returns a randomized training data with evenly amount for foreground and background.
    :param matrices:
    :param ground_truth_sift_label:
    :return: An evenly distributed training data
    """
    foreground_length = sum(ground_truth_sift_label)
    total_length = len(ground_truth_sift_label)
    background_length = total_length - foreground_length

    if foreground_length < background_length:
        indices = np.where(ground_truth_sift_label == 1)[0]
        matrices_foreground = matrices[indices]
        matrices_foreground = np.resize(
            matrices_foreground,
            (len(ground_truth_sift_label) - 2 * foreground_length, 3, 5),
        )
        labels = np.ones(len(ground_truth_sift_label) - 2 * sum(ground_truth_sift_label))
        matrices = np.append(matrices, matrices_foreground, axis=0)
        ground_truth_sift_label = np.append(ground_truth_sift_label, labels)

        p = np.random.permutation(len(matrices))
        return matrices[p], ground_truth_sift_label[p]

    else:
        indices = np.where(ground_truth_sift_label == 0)[0]
        matrices_backgound = matrices[indices]
        matrices_backgound = np.resize(
            matrices_backgound,
            (len(ground_truth_sift_label) - 2 * background_length, 3, 5),
        )
        labels = np.zeros(len(ground_truth_sift_label) - 2 * background_length)
        matrices = np.append(matrices, matrices_backgound, axis=0)
        ground_truth_sift_label = np.append(ground_truth_sift_label, labels)

        p = np.random.permutation(len(matrices))
        return matrices[p], ground_truth_sift_label[p]


def match_keypoints(current_keypoints, current_descriptors, next_keypoints, next_descriptors, frame_num):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=100)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(current_descriptors, next_descriptors, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.45 * n.distance:
            good.append(m)

    if len(good) > 7:
        src_pts = np.float32([current_keypoints[m.queryIdx].pt for m in good]).reshape(
            -1, 1, 2
        )
        dst_pts = np.float32([next_keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # M, _ = cv2.findFundamentalMat(src_pts, dst_pts, cv2.RANSAC, 5.0, confidence=0.9)
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0, confidence=0.9)
        # norm = np.linalg.norm(current_descriptors[0])
        # print("the M is: ", M)

        if M is None:
            # print("none")
            return np.full((3, 5), np.inf)

        coordx, coordy = current_keypoints[0].pt
        extra = np.array([coordx, coordy, frame_num]).reshape(3, -1)
        correspond_cord = M @ np.c_[coordx, coordy, frame_num + 1].reshape(3, -1)
        correspond_cord = (correspond_cord / correspond_cord[2:])
        M = np.c_[M, extra, correspond_cord]
        # print("good")
        return M

    else:
        # print("No good")
        return np.zeros(1)


def find_nearest_keypoints(feature_distance, num_of_neighbour, current_frame_features,
                           next_frame_features, frame_num):
    if num_of_neighbour > len(feature_distance) - 1:
        idx = np.argpartition(feature_distance, len(feature_distance) - 1)
    else:
        idx = np.argpartition(feature_distance, num_of_neighbour)
    partial_kp = []
    # find num_of_neighbour good neighbour's indices
    for index in idx[:num_of_neighbour]:
        partial_kp.append(current_frame_features[0][index])
    partial_des = current_frame_features[1][idx[:num_of_neighbour]]
    M = match_keypoints(partial_kp, partial_des, next_frame_features[0], next_frame_features[1], frame_num)

    return M


def calculate_motion_matrices(current_frame_features, next_frame_features, img, next_img, frame_num):
    kp_coords = []

    for i in range(len(current_frame_features[0])):
        keypoint = current_frame_features[0][i]
        kp_coords.append(keypoint.pt)

    matrices_on_each_frame = []
    for i in range(len(current_frame_features[0])):
        keypoint = current_frame_features[0][i]
        # each kp compare all kps distance

        feature_distance = distance.cdist(
            np.asarray([keypoint.pt]), np.asarray(kp_coords)
        ).squeeze()

        num_of_neighbour = 8
        trans_matrix = find_nearest_keypoints(
            feature_distance,
            num_of_neighbour,
            current_frame_features,
            next_frame_features,
            frame_num
        )
        # count = 0
        # while True:
        #     if (trans_matrix == np.inf).all():
        #         # print("None: ", trans_matrix, "\n")
        #         break
        #     elif (trans_matrix == 0).all():
        #         # print(trans_matrix)
        #         if count > 10:
        #             # print("over threshold" , count)
        #             trans_matrix = np.full((3, 5), np.inf)
        #             break
        #         else:
        #             num_of_neighbour += 2
        #             trans_matrix = find_nearest_keypoints(
        #                 feature_distance,
        #                 num_of_neighbour,
        #                 current_frame_features,
        #                 next_frame_features,
        #             )
        #             count += 1
        #     else:
        #         RGB_coord = np.round(keypoint.pt)
        #         RGB = img[int(RGB_coord[1]), int(RGB_coord[0]), :].reshape(3, -1)
        #         trans_matrix = np.c_[trans_matrix, RGB]
        #         trans_matrix = RGB
        #         print("RGB: ", trans_matrix, "\n")
        #         break

        for count in range(40):
            if (trans_matrix == np.inf).all():
                # print("None: ", trans_matrix, "\n")
                break
            elif (trans_matrix == 0).all():
                # print(trans_matrix)
                if count == 39:
                    # print("over threshold" , count)
                    trans_matrix = np.full((3, 5), np.inf)
                else:
                    num_of_neighbour += 1
                    trans_matrix = find_nearest_keypoints(
                        feature_distance,
                        num_of_neighbour,
                        current_frame_features,
                        next_frame_features,
                        frame_num
                    )
            else:
                # Current_coord = trans_matrix[:, -2]
                Next_coord = trans_matrix[:, -1]
                #  print("Current in frame: ", Current_coord)
                #  print("Predict in next frame: ", Next_coord)
                if Next_coord[1] > next_img.shape[0] or Next_coord[0] > next_img.shape[1] or Next_coord[1] < 0 or \
                        Next_coord[0] < 0:
                    trans_matrix = np.full((3, 5), np.inf)
                else:
                    # Current_RGB = img[int(Current_coord[1]), int(Current_coord[0]), :].reshape(3, -1)
                    # Next_RGB = next_img[int(Next_coord[1]), int(Next_coord[0]), :].reshape(3, -1)
                    # trans_matrix = np.c_[trans_matrix, Current_RGB, Next_RGB]
                    break

        matrices_on_each_frame.append(trans_matrix)
    return np.asarray(matrices_on_each_frame)


def Retrieve_all_transformation_matrix(feature_sequence, color_imgs, output_path, MODE):
    print(output_path)
    if not os.path.isdir(os.path.join(output_path, "Homography_Matrices_npy")):
        os.mkdir(os.path.join(output_path, "Homography_Matrices_npy"))
    index = 0
    matrices_on_each_frame = {}
    feature_path = os.path.join(output_path, "feature_sequence.npy")
    feature_sequence = np.asarray(feature_sequence, dtype=object)
    while index + 1 < len(feature_sequence):
        current_frame_features = feature_sequence[index]
        next_frame_features = feature_sequence[index + 1]
        matrices_on_each_frame[index] = calculate_motion_matrices(current_frame_features, next_frame_features,
                                                                  color_imgs[index], color_imgs[index + 1], index)
        # delete corrupted keypoints
        if matrices_on_each_frame[index].__contains__(np.inf):
            # print("in Frame ", index)
            # print("Before deleting, we have: ", len(feature_sequence[index][1]))
            bad_keypoint_index = np.where(matrices_on_each_frame[index] == np.inf)
            matrices_on_each_frame[index] = matrices_on_each_frame[index][
                np.where(matrices_on_each_frame[index] != np.inf)].reshape(-1, 3, 5)
            feature_sequence[index][0] = np.delete(feature_sequence[index][0], np.unique(bad_keypoint_index[0]))

            # print("Deleting corrupted keypoints Current mode: ", MODE)
            # if mode is SIFT -> reshape(-1,128) | Surf -> (-1,64) | ORB -> (-1,32)
            if MODE == "SIFT":
                feature_sequence[index][1] = np.delete(feature_sequence[index][1], np.unique(bad_keypoint_index[0]),
                                                       axis=0).reshape(-1, 128)
            elif MODE == "SURF":
                feature_sequence[index][1] = np.delete(feature_sequence[index][1], np.unique(bad_keypoint_index[0]),
                                                       axis=0).reshape(-1, 64)
            else:
                feature_sequence[index][1] = np.delete(feature_sequence[index][1], np.unique(bad_keypoint_index[0]),
                                                       axis=0).reshape(-1, 32)
            # print("After deleting, the length is: ", len(feature_sequence[index][1]))
        # Write Matrix data into folder with TXT format
        # save_data(
        #     matrices_on_each_frame[index],
        #     os.path.join(
        #         output_path,
        #         "Homography_Matrices_txt",
        #         "Homography_on_frame{:03d}.txt".format(index),
        #     ), "txt"
        # )

        save_data(
            matrices_on_each_frame[index],
            os.path.join(
                output_path,
                "Homography_Matrices_npy",
                "Homography_on_frame{:03d}.npy".format(index),
            ), "npy"
        )

        index += 1

    save_data(feature_sequence, feature_path, "npy")


def load_data(input_path):
    if not os.path.exists(input_path): raise "Cannot find files"
    feature_sequence = np.load(
        os.path.join(input_path, "feature_sequence.npy"), allow_pickle=True
    )

    matrices_on_each_frame = {}
    for i in range(len(feature_sequence) - 1):
        matrices_on_each_frame[i] = np.load(
            os.path.join(
                input_path,
                "Homography_Matrices_npy",
                "Homography_on_frame{:03d}.npy".format(i),
            ),
            allow_pickle=True,
        )
    return feature_sequence, matrices_on_each_frame


def _pickle_keypoints(point):
    # allow pickle cv2.keypoints object
    return (
        cv2.KeyPoint,
        (
            *point.pt,
            point.size,
            point.angle,
            point.response,
            point.octave,
            point.class_id,
        ),
    )


copyreg.pickle(cv2.KeyPoint().__class__, _pickle_keypoints)
