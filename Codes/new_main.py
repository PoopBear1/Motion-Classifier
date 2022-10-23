import os
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from scipy.spatial import distance
from model import *
import copy

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))
data_path = os.path.join(ROOT, "Data")
image_path = os.path.join(data_path, "JPEGImages")
label_path = os.path.join(data_path, "Annotations")
output_path = os.path.join(ROOT, "Runs")
MODE = "NN"


def rearrange_data(ground_truth_labels, matrices_on_each_frame):
    """
    :param ground_truth_labels: labels array
    :param matrices_on_each_frame: data array
    :return: training matrices and label matrices || training dataloader and validation dataloader
    """
    # del ground_truth_labels[len(ground_truth_labels) - 1]
    num_frames = len(matrices_on_each_frame)

    temp_matrices = copy.deepcopy(matrices_on_each_frame)

    temp_labels = copy.deepcopy(ground_truth_labels)
    #################### duplicated data #####################

    for frame_id in range(len(temp_labels)):
        temp_matrices[frame_id], temp_labels[frame_id] = duplicate_data(
            temp_matrices[frame_id], temp_labels[frame_id])

    matrices_train = temp_matrices[0]
    labels_train = temp_labels[0]

    for i in range(1, 4):
        matrices_train = np.concatenate(
            (matrices_train, matrices_on_each_frame[i]), axis=0
        )
        labels_train = np.concatenate(
            (labels_train, ground_truth_labels[i]), axis=0
        )

    matrices_train = np.concatenate(
        (matrices_train, matrices_on_each_frame[num_frames - 1]), axis=0
    )
    labels_train = np.concatenate(
        (labels_train, ground_truth_labels[num_frames - 1]), axis=0
    )

    full_dataset = To_dataset(matrices_train, labels_train)
    train_size = int(0.8 * len(full_dataset))
    validate_size = len(full_dataset) - train_size
    train_dataset, validation_dataset = torch.utils.data.random_split(full_dataset, [train_size, validate_size])
    training_dataloader = To_dataloader(train_dataset, size=train_size, validate=False)
    validation_dataloader = To_dataloader(validation_dataset, size=validate_size, validate=True)
    print("Ratio of training_set / Validate_set :", train_size, '/', validate_size)
    return training_dataloader, validation_dataloader, validate_size


def get_model(matrices_on_each_frame, ground_truth_labels, path, model=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("model is training on {}".format(device))

    train_dataloader, validation_dataloader, validate_size = rearrange_data(ground_truth_labels,
                                                                            matrices_on_each_frame)

    model = train(train_dataloader, validation_dataloader, validate_size, path, model, ROOT)

    return model


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


def read_images(path, mode='gray'):
    alist = sorted(os.listdir(path))
    image_sequence = []
    for img in alist:
        frame = cv2.imread(os.path.join(path, img))
        if mode == 'gray':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image_sequence.append(frame)

    return image_sequence


def generate_compare_txt(image_path, f, filename_list):
    for index in range(len(filename_list) - 1):
        source = filename_list[index]
        compare = filename_list[index + 1]
        f.write(os.path.join(image_path, source) + " " + os.path.join(image_path, compare) + "\n")


def save_data(ndarray, path, mode):
    if mode == 'txt':
        f = open(path, "a")
        string_matrix = str(ndarray)
        f.write(string_matrix + "\n\n")
        f.close()

    elif mode == 'npy':
        np.save(path, ndarray)


def load_data(input_path):
    refine_kp_path = os.path.join(input_path, "Refine_matches")
    matrix_type = "Fundamental"
    matrices_path = os.path.join(input_path, "{}_Matrices_npy".format(matrix_type))

    if not os.path.exists(input_path) or not os.path.exists(refine_kp_path):
        raise "Cannot find files"

    match_kps_list = sorted([os.path.join(refine_kp_path, i) for i in os.listdir(refine_kp_path)])
    index = 0

    matrices_on_each_frame = {}
    feature_sequence = []

    while index < len(match_kps_list):
        current_kp = np.load(match_kps_list[index])
        feature_sequence.append(current_kp)

        matrices_on_each_frame[index] = np.load(
            os.path.join(
                matrices_path,
                "{}_on_frame{:03d}.npy".format(matrix_type, index),
            ),
        )
        index += 1
    return feature_sequence, matrices_on_each_frame

    ################################### Raw Keypoints version
    # matrix_type = "Fundamental"
    # matrices_path = os.path.join(input_path, "{}_Matrices_npy".format(matrix_type))
    # kp_path = os.path.join(input_path, "KP_Matches")
    #
    # if not os.path.exists(input_path):
    #     raise "Cannot find files"
    #
    # match_kps_list = sorted([os.path.join(kp_path, i) for i in os.listdir(kp_path)])
    #
    # index = 0
    #
    # matrices_on_each_frame = {}
    # feature_sequence = []
    # while index < len(match_kps_list):
    #     current_kp = np.load(match_kps_list[index])["keypoints0"]
    #     feature_sequence.append(current_kp)
    #     matrices_on_each_frame[index] = np.load(
    #         os.path.join(
    #             matrices_path,
    #             "{}_on_frame{:03d}.npy".format(matrix_type, index),
    #         ),
    #     )
    #     index += 1
    # return feature_sequence, matrices_on_each_frame


def match_keypoints(current_partial_kp, next_partial_kp, frame_num):
    # mode = "F"
    mode = "H"
    if mode == "F":
        if len(current_partial_kp) > 6:
            current_partial_kp = np.float32(current_partial_kp).reshape(-1, 1, 2)
            next_partial_kp = np.float32(next_partial_kp).reshape(-1, 1, 2)

            F, _ = cv2.findFundamentalMat(current_partial_kp, next_partial_kp, cv2.RANSAC, 5.0, confidence=0.9)

            if F is None:
                return np.zeros(1)
            #################################################### Refine the F Matrix
            # for i in range(current_partial_kp.shape[0]):
            #     current_kp_cord = current_partial_kp[i]
            #     next_kp_cord = next_partial_kp[i]
            #     current_kp_cord = np.c_[current_kp_cord, 1]
            #     next_kp_cord = np.c_[next_kp_cord, 1]
            #     if np.abs(next_kp_cord @ F @ current_kp_cord.T - 0) < 1e-4:
            #         continue
            #     else:
            #         # print("not a valid F matrix")
            #         return np.zeros(1)
            # extra = np.array([current_partial_kp[0], current_partial_kp[1], frame_num]).reshape(3, -1)
            # F = np.c_[F, extra]
            return F

        else:
            print("No good")
            return np.zeros(1)
    else:

        if len(current_partial_kp) > 7:
            current_partial_kp = np.float32(current_partial_kp).reshape(-1, 1, 2)
            next_partial_kp = np.float32(next_partial_kp).reshape(-1, 1, 2)

            H, _ = cv2.findHomography(current_partial_kp, next_partial_kp, cv2.RANSAC, 5.0, confidence=0.95)

            if H is None:
                return np.zeros(1)
            current_cord = np.c_[current_partial_kp[0], frame_num].reshape(3, -1)
            next_cord = np.c_[next_partial_kp[0], frame_num + 1].reshape(3, -1)
            H = np.c_[H, current_cord, next_cord]
            # print(current_cord.shape,next_cord.shape,H.shape)

            #################################################### Refine the H Matrix
            # for i in range(current_partial_kp.shape[0]):
            #     current_kp_cord = current_partial_kp[i]
            #     next_kp_cord = next_partial_kp[i]
            #     current_kp_cord = np.c_[current_kp_cord, 1]
            #     next_kp_cord = np.c_[next_kp_cord, 1]
            #
            #     transfer_cord = (H @ current_kp_cord.T).T
            #     if transfer_cord[:, 2] == 0:
            #         return np.zeros(1)
            #
            #     transfer_cord = transfer_cord / transfer_cord[:, 2]

            # if np.linalg.norm(transfer_cord - next_kp_cord) <= 1:
            #     continue
            # else:
            #     print("not a valid F matrix")
            #     return np.zeros(1)
            # print(transfer_cord.shape)
            # if transfer_cord[:, 0] < 0 or transfer_cord[:, 0] > 1920 or transfer_cord[:, 1] < 0 or transfer_cord[:,
            #                                                                                        1] > 1080:
            #     return np.zeros(1)

            return H


def Extract_labels(feature_sequence, ground_truth_frames):
    ground_truth_labels = {}
    index = 0
    background = 0
    foreground = 0
    for kps in feature_sequence:
        each_frame_labels = []
        for kp in kps:
            coords = np.flip(np.asarray(kp).astype(int))
            if ground_truth_frames[index][coords[0]][coords[1]].all() == 0:
                each_frame_labels.append(0)
                background += 1
            else:
                each_frame_labels.append(1)
                foreground += 1
        ground_truth_labels[index] = np.asarray(each_frame_labels)
        index += 1
        if index == 1:
            print(f"The first item: foreground sum = {foreground} and background sum = {background}")

    print(f"foreground sum = {foreground}  and background sum = {background}")
    return ground_truth_labels


def find_nearest_keypoints(feature_distance,
                           num_of_neighbour,
                           current_frame_features,
                           next_frame_features,
                           frame_num):
    if num_of_neighbour > len(feature_distance) - 1:
        idx = np.argpartition(feature_distance, len(feature_distance) - 1)
    else:
        idx = np.argpartition(feature_distance, num_of_neighbour)
    current_partial_kp = []
    next_partial_kp = []
    for index in idx[:num_of_neighbour]:
        current_partial_kp.append(current_frame_features[index])
        next_partial_kp.append(next_frame_features[index])

    M = match_keypoints(current_partial_kp, next_partial_kp, frame_num)

    return M


def calculate_motion_matrices(current_compare, frame_num):
    current_kp = current_compare["keypoints0"]
    next_kp = current_compare["keypoints1"]
    # matches = current_compare["matches"]

    matrices_on_each_frame = []
    found_matrix = 0
    for i in range(current_kp.shape[0]):
        keypoint = current_kp[i].reshape(-1, 2)
        # each kp compare all kps distance
        feature_distance = distance.cdist(keypoint, current_kp).squeeze()

        num_of_neighbour = 8
        trans_matrix = find_nearest_keypoints(
            feature_distance,
            num_of_neighbour,
            current_kp,
            next_kp,
            frame_num
        )
        for count in range(20):
            if (trans_matrix == np.inf).all():
                # print("None: ", trans_matrix, "\n")
                break
            elif (trans_matrix == 0).all():
                # print(trans_matrix)
                if count == 19:
                    # print("over threshold", count)
                    trans_matrix = np.full((3, 5), np.inf)
                else:
                    num_of_neighbour += 1
                    trans_matrix = find_nearest_keypoints(
                        feature_distance,
                        num_of_neighbour,
                        current_kp,
                        next_kp,
                        frame_num
                    )
            else:
                found_matrix += 1
                break
                # # Current_coord = trans_matrix[:, -2]
                # Next_coord = trans_matrix[:, -1]
                # #  print("Current in frame: ", Current_coord)
                # #  print("Predict in next frame: ", Next_coord)
                # if Next_coord[1] > next_img.shape[0] or Next_coord[0] > next_img.shape[1] or Next_coord[1] < 0 or \
                #         Next_coord[0] < 0:
                #     trans_matrix = np.full((3, 5), np.inf)
                # else:
                #     # Current_RGB = img[int(Current_coord[1]), int(Current_coord[0]), :].reshape(3, -1)
                #     # Next_RGB = next_img[int(Next_coord[1]), int(Next_coord[0]), :].reshape(3, -1)
                #     # trans_matrix = np.c_[trans_matrix, Current_RGB, Next_RGB]
                #     break
        # print(trans_matrix)
        matrices_on_each_frame.append(trans_matrix)
    matrices_on_each_frame = np.asarray(matrices_on_each_frame)

    return matrices_on_each_frame


def Retrieve_all_transformation_matrix(match_kps_list):
    # matrix_type = "Homography"
    matrix_type = "Fundamental"
    refine_kps_path = os.path.join(output_path, "Refine_matches")
    if not os.path.isdir(os.path.join(output_path, "{}_Matrices_npy").format(matrix_type)):
        os.mkdir(os.path.join(output_path, "{}_Matrices_npy".format(matrix_type)))

    if not os.path.isdir(refine_kps_path):
        os.mkdir(refine_kps_path)

    index = 0
    matrices_on_each_frame = {}
    while index < len(match_kps_list):
        current_compare = np.load(match_kps_list[index])
        matrices_on_each_frame[index] = calculate_motion_matrices(current_compare, index)

        current_kps = current_compare["keypoints0"]
        print("before tuning, we have {} key points".format(current_kps.shape))
        if matrices_on_each_frame[index].__contains__(np.inf):
            matrices_on_each_frame[index] = matrices_on_each_frame[index][
                np.where(matrices_on_each_frame[index] != np.inf)].reshape(-1, 3, 5)
            current_kps = current_kps[np.where(matrices_on_each_frame[index] != np.inf)[0]]
            current_kps = current_kps[np.unique(np.where(matrices_on_each_frame[index] != np.inf)[0])]

        print("After cut cropped key points, we have {} key points".format(current_kps.shape))
        np.save(os.path.join(refine_kps_path, "matched_kp_on_frame{:03d}".format(index)), current_kps)
        print("Saving computed matrix")
        save_data(
            matrices_on_each_frame[index],
            os.path.join(
                output_path,
                "{}_Matrices_npy".format(matrix_type),
                "{}_on_frame{:03d}.npy".format(matrix_type, index),
            ), "npy"
        )
        index += 1


def extract_feature(input_txt_path, output_path):
    # This part of code is directly copied from dfm.py
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    import Codes.dfm.dfm as dfm
    # For all pairs in input_pairs perform DFM
    print("Matching Keypoint....")
    model = dfm.fm
    with open(input_txt_path) as f:
        for line in f:
            pairs = line.split(' ')
            pairs[1] = pairs[1].split('\n')[0]

            img_A = np.array(Image.open('./' + pairs[0]))
            img_B = np.array(Image.open('./' + pairs[1].split('\n')[0]))

            H, H_init, points_A, points_B = model.match(img_A, img_B)

            keypoints0 = points_A.T
            keypoints1 = points_B.T

            if pairs[0].count('/') > 0:

                p1 = pairs[0].split('/')[pairs[0].count('/')].split('.')[0]
                p2 = pairs[1].split('/')[pairs[0].count('/')].split('.')[0]

            elif pairs[0].count('/') == 0:
                p1 = pairs[0].split('.')[0]
                p2 = pairs[1].split('.')[0]

            np.savez_compressed(output_path + '/' + p1 + '_' + p2 + '_' + 'matches',
                                keypoints0=keypoints0, keypoints1=keypoints1)


def NN_predict(matrices, model):
    model.eval()
    predicted_labels = []
    for index in range(len(matrices)):
        matrix = matrices[index]
        matrix = matrix.reshape(1, 3, -1)
        matrix = torch.tensor(matrix, dtype=torch.float32)
        matrix = matrix.unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(matrix)
        thresh = torch.tensor([0.5]).to(device)
        predicted = (outputs > thresh).float() * 1
        predicted_labels.append(predicted.item())
    return predicted_labels


def NN_clustering(matrices_on_each_frame, feature_sequence, rgb_images, model):
    # summary(MatClassificationNet().to(device), input_size=(1, 3, 5))
    for i in range(len(matrices_on_each_frame)):
        matrices = matrices_on_each_frame[i]
        labels = NN_predict(matrices, model)
        image = rgb_images[i]
        current_kp = feature_sequence[i]
        Show_predictions(current_kp, labels, i, image, video)


def Show_predictions(current_kp, labels, frame_num, image, video):
    output_path = os.path.join(ROOT, "Runs", video, "Outputs")
    background_path = os.path.join(output_path, "Background")
    foreground_path = os.path.join(output_path, "Foreground")
    mask_path = os.path.join(output_path, "Mask")
    tracking_path = os.path.join(output_path, "Track")
    cut_path = os.path.join(output_path, "Cut")
    if not os.path.isdir(background_path):
        os.makedirs(background_path)
    if not os.path.isdir(foreground_path):
        os.makedirs(foreground_path)
    if not os.path.isdir(mask_path):
        os.makedirs(mask_path)
    if not os.path.isdir(tracking_path):
        os.makedirs(tracking_path)
    if not os.path.isdir(cut_path):
        os.makedirs(cut_path)

    foreground_points = []
    background_points = []
    kp_coords = []
    for i in range(len(labels)):
        if labels[i] == 0:
            background_points.append(current_kp[i])
        else:
            foreground_points.append(current_kp[i])
            kp_coords.append(current_kp[i])

    background_img = image.copy()
    for bg_points in background_points:
        cv2.circle(background_img, (int(bg_points[0]), int(bg_points[1])), radius=1, color=(0, 0, 255))
    cv2.imwrite(os.path.join(background_path, "Background{:03d}.jpg".format(frame_num)), background_img)
    for fg_points in foreground_points:
        cv2.circle(image, (int(fg_points[0]), int(fg_points[1])), radius=1, color=(255, 0, 0))
    cv2.imwrite(os.path.join(foreground_path, "Foreground{:03d}.jpg".format(frame_num)), image)


if __name__ == "__main__":
    videos = os.listdir(image_path)
    for video in videos:
        if video != ".DS_Store":
            output_path = os.path.join(output_path, video)
            if not os.path.isdir(output_path):
                os.makedirs(output_path)
            all_images = os.path.join(image_path, video)
            all_gt = os.path.join(label_path, video)
            gt_images = read_images(all_gt)
            rgb_images = read_images(all_images, "color")
        # first time generating data:
        # f = open(os.path.join(output_path, "image_pairs.txt"), "w")
        # generate_compare_txt(all_images, f, sorted(os.listdir(all_images)))
        # f.close()
        # txt_path = os.path.join(output_path, "image_pairs.txt")
        #
        matches_path = os.path.join(output_path, "KP_Matches")
        # extract_feature(txt_path, output_path=matches_path)
        match_kps_list = sorted([os.path.join(matches_path, file) for file in os.listdir(matches_path)])
        # Retrieve_all_transformation_matrix(match_kps_list)

        ## train model:
        # when we have generated data:
        feature_sequence, matrices_on_each_frame = load_data(output_path)
        ground_truth_labels = Extract_labels(feature_sequence, gt_images)

        model_path = os.path.join(ROOT, "Runs", video, "ckpt.pth")
        if os.path.exists(model_path):
            chkpt = torch.load(model_path, map_location=device)
            model = MatClassificationNet().to(device)
            model.load_state_dict(chkpt)
        else:
            model = get_model(matrices_on_each_frame, ground_truth_labels, video)

        ## Validate model
        NN_clustering(matrices_on_each_frame, feature_sequence, rgb_images, model)
