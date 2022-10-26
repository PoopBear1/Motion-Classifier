import cv2
from skimage.segmentation import slic, mark_boundaries
from helperfunction import *
import time
from sklearn import mixture, svm
from sklearn.decomposition import PCA
from pathlib import Path
from model import *
import copy
import torch
from sklearn import svm
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))
data_path = os.path.join(ROOT, "Data")
image_path = os.path.join(data_path, "JPEGImages")
label_path = os.path.join(data_path, "Annotations")
output_path = os.path.join(ROOT, "Runs")
MODE = "SURF"


# TOTAL Data version:
# def rearrange_data(ground_truth_labels, matrices_on_each_frame, mode="tradition"):
#     """
#     :param ground_truth_labels: labels array
#     :param matrices_on_each_frame: data array
#     :param mode: tradition means(GMM/SVM) ortherwise NN mode
#     :return: training matrices and lable matrices || training dataloader and validation dataloader
#     """
#     del ground_truth_labels[len(ground_truth_labels) - 1]
#     num_frames = len(matrices_on_each_frame)
#
#     temp_matrices = copy.deepcopy(matrices_on_each_frame)
#     temp_labels = copy.deepcopy(ground_truth_labels)
#
#     #################### raw data #####################
#     # matrices_train = temp_matrices[0]
#     # labels_train = temp_labels[0]
#     # for i in range(1, num_frames):
#     #     matrices_train = np.concatenate(
#     #         (matrices_train, matrices_on_each_frame[i]), axis=0
#     #     )
#     #     labels_train = np.concatenate(
#     #         (labels_train, ground_truth_labels[i]), axis=0
#     #     )
#     #####################################################
#
#     #################### duplicated data #####################
#     for frame_id in range(len(temp_labels)):
#         temp_matrices[frame_id], temp_labels[frame_id] = duplicate_data(
#             temp_matrices[frame_id], temp_labels[frame_id])
#
#     matrices_train = temp_matrices[0]
#     labels_train = temp_labels[0]
#
#     for i in range(num_frames):
#         matrices_train = np.concatenate(
#             (matrices_train, matrices_on_each_frame[i]), axis=0
#         )
#         labels_train = np.concatenate(
#             (labels_train, ground_truth_labels[i]), axis=0
#         )
#     #####################################################
#
#     if mode == "tradition":
#         training_size = int(0.2 * len(matrices_train))
#
#         return matrices_train[:training_size], labels_train[:training_size]
#
#     else:
#         full_dataset = To_dataset(matrices_train, labels_train)
#
#         train_size = int(0.9 * len(full_dataset))
#         print("Ratio of training_data / full_data :", train_size, '/', len(full_dataset))
#         validate_size = len(full_dataset) - train_size
#         train_dataset, validation_dataset = torch.utils.data.random_split(full_dataset, [train_size, validate_size])
#         training_dataloader = To_dataloader(train_dataset)
#         validation_dataloader = To_dataloader(validation_dataset)
#         return training_dataloader, validation_dataloader, validate_size

# Frames input version:
def rearrange_data(ground_truth_labels, matrices_on_each_frame, mode="tradition"):
    """
    :param ground_truth_labels: labels array
    :param matrices_on_each_frame: data array
    :param mode: tradition means(GMM/SVM) ortherwise NN mode
    :return: training matrices and lable matrices || training dataloader and validation dataloader
    """
    del ground_truth_labels[len(ground_truth_labels) - 1]
    num_frames = len(matrices_on_each_frame)

    temp_matrices = copy.deepcopy(matrices_on_each_frame)
    temp_labels = copy.deepcopy(ground_truth_labels)

    #################### raw data #####################
    # matrices_train = temp_matrices[0]
    # labels_train = temp_labels[0]
    # for i in range(1, num_frames):
    #     matrices_train = np.concatenate(
    #         (matrices_train, matrices_on_each_frame[i]), axis=0
    #     )
    #     labels_train = np.concatenate(
    #         (labels_train, ground_truth_labels[i]), axis=0
    #     )
    #####################################################

    #################### duplicated data #####################

    for frame_id in range(len(temp_labels)):
        temp_matrices[frame_id], temp_labels[frame_id] = duplicate_data(
            temp_matrices[frame_id], temp_labels[frame_id])

    matrices_train = temp_matrices[0]
    labels_train = temp_labels[0]

    # matrices_val = temp_matrices[1]
    # labels_val = temp_labels[1]
    # split_size = int(0.05 * num_frames)
    for i in range(1, 20):
        # set frame into Train
        # if i % split_size == 0:
        #     # print(matrices_on_each_frame[i].shape)
        #     matrices_train = np.concatenate(
        #         (matrices_train, matrices_on_each_frame[i]), axis=0
        #     )
        #     labels_train = np.concatenate(
        #         (labels_train, ground_truth_labels[i]), axis=0
        #     )
        # try next frame as val set
        # if i + 1 < num_frames:
        #     matrices_val = np.concatenate(
        #         (matrices_val, matrices_on_each_frame[i + 1]), axis=0
        #     )
        #     labels_val = np.concatenate(
        #         (labels_val, ground_truth_labels[i]), axis=0
        #     )
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
    #####################################################

    if mode == "tradition":
        training_size = int(0.2 * len(matrices_train))
        return matrices_train[:training_size], labels_train[:training_size]

    else:
        # if try next frame as val set
        # print("Ratio of training_frame / frame :", split_size, '/', num_frames)
        # full_dataset = To_dataset(matrices_train, labels_train)
        # validate_size = matrices_val.shape[0]
        # training_dataloader = To_dataloader(full_dataset)
        # validation_dataset = To_dataset(matrices_val, labels_val)
        # validation_dataloader = To_dataloader(validation_dataset)
        # print("Ratio of training_set / Validate_set :", matrices_train.shape[0], '/', validate_size)
        ############################

        # print("Ratio of training_frame / frame :", split_size, '/', num_frames)
        full_dataset = To_dataset(matrices_train, labels_train)
        train_size = int(0.8 * len(full_dataset))
        validate_size = len(full_dataset) - train_size
        train_dataset, validation_dataset = torch.utils.data.random_split(full_dataset, [train_size, validate_size])
        training_dataloader = To_dataloader(train_dataset, size=train_size)
        validation_dataloader = To_dataloader(validation_dataset, validate_size, True)
        print("Ratio of training_set / Validate_set :", train_size, '/', validate_size)
        return training_dataloader, validation_dataloader, validate_size


def NN_predict(matrices, feature_points, model):
    labels = []
    # probs = []
    # confidence = {}
    # fg_threshold = []
    # bg_threshold = []
    # 这个地方传入 feature sequence然后先检查长度与matrices是否一致 如果一致的话就是可以进行操作： 一致。

    # 第几个matrix就是第几个feature points,用这个index的坐标当作key，预测概率之差当作value 组成一个dict
    # 对这个dict 进行排序，然后取整体长度的前95%高的置信度范围，剩下的舍弃不要
    # 根据index 重组labels
    count = 0
    for index in range(len(matrices)):
        matrix = matrices[index]
        matrix = matrix.reshape(1, 3, -1)
        matrix = torch.tensor(matrix, dtype=torch.float32)
        matrix = matrix.unsqueeze(0).to(device)
        outputs = model(matrix)
        prob, predicted_label = outputs.max(1)
        prob = float(prob.item())
        predicted_label = predicted_label.item()
        # probs.append(prob)
        # confidence[count] = [prob, predicted_label]
        # print(confidence_sort[count])
        labels.append(predicted_label)
    # bg_threshold = sorted(bg_threshold, reverse=True)
    # bg_threshold = bg_threshold[int(len(bg_threshold)*0.8)]
    # fg_threshold = sorted(fg_threshold, reverse=True)
    # fg_threshold = fg_threshold[int(len(fg_threshold)*0.8)]

    # marks = []
    # for index in range(len(matrices)):
    #     matrix = matrices[index]
    #     matrix = matrix.reshape(1, 3, -1)
    #     matrix = torch.tensor(matrix, dtype=torch.float32)
    #     matrix = matrix.unsqueeze(0).to(device)
    #     outputs = model(matrix)
    #     prob, predicted_label = outputs.max(1)
    #     if predicted_label == 0 and prob <= bg_threshold:
    #         labels.append(predicted_label)
    #
    #     elif predicted_label == 1 and prob <= fg_threshold:
    #         labels.append(predicted_label)
    #
    #     else:
    #         marks.append(index)

    # probs.append(prob)
    # confidence[count] = [prob, predicted_label]
    # print(confidence_sort[count])
    # labels.append(predicted_label)

    # print(len(marks))
    # sorted_dict = sorted(confidence.items(), key=lambda x: x[1][0], reverse=True)
    # sorted_dict = dict(sorted_dict[:int(len(sorted_dict) * 0.9)])
    # index_to_delete = list(confidence.keys() - sorted_dict.keys())
    # index_to_delete = sorted(index_to_delete, reverse=True)
    #
    # # print("low confidence points index : ", index_to_delete)
    #
    # if type(feature_points) is not list:
    #     feature_points = feature_points.tolist()
    #
    # for i in sorted(marks, reverse=True):
    #     del feature_points[i]

    # for index in index_to_delete:
    #     del confidence[index]
    #     del feature_points[index]
    #
    # for i in range(max(confidence)+1):
    #     if i in confidence.keys():
    #         # print(i, confidence[i])
    #         labels.append(confidence[i][1])

    # print("look at here,", len(labels), len(feature_points))
    return labels


def filter_outliers(fg_kp, kp_coords):
    if len(kp_coords) < 2: return None
    n_sample = 20
    if len(kp_coords) < n_sample:
        n_sample = len(kp_coords)

    clf = LocalOutlierFactor(n_neighbors=n_sample, metric="euclidean")
    labels = clf.fit_predict(kp_coords)

    i = 0
    while i < len(labels):
        if labels[i] == -1:
            fg_kp = np.concatenate((fg_kp[:i], fg_kp[i + 1:]), axis=0)
            kp_coords = np.concatenate((kp_coords[:i], kp_coords[i + 1:]), axis=0)
            labels = np.concatenate((labels[:i], labels[i + 1:]), axis=0)
        else:
            i += 1
    min_coords = np.rint(np.amin(kp_coords, axis=0)).astype(int)
    max_coords = np.rint(np.amax(kp_coords, axis=0)).astype(int)

    return fg_kp.tolist(), min_coords, max_coords
    # return fg_kp.tolist()


def NN_clustering(feature_sequence, matrices_on_each_frame, gray_imgs, model, video):
    # summary(MatClassificationNet().to(device), input_size=(1, 3, 6))

    model.eval()
    for i in range(len(matrices_on_each_frame)):
        print("in frame ", i)
        matrices = matrices_on_each_frame[i]
        labels = NN_predict(matrices, feature_sequence[i][0], model)
        Show_predictions(feature_sequence, labels, i, gray_imgs, video)


def train_GMM(ground_truth_labels, matrices_on_each_frame):
    matrices_train, labels_train = rearrange_data(ground_truth_labels, matrices_on_each_frame)
    print("total matrices to be trained: ", len(matrices_train))
    all_matrices = np.asarray(matrices_train)
    nsamples, nx, ny = matrices_train.shape
    X_train = all_matrices.reshape((nsamples, nx * ny))
    print("matrices convert to 2D: ", X_train.shape)
    # Dimensionality reduction
    # pca = PCA(n_components=0.99)
    # pca.fit(X_train)
    # new_data = pca.transform(X_train)
    # print("after Pca, its dimension", new_data.shape)

    clf = mixture.GaussianMixture(n_components=2)
    clf.fit(X_train)
    return clf


def GMM_clustering(model, feature_sequence, matrices_on_each_frame, gray_imgs):
    for i in matrices_on_each_frame:
        matrices = matrices_on_each_frame[i]
        labels = []
        for matrix in matrices:
            matrix = matrix.reshape(1, -1)
            label = model.predict(matrix)
            labels.append(label)
        Show_predictions(feature_sequence, labels, i, gray_imgs)


def Get_mask(foreground_kp, image, K):
    """
    Returns the binary mask using SLIC
    :param image:
    :param foreground_kp: which is the key points of the foreground
    :return: a ndarray of binary mask
    """

    threshold = 3
    labels = slic(
        image,
        K / 2,
        compactness=10.0,
        max_iter=20,
        sigma=1,
        enforce_connectivity=True,
        start_label=1,
    )
    # plt.figure()
    # plt.imshow(mark_boundaries(image,labels))
    # plt.show()
    # plt.close()
    region_list = np.zeros(np.max(labels))
    try:
        for keypoint in foreground_kp:
            coords = np.flip(np.asarray(keypoint.pt).astype(int))
            region_index = labels[coords[0]][coords[1]]
            region_list[region_index] += 1

    except:
        pass

    mask = np.zeros(labels.shape)
    for index in range(len(region_list)):
        if region_list[index] > threshold:
            mask[np.where(labels == index)] = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5 + threshold, 5 + threshold))
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.normalize(closing, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    # contours, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    return mask


def Show_predictions(feature_sequence, labels, frame_id, video_frames, video):
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

    kp0 = []
    kp1 = []
    kp_coords = []
    for i in range(len(labels)):
        if labels[i] == 0:
            kp0.append(feature_sequence[frame_id][0][i])
        else:
            kp1.append(feature_sequence[frame_id][0][i])
            kp_coords.append(feature_sequence[frame_id][0][i].pt)

    print("Before filtering: ", len(kp1))
    if filter_outliers(np.asarray(kp1), np.asarray(kp_coords)) is not None:
        kp1, min_coords, max_coords = filter_outliers(np.asarray(kp1), np.asarray(kp_coords))
        print("Aftering filtering: ", len(kp1))
        print("$ the background : ", len(kp0))
        print("#############")
        tempImg = copy.deepcopy(video_frames[frame_id])
        background = cv2.drawKeypoints(tempImg, kp0, None, color=[0, 0, 255])
        # Draw /background points
        cv2.imwrite(os.path.join(background_path, "Background{}.jpg".format(frame_id)), background)
        foreground = cv2.drawKeypoints(tempImg, kp1, None, color=[255, 0, 0])
        # Draw foreground points
        cv2.imwrite(os.path.join(foreground_path, "Foreground{}.jpg".format(frame_id)), foreground)
        thresh = len(kp0) + len(kp1)

        # Draw Tracking rectangle
        image_track = cv2.rectangle(tempImg, tuple(min_coords), tuple(max_coords), (255, 0, 0), 2)
        cv2.imwrite(os.path.join(tracking_path, "Track{:03d}.png".format(frame_id)), image_track)

        # Draw Grab-Cut Graph

        mask = np.zeros(video_frames[frame_id].shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        rect = (min_coords[0], min_coords[1], max_coords[0] - min_coords[0], max_coords[1] - min_coords[1])
        if max_coords[0] - min_coords[0] != 0 or max_coords[1] - min_coords[1] != 0:
            # thresh = rect(max_coords[0]-min_coords[0] * max_coords[1]-min_coords[1])
            cv2.grabCut(video_frames[frame_id], mask, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")

            tempImg = video_frames[frame_id] * mask2[:, :, np.newaxis]
            cv2.imwrite(os.path.join(cut_path, "Cut{:03d}.jpg".format(frame_id)), tempImg)

        # Draw mask
        mask = Get_mask(kp1, tempImg, 100)
        cv2.imwrite(os.path.join(mask_path, "Mask{:03d}.jpg".format(frame_id)), mask)


def train_model(matrices_on_each_frame, ground_truth_labels, path, model=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("model is training on {}".format(device))
    # train_dataloader, validation_dataloader, validate_size = rearrange_data(ground_truth_labels,
    #                                                                         matrices_on_each_frame, "NN")

    train_dataloader, validation_dataloader, validate_size = rearrange_data(ground_truth_labels,
                                                                            matrices_on_each_frame, "NN")
    model = train(train_dataloader, validation_dataloader, validate_size, path, model, ROOT)
    return model


def train_SVM(matrices_on_each_frame, ground_truth_labels):
    predictor = svm.SVC(gamma='scale', C=1.0, decision_function_shape='ovr', kernel='rbf')
    matrices_train, labels_train = rearrange_data(ground_truth_labels, matrices_on_each_frame)
    all_matrices = np.asarray(matrices_train)
    samples, nx, ny = all_matrices.shape
    X_train = all_matrices.reshape((samples, nx * ny))

    predictor.fit(X_train, labels_train)
    for i in matrices_on_each_frame:
        matrices = matrices_on_each_frame[i]
        labels = []
        for matrix in matrices:
            label = predictor.predict(matrix.reshape(1, -1))
            labels.append(label)
        Show_predictions(feature_sequence, labels, i, color_imgs)


if __name__ == "__main__":
    # calculate feature points on all saved Video
    videos = os.listdir(image_path)
    for video in videos:
        if video != ".DS_Store":
            print(video)
            if not os.path.isdir(os.path.join(output_path, video)):
                os.makedirs(os.path.join(output_path, video))

            gray_imgs = read_images(os.path.join(image_path, video))
            color_imgs = read_images(os.path.join(image_path, video), "color")
            ground_truth_frames = read_images(os.path.join(label_path, video))
            # if we first extract feature points:
            feature_sequence = extract_features(color_imgs, "SURF", 1000)
            Retrieve_all_transformation_matrix(feature_sequence, color_imgs, os.path.join(output_path, video), "SURF")

            # # else we load feature points:
            feature_sequence, matrices_on_each_frame = load_data(os.path.join(output_path, video))
            ground_truth_labels = Extract_labels(feature_sequence, ground_truth_frames)

            # print("{} images total for training".format(len(feature_sequence)))
            #
            # NN method
            # if we have pre-trained model:
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # print("NN is predicting on {}".format(device))
            # chkpt = torch.load(os.path.join(ROOT, "Runs", video, "ckpt.pth"), map_location=device)
            # model = MatClassificationNet().to(device)
            # model.load_state_dict(chkpt)

            # model = train_model(matrices_on_each_frame, ground_truth_labels, video, model)

            # else we train NN model for the first time:
            model = train_model(matrices_on_each_frame, ground_truth_labels, video)
            NN_clustering(feature_sequence, matrices_on_each_frame, color_imgs, model, video)

            # Other traditional methods

            # SVM method - work, but poor performance
            # SVM-train
            # train_SVM(matrices_on_each_frame, ground_truth_labels)

            # GMM method - fail
            # model = train_GMM(ground_truth_labels, matrices_on_each_frame)
            # GMM_clustering(model, feature_sequence, matrices_on_each_frame, color_imgs)

