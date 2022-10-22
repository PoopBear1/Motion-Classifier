import cv2
import os


def MaskMakeVedio(file_path, type_V="temp_mask", file_name_template="%03d.png"):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    cap = cv2.VideoCapture(os.path.join(file_path, type_V + file_name_template))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    videoWrite = cv2.VideoWriter(
        os.path.join(file_path, "MaskDemo.avi"),
        fourcc,
        10,
        (width, height),
        isColor=False,
    )
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # videoWrite.write(cv2.flip(frame,1))
        videoWrite.write(frame)

    cap.release()
    videoWrite.release()


def TrackMakeVedio(file_path, type_V="track", file_name_template="%03d.png"):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    cap = cv2.VideoCapture(os.path.join(file_path, type_V + file_name_template))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    videoWrite = cv2.VideoWriter(
        os.path.join(file_path, "TrackDemo.avi"), fourcc, 10, (width, height)
    )
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # videoWrite.write(cv2.flip(frame,1))
        videoWrite.write(frame)

    cap.release()
    videoWrite.release()


def ResultsMakeVedio(file_path, type_V="result", file_name_template="%05d.jpg"):
    print(file_path)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    cap = cv2.VideoCapture(os.path.join(file_path, file_name_template))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    videoWrite = cv2.VideoWriter(
        os.path.join(file_path, "car.avi"), fourcc, 60, (width, height)
    )
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # videoWrite.write(cv2.flip(frame,1))
        videoWrite.write(frame)

    cap.release()
    videoWrite.release()

input_path = os.path.join(os.getcwd(), "temp")
ResultsMakeVedio(input_path)