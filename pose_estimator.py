import mediapipe as mp
import cv2
import numpy as np
import pickle

class PoseEstimator:
    def __init__(self):
        file = open("./Models/SVMPickle", 'rb')
        self.clf_SVM = pickle.load(file)
        file = open("./Models/Scaler", 'rb')
        self.scaler = pickle.load(file)
        self.mpPose = mp.solutions.pose
        self.points = self.mpPose.PoseLandmark  # Landmarks
        self.pose = self.mpPose.Pose()
        self.mpDraw = mp.solutions.drawing_utils  # For drawing keypoints


    def puttext(self, image, RF_intention):
        strings_rf = "empty"
        color_rf = (0, 0, 0)
        if RF_intention == True:
            strings_rf = "model" + ": Intention Yes"
            color_rf = (0, 255, 0)
        if RF_intention == False:
            strings_rf = "model" + ": Intention No"
            color_rf = (0, 0, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 50)
        fontScale = 1
        thickness = 2
        image = cv2.putText(image, strings_rf, org, font,
                            fontScale, color_rf, thickness, cv2.LINE_AA)
        return image


    def estimate_pose(self, frame):
        results = self.pose.process(frame)
        temp = []
        if results.pose_landmarks:
            self.mpDraw.draw_landmarks(frame, results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
            for j in results.pose_landmarks.landmark:
                temp = temp + [j.x, j.y, j.z, j.visibility]
            x = np.asanyarray(temp).reshape(1, -1)
            x_scaled = self.scaler.transform(x)
            # y=clf_SVM.predict(x_scaled)
            # y=clf_RF.predict(x)
            y = self.clf_SVM.predict(x_scaled)
            print(bool(y[0]))
            return bool(y[0])
        return False


# pe = PoseEstimator()
# cap = cv2.VideoCapture(0)
# t = 0
# while t < 150:
#     t = t+1
#     print(t)
#     ret, frame = cap.read()
#     # if frame is read correctly ret is True
#     if not ret:
#         print("Can't receive frame (stream end?). Exiting ...")
#         break
#     print(pe.estimate_pose(frame))
#     res = pe.estimate_pose(frame)
#     frame2 = pe.puttext(frame, res)
#     cv2.imshow("Pose detection", frame2)
#     cv2.imwrite("test.jpg", frame2)
#     cv2.waitKey(30)
