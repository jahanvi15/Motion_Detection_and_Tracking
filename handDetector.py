import cv2
import mediapipe as mp
import math

# Hands Detection Objects
mpHands = mp.solutions.hands
Hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

# Pose Detection Objects
mpPose = mp.solutions.pose
pose = mpPose.Pose()

# Face Mesh Detection Objects
mpMesh = mp.solutions.face_mesh
Mask = mpMesh.FaceMesh()


video = cv2.VideoCapture(0)
while True:
    status, frame = video.read()
    ImgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    roi = frame[50: 350, 250: 490]

    #Detection Area Bounds
    center_coordinates_left = (560, 100)
    cv2.circle(frame, center_coordinates_left, 50, (0, 255, 255), cv2.FILLED)
    cv2.putText(frame, "LEFT", (518, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))

    center_coordinates_right = (100, 100)
    cv2.circle(frame, center_coordinates_right, 50, (0, 255, 255), cv2.FILLED)
    cv2.putText(frame, "RIGHT", (55, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))

    # Detection registration
    # 1. Pose
    Result = pose.process(ImgRGB)
    if Result.pose_landmarks:
        mpDraw.draw_landmarks(frame, Result.pose_landmarks, mpPose.POSE_CONNECTIONS)
        #print(id, Result.pose_landmarks)

    # 2. Face
    Result = Mask.process(ImgRGB)
    if Result.multi_face_landmarks:
         for facelms in Result.multi_face_landmarks:
             for id, lm in enumerate(facelms.landmark):
                 h, w, c = frame.shape
                 cx, cy = (lm.x * w), (lm.y * h)
                 print(id, cx, cy)
                 #coordinates1 = []
                 #coordinates2 = []
                 #if id == 159:
                 #    coordinates1.append((cx, cy))

                 #if id == 145:
                     #coordinates2.append((cx, cy))
                     #print(math.dist(coordinates1, coordinates2))
                 #if id == 159 and id == 146:
                 #    cv2.putText(frame, "Eyes close")
             mpDraw.draw_landmarks(frame, facelms, mpMesh.FACEMESH_TESSELATION, drawSpec, drawSpec)

    # # 3. Hands
    Result = Hands.process(ImgRGB)
    if Result.multi_hand_landmarks:
         for handlms in Result.multi_hand_landmarks:
             for id, lms in enumerate(handlms.landmark):
                 h, w, c = frame.shape
                 cx, cy = int(lms.x * w), int(lms.y * h)
                 #print(id, cx, cy)
                 if id == 4:
                     if (535 < cx < 585 and 75 < cy < 125) or (75 < cx < 125 and 75 < cy < 125):
                         cv2.circle(frame, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

             mpDraw.draw_landmarks(frame, handlms, mpHands.HAND_CONNECTIONS)

    cv2.imshow("MOTION DETECTOR", frame)

    cv2.imshow("Region of interest", roi)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break


video.release()
cv2.destroyWindows()
