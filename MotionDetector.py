import cv2, time, pandas
from datetime import datetime

#data initializers
#motiontrack=[None, None]
#time=[]
#df1=pandas.DataFrame(columns=["START", "END"])
#motion=0

Video1=cv2.VideoCapture(r"C:\Users\user\Pictures\Camera Roll\video1.mp4")

SampleImage = cv2.imread(r"C:\Users\user\Pictures\Camera Roll\Sample1.png")
SampleImage = cv2.cvtColor(SampleImage, cv2.COLOR_BGR2GRAY)
SampleImage = cv2.GaussianBlur(SampleImage, (21, 21), 0)

while True:
    status, frame = Video1.read()
    Gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    Gray1 = cv2.GaussianBlur(Gray1, (21, 21), 0)
    diff = cv2.absdiff(SampleImage, Gray1)
    threshVid = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)[1]
    threshVid = cv2.dilate(threshVid, None, iterations= 2)
    ImgContour, res = cv2.findContours(threshVid.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in ImgContour:
        if cv2.contourArea(contour) < 350000:
            continue
        #motion=1
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x,y), (x + w, y + h), (0, 255, 0), 5)
        cv2.imshow("Motion Detector", frame)


#    motiontrack.append(motion)
#    motiontrack=motiontrack[-2:]
#    if motiontrack[-1]==1 and motiontrack[-2]==0:
#        time.append(datetime.now())
#    if motiontrack[-1]==0 and motiontrack[-2]==1:
#        time.append(datetime.now())


    key = cv2.waitKey(1)
    if key == ord('q'):
#        if motion==1:
#            time.append(datetime.now())
        break

#for i in range(0, len(time), 2):
#   df1 = df1.append({"Start": time[i], "End": time[i + 1]}, ignore_index=True)
#df1.to_csv("Time_of_movements.csv")

Video1.relase()
cv2.destroyWindows()