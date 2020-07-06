"""
problem: develop a system for selfie capture

working:
1. fetch frames from webcam
2. use prebuild model to detect face.
3. if person is smiling, click image in every 1 sec interval

note: code should be properly documented with comments and flowchart shoud match with your code(not just logic)
"""


# to calculate distance between the points on mouth
import os

from scipy.spatial import distance as dist

# video streaming library , reading video frames
from imutils.video import VideoStream, FPS

from imutils import face_utils
import imutils

#for delaying purpose
import time

# pre-trained model which gives the location of each part available on face
import dlib

# image processing library
import cv2



"""Dlib is a landmark’s facial detector with pre-trained models, 
the dlib is used to estimate the location of 68 coordinates (x, y) that map the facial points on a person’s face"""

def smile(mouth):
    """this function calculates the distnce between coordinates of the mouth received from shape detector """

    # please refer the image
    A = dist.euclidean(mouth[3], mouth[9])
    B = dist.euclidean(mouth[2], mouth[10])
    C = dist.euclidean(mouth[4], mouth[8])

    # as some people have different mouth shape so we are finding avg points so ,
    # that every person's MAR's prediction will be most accurate
    avg = (A + B + C) / 3

    # horizontal distance between the righmost point of the lip and leftmost point of the lips
    D = dist.euclidean(mouth[0], mouth[6])

    #formula to find (MOUTH_ASPECT_RATIO = ((A+B+C)/3) / D ))
    mar = avg / D
    return mar


COUNTER = 0
TOTAL = 0


# # initialize dlib's face detector (HOG-based) and then create
# # the facial landmark predictor
shape_predictor = "shape_predictor_68_face_landmarks.dat"

# Returns the default face detector
detector = dlib.get_frontal_face_detector()

# initializing predictor which outputs a set of point locations that define the pose of the object like eyes, mouth , nose
predictor = dlib.shape_predictor(shape_predictor)

# starting point of the mouth and ending point of the mouth
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

print("[INFO] starting video stream thread...")
vs = VideoStream(src=0).start()
fileStream = False
time.sleep(1.0)

# starts a timer that we can use to measure FPS,
fps = FPS().start()
cv2.namedWindow("Frame")

while True:
    # read frames
    frame = vs.read()

    # resize frames
    frame = imutils.resize(frame, width=450)

    # conversion of frame to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detects the eyes, nose , mouth, eyebrows
    rects = detector(gray, 0)
    for rect in rects:

        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # initializing coordinates of mouth stating point and ending point
        mouth = shape[mStart:mEnd]


        # passing mouth points to smile function returns mouth aspect ratio
        mar = smile(mouth)

        # detecting boundaries of the mouth
        mouthHull = cv2.convexHull(mouth)


        #draw boundaries to the mouth
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

        if mar <= .29 or mar > .35:
            COUNTER += 1
        else:
            if COUNTER >= 15:
                TOTAL += 1
                frame = vs.read()
                # if the person is smiling for one second then the camera will capture the image
                time.sleep(.1)
                frame2 = frame.copy()
                img_name = "smile:{}.png".format(TOTAL)
                cv2.imwrite(img_name, frame)

                print("{} saved successfully!".format(img_name))
            COUNTER = 0

        # put the MAR of the person on the image
        cv2.putText(frame, "MAR: {}".format(mar), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)

    # get the new frames
    fps.update()

    key2 = cv2.waitKey(1) & 0xFF
    if key2 == ord('q'):
        break
# stop receiving frames
fps.stop()

# destroy all windows
cv2.destroyAllWindows()

#stop video source
vs.stop()