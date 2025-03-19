import cv2
import time
import HandTrackingModule as htm
from multiprocessing import Process
import threading
from collections import deque
import sys
import imutils
import math
import water_simulation_with_hand_tracking as water_simulation

a = threading.Thread(target=water_simulation.main, args=())
a.start()

# camera stuff
wCam, hCam = 640, 400
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, wCam)
cap.set(4, hCam)

# u can change these variables
landmarks = [8]
no_of_loops = 3
camera_display = False

# constants
pTime=0
arucoTime = 0
nohand_frames = 0
aruco_data = None
projector_corners = None
detector = htm.handDetector(detectionCon=0.4)
prev_finger_data = deque(maxlen=no_of_loops)  #stores last 2 loops of finger data
aruco_type = "DICT_4X4_50"
ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}


# verify that the supplied ArUCo tag exists and is supported by OpenCV
if ARUCO_DICT.get(aruco_type, None) is None:
    print(f"[INFO] ArUCo tag of '{aruco_type}' is not supported")
    sys.exit(0)

# load the ArUCo dictionary, grab the ArUCo parameters, and detect the markers
print(f"[INFO] detecting '{aruco_type}' tags...")
arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])
arucoParams = cv2.aruco.DetectorParameters()

def zoom(img, zoom_factor):
    cropped = img[int((hCam*zoom_factor-hCam)/(2*zoom_factor)):int((hCam*zoom_factor+hCam)//(2*zoom_factor)), int((wCam*zoom_factor-wCam)//(2*zoom_factor)):int((wCam*zoom_factor+wCam)//(2*zoom_factor))]
    return cropped
    #return cv2.resize(cropped, None, fx=zoom_factor + zoom2, fy=zoom_factor + zoom2)

def get_aruco_data(image, aruco_data):
    (corners, ids, rejected) = cv2.aruco.ArucoDetector(arucoDict, arucoParams).detectMarkers(image)

    # verify *at least* one ArUco marker was detected
    if len(corners) >= 2:
        # flatten the ArUco IDs list
        ids = ids.flatten()
        if ids[0] == 5 and ids[1] == 6:
            aruco_data = [0,0,0]    #top left corner, angle difference, multiplier
            projector_corners = [0,0]
        else:
            print("Aruco markers detected are not 5 and 6")
        # loop over the detected ArUCo corners
        for (markerCorner, markerID) in zip(corners, ids):
            # extract the marker corners (which are always returned in
            # top-left, top-right, bottom-right, and bottom-left order)
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners
            # convert each of the (x, y)-coordinate pairs to integers
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            if camera_display == True:
                # draw the bounding box of the ArUCo detection
                cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
                cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
                cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
                cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)

                # draw the ArUco marker ID on the image
                cv2.putText(image, str(markerID),
                    (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
            
            if markerID == 5:
                projector_corners[0] = bottomRight
            elif markerID == 6:
                projector_corners[1] = bottomLeft
        
        aruco_data[0] = projector_corners[0]
        aruco_data[1] = -angle_btwn_vectors((1,0), calculate_vector(projector_corners[0], projector_corners[1]))
        aruco_data[2] = 1920/calculate_distance(projector_corners[0], projector_corners[1])
    
    else:
        print("Two Aruco markers not detected")
        
    return aruco_data

def angle_btwn_vectors(vector1, vector2):
    dot_product = sum(p*q for p,q in zip(vector1, vector2))
    magnitude1 = math.sqrt(sum([val**2 for val in vector1]))
    magnitude2 = math.sqrt(sum([val**2 for val in vector2]))
    return math.degrees(math.acos(dot_product/(magnitude1*magnitude2)))

def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def calculate_vector(vector1, vector2):
    return (vector2[0] - vector1[0], vector2[1] - vector1[1])

def rotate_vector(vector, angle):
    x = vector[0]*math.cos(math.radians(angle)) - vector[1]*math.sin(math.radians(angle))
    y = vector[0]*math.sin(math.radians(angle)) + vector[1]*math.cos(math.radians(angle))
    return (x, y)

print("[INFO] Starting hand tracking...")
while True:
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    success, img = cap.read()
    img = zoom(img, 1.4)    
    img = imutils.resize(img, height=1200)

    if cTime - arucoTime > 3:
        aruco_data = get_aruco_data(img, aruco_data)
        arucoTime = cTime

    if aruco_data == None and camera_display == True:
        cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)
        continue

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        new_finger_data = [(lmList[i][1], lmList[i][2]) for i in landmarks]
        prev_finger_data.append(new_finger_data)
        with open("memory.txt", "w") as file:
            if len(prev_finger_data) == no_of_loops:
                average_finger_data = [[int(sum(x_or_y)/no_of_loops) for x_or_y in zip(*finger)] for finger in zip(*prev_finger_data)]                
                for i in average_finger_data:
                    projector_corner_to_finger = calculate_vector(aruco_data[0], i)
                    # cv2.line(img, aruco_data[0], i, (0, 255, 0), 2)
                    rotated_vector = rotate_vector(projector_corner_to_finger, aruco_data[1])
                    # cv2.line(img, aruco_data[0], (int(rotated_vector[0]+aruco_data[0][0]), int(rotated_vector[1]+aruco_data[0][1])), (255, 0, 0), 2)
                    scaled_vector = (int(rotated_vector[0]*aruco_data[2]), int(rotated_vector[1]*aruco_data[2]*0.96))
                    # cv2.line(img, aruco_data[0], (int(scaled_vector[0]+aruco_data[0][0]), int(scaled_vector[1]+aruco_data[0][1])), (255,0,0), 2)
                    file.write(str(scaled_vector[0]) + "\n" + str(scaled_vector[1]) + "\n")
            else:
                for i in new_finger_data:
                    projector_corner_to_finger = calculate_vector(aruco_data[0], i)
                    rotated_vector = rotate_vector(projector_corner_to_finger, aruco_data[1])
                    scaled_vector = (int(rotated_vector[0]*aruco_data[2]), int(rotated_vector[1]*aruco_data[2]))
                    file.write(str(scaled_vector[0]) + "\n" + str(scaled_vector[1]) + "\n")
        nohand_frames = 0
    else:
        if nohand_frames > 10 and nohand_frames < 20:
            file = open("memory.txt", "w")
            file.close()
        if nohand_frames < 20:
            nohand_frames += 1

    if camera_display == True:
        cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


