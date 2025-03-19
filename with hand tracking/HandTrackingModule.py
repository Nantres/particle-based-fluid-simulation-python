import cv2
import mediapipe as mp
import time

class handDetector():
   def __init__(self, mode=False, maxHands = 2, model_complexity=1,  detectionCon=0.5, trackCon = 0.5): # define parameters of object
      self.mode = mode # setting attributes using values
      self.maxHands = maxHands
      self.modelComplexity = model_complexity
      self.detectionCon = detectionCon
      self.trackCon = trackCon

      self.mpHands = mp.solutions.hands
      self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity, self.detectionCon, self.trackCon)
      self.mpDraw = mp.solutions.drawing_utils

   def findHands(self, img, draw=True):
      imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      self.results = self.hands.process(imgRGB)
      # print(results.multi_hand_landmarks)

      self.handLms = self.results.multi_hand_landmarks
      if self.handLms:
         for handLm in self.handLms: # for each hand
            if draw:
               self.mpDraw.draw_landmarks(img, handLm, self.mpHands.HAND_CONNECTIONS) # draw landmarks on displayed image
      return img

   def findPosition(self, img, handNo=0, draw=True, lmNo=8):
      lmList = []
      if self.handLms:
         handlm = self.handLms[handNo]
         for id, lm in enumerate(handlm.landmark): # iterate through landmarks
            h, w, c = img.shape # get height, width and channels of img
            cx, cy = int(lm.x*w), int(lm.y*h)
            # print(id,cx,cy)
            lmList.append([id, cx, cy])
            if id == lmNo:
               if draw:
                  cv2.circle(img, (cx, cy), 7, (255, 50 ,255), cv2.FILLED) # pos, size, color
      return (lmList)

   def findPositionAll(self, img, draw=True, lmNo=8):
      lmList = []
      if self.handLms:
         for handNo, handlm in enumerate(self.handLms):
            lmList.append([])
            for id, lm in enumerate(handlm.landmark): # iterate through landmarks
               h, w, c = img.shape # get height, width and channels of img
               cx, cy = int(lm.x*w), int(lm.y*h)
               # print(id,cx,cy)
               lmList[handNo].append([id, cx, cy])
               if id == lmNo:
                  if draw:
                     cv2.circle(img, (cx, cy), 7, (255, 50 ,255), cv2.FILLED) # pos, size, color
      return (lmList)


# if __name__ == "__main__":
#    main()