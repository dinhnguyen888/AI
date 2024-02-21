import cv2
import numpy as np

def diffUpDown(img):
    height, width, depth = img.shape
    half = height // 2
    top = img[0:half, 0:width]
    bottom = img[half:half+half, 0:width]
    top = cv2.flip(top, 1)
    bottom = cv2.resize(bottom, (32, 64))
    top = cv2.resize(top, (32, 64))
    return mse(top, bottom)

def diffLeftRight(img):
    height, width, depth = img.shape
    half = width // 2
    left = img[0:height, 0:half]
    right = img[0:height, half:half + half - 1]
    right = cv2.flip(right, 1)
    left = cv2.resize(left, (32, 64))
    right = cv2.resize(right, (32, 64))
    return mse(left, right)

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def isNewRoi(rx, ry, rw, rh, rectangles):
    for r in rectangles:
        if abs(r[0] - rx) < 40 and abs(r[1] - ry) < 40:
            return False
    return True

def detectRegionsOfInterest(frame, cascade):
    scaleDown = 2
    frame = cv2.resize(frame, (frame.shape[1] // scaleDown, frame.shape[0] // scaleDown))
    cars = cascade.detectMultiScale(frame, 1.2, 1)

    newRegions = []
    minY = int(frame.shape[0] * 0.3)

    for (x, y, w, h) in cars:
        roi = [x, y, w, h]
        roiImage = frame[y:y+h, x:x+w]

        carWidth = roiImage.shape[0]
        if y > minY:
            diffX = diffLeftRight(roiImage)
            diffY = round(diffUpDown(roiImage))
            if 1600 < diffX < 3000 and diffY > 12000:
                rx, ry, rw, rh = roi
                newRegions.append([rx * scaleDown, ry * scaleDown, rw * scaleDown, rh * scaleDown])

    return newRegions

def carTracking(video_path):
    rectangles = []
    cascade = cv2.CascadeClassifier('cars.xml')
    vc = cv2.VideoCapture(video_path)

    if not vc.isOpened():
        print("Error: Could not open video file.")
        return

    frameCount = 0

    while True:
        rval, frame = vc.read()

        if not rval:
            print("End of video.")
            break

        newRegions = detectRegionsOfInterest(frame, cascade)

        for region in newRegions:
            if isNewRoi(region[0], region[1], region[2], region[3], rectangles):
                rectangles.append(region)

        for r in rectangles:
            cv2.rectangle(frame, (int(r[0]), int(r[1])), (int(r[0] + r[2]), int(r[1] + r[3])), (0, 0, 255), 3)

        frameCount += 1
        if frameCount > 30:
            frameCount = 0
            rectangles = []

        cv2.imshow("Car Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vc.release()
    cv2.destroyAllWindows()

# Example usage:
video_file_path = 'road.avi'
carTracking(video_file_path)
