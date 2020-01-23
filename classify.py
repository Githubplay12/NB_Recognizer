from __future__ import print_function
import joblib
from hogit import HOG
import dataset_trans
import argparse
import mahotas
import cv2
import imutils

ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model', required = True)
ap.add_argument('-i', '--image', required = True)
args = vars(ap.parse_args())

model = joblib.load(args['model'])

hog = HOG(orientations = 18, pixelsPerCell = (10, 10),
          cellsPerBlock = (1, 1), transform = True)

image = cv2.imread(args['image'])
print(image.shape[:2])
if image.shape[1] > 300 or image.shape[0] > 250:
    image = imutils.resize(image, width = 300, height = 250)
#image = imutils.rotate(image, 90)
cv2.imshow('resized and rotated', image)
cv2.waitKey(0)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 30, 150)
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)

contours = sorted([(c, cv2.boundingRect(c)[0]) for c in contours], key = lambda x: x[1])

for (c, _) in contours:
    (x, y, w, h) = cv2.boundingRect(c)

    if w >= 7 and h >= 20:
        roi = gray[y:y + h, x:x + w]
        thresh = roi.copy()
        T = mahotas.thresholding.otsu(roi)
        thresh[thresh > T] = 255
        thresh = cv2.bitwise_not(thresh)

        thresh = dataset_trans.deskew(thresh, 20)
        thresh = dataset_trans.center_extent(thresh, (20, 20))

        cv2.imshow('thresh', thresh)

        hist = hog.describe(thresh)
        digit = model.predict([hist])[0]
        print(f'I think that number is {digit}')

        cv2.rectangle(image, (x, y), (x + w, y + h),
                      (0, 255, 0), 1)
        cv2.putText(image, str(digit), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.imshow('image', image)
        cv2.waitKey(0)


