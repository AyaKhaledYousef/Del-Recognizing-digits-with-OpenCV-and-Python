''' imutils is a series of convenience functions to make basic 
image processing functions such as translation, rotation, resizing, 
skeletonization, displaying Matplotlib images, sorting contours,
 detecting edges, and much more easier with OpenCV. '''

# https://github.com/PyImageSearch/imutils
import imutils
from imutils.perspective import four_point_transform
from imutils import contours
import cv2
import matplotlib.pyplot as plt

# define the dictionary of digit segments so we can identify
# each digit on the thermostat
DIGITS_LOOKUP = {
	(1, 1, 1, 0, 1, 1, 1): 0,
	(0, 0, 1, 0, 0, 1, 0): 1,
	(1, 0, 1, 1, 1, 1, 0): 2,
	(1, 0, 1, 1, 0, 1, 1): 3,
	(0, 1, 1, 1, 0, 1, 0): 4,
	(1, 1, 0, 1, 0, 1, 1): 5,
	(1, 1, 0, 1, 1, 1, 1): 6,
	(1, 0, 1, 0, 0, 1, 0): 7,
	(1, 1, 1, 1, 1, 1, 1): 8,
	(1, 1, 1, 1, 0, 1, 1): 9
}
# =============================================================================
# Step #1: Localize the LCD on the thermostat.
# =============================================================================
# load the example image
image = cv2.imread("example.jpg")
# pre-process the image by resizing it, converting it to
# graycale, blurring it, and computing an edge map
image = imutils.resize(image, height=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 50, 200, 255)
#plt.imshow(edged,'gray')

# =============================================================================
# Extract the contours to find the LCD regions
# =============================================================================
# find contours in the edge map, then sort them by their
# size in descending order
'''first it finds the contours from a edge detected image, 
then it sorts the contours to find the five largest contours'''
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
#sort the contours by their area, 
#ensuring that contours with a larger area are placed at the front of the list
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
displayCnt = None

# loop over the contours
'''cv2.arcLength() is used to calculate the perimeter of the contour.
If the second argument is True then it considers the contour to be closed.
Then this perimeter is used to calculate
the epsilon value for cv2.approxPolyDP()'''


for c in cnts:
	# approximate the contour
	peri = cv2.arcLength(c, True)
    # approxPolyDP to smooth and approximate the quadrilateral.
    # It works for the cases where there are sharp edges in the contours like a document boundary.
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	# if the contour has four vertices, then we have found
	# the thermostat display
	if len(approx) == 4:
		displayCnt = approx
		break
# =============================================================================
# Extract the LCD via a four point perspective transform:
# =============================================================================
warped = four_point_transform(gray, displayCnt.reshape(4, 2))
plt.imshow(warped,'gray')
output = four_point_transform(image, displayCnt.reshape(4, 2))
# =============================================================================
# extract the digits from the LCD:
# =============================================================================
thresh = cv2.threshold(warped, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
# apply a series of morphological operations to clean up the thresholded image

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

plt.imshow(thresh,'gray')
# =============================================================================
# 
# =============================================================================
# find contours in the thresholded image, then initialize the
# digit contours lists
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
digitCnts = [] # to store the contours of the digits themselves.
# loop over the digit area candidates
for c in cnts:
	# compute the bounding box for each contour
	(x, y, w, h) = cv2.boundingRect(c)
	# if the contour is sufficiently large, it must be a digit
	if w >= 15 and (h >= 30 and h <= 40):
		digitCnts.append(c)
# =============================================================================
# 
# =============================================================================
# sort the contours from left-to-right, then initialize the
# actual digits themselves
digitCnts = contours.sort_contours(digitCnts,
	method="left-to-right")[0]
digits = []

# =============================================================================
# 
# =============================================================================
# loop over each of the digits
for c in digitCnts:
	# extract the digit ROI
	(x, y, w, h) = cv2.boundingRect(c)
	roi = thresh[y:y + h, x:x + w]
	# compute the width and height of each of the 7 segments
	# we are going to examine
	(roiH, roiW) = roi.shape
	(dW, dH) = (int(roiW * 0.25), int(roiH * 0.15))
	dHC = int(roiH * 0.05)
	# define the set of 7 segments
	segments = [
		((0, 0), (w, dH)),	# top
		((0, 0), (dW, h // 2)),	# top-left
		((w - dW, 0), (w, h // 2)),	# top-right
		((0, (h // 2) - dHC) , (w, (h // 2) + dHC)), # center
		((0, h // 2), (dW, h)),	# bottom-left
		((w - dW, h // 2), (w, h)),	# bottom-right
		((0, h - dH), (w, h))	# bottom
	]
	on = [0] * len(segments)
    # loop over the segments
	for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
		# extract the segment ROI, count the total number of
		# thresholded pixels in the segment, and then compute
		# the area of the segment
		segROI = roi[yA:yB, xA:xB]
		total = cv2.countNonZero(segROI)
		area = (xB - xA) * (yB - yA)
		# if the total number of non-zero pixels is greater than
		# 50% of the area, mark the segment as "on"
		if total / float(area) > 0.5:
			on[i]= 1
	# lookup the digit and draw it on the image
	digit = DIGITS_LOOKUP[tuple(on)]
	digits.append(digit)
	cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)
	cv2.putText(output, str(digit), (x - 10, y - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
# =============================================================================
#     
# =============================================================================
# display the digits
print(u"{}{}.{} \u00b0C".format(*digits))
cv2.imshow("Input", image)
cv2.imshow("Output", output)
cv2.waitKey(0)