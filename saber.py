import cv2
import numpy as np
# from google.colab.patches import cv2_imshow

def getContours(img,cThread=[100,100],minArea=1000, filter=4):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur,cThread[0],cThread[1])
    kernel = np.ones((5,5))
    imgDilation = cv2.dilate(imgCanny, kernel, iterations = 3)
    imgThre = cv2.erode(imgDilation, kernel, iterations = 2)

    contours, _ = cv2.findContours(imgThre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_countours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > minArea:
            peri = cv2.arcLength(contour,True)
            approx = cv2.approxPolyDP(contour,0.02*peri,True)
            bbox = cv2.boundingRect(approx)
            if filter > 0:
                if len(approx) == filter:
                    final_countours.append([len(approx),area,approx,bbox,contour])
            else:
                final_countours.append([len(approx),area,approx,bbox,contour])
    final_countours = sorted(final_countours, key = lambda x:x[1], reverse=True)

    return img, final_countours

def reorder(myPoints):
	myPointsNew = np.zeros_like(myPoints)
	myPoints = myPoints.reshape((4,2))

	add = myPoints.sum()
	diff = np.diff(myPoints,axis =1)

	myPointsNew[0] = myPoints[np.argmin(add)]
	myPointsNew[3] = myPoints[np.argmax(add)]
	
	myPointsNew[1] = myPoints[np.argmin(diff)]
	myPointsNew[2] = myPoints[np.argmax(diff)]

	return myPointsNew

def wrapImage(img, points, widthImg, heightImg, pad = 0):
    points= get_4_contour(points)
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
    matrix = cv2.getPerspectiveTransform(pts1, pts2) 
    wrap = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    wrap = wrap[pad:wrap.shape[0]-pad,pad:wrap.shape[1]-pad]
    return wrap

def get_4_contour(points):
    center = np.mean(points, axis=0).astype(int)

    points_above_center = np.array([point.squeeze() for point in points if point.squeeze()[1] < center[0][1]])
    points_below_center = np.array([point.squeeze() for point in points if point.squeeze()[1] >= center[0][1]])

    top_left = points_above_center[np.argmin(points_above_center[:, 0])]
    top_right = points_above_center[np.argmax(points_above_center[:, 0])]
    botton_left = points_below_center[np.argmin(points_below_center[:, 0])]
    botton_right = points_below_center[np.argmax(points_below_center[:, 0])]
    return np.array([[top_left], [top_right], [botton_left],[botton_right]])