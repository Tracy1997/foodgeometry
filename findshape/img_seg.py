import cv2
import numpy as np
import sys

def getAreaOfFood(img1):
	# convert to hsv. otsu threshold in s to remove plate
	hsv_img = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
	h,s,v = cv2.split(hsv_img)
	background = cv2.inRange(hsv_img, np.array([0,0,0]), np.array([180,140,255]))
	not_background = cv2.bitwise_not(background)
	fruit = cv2.bitwise_and(img1,img1,mask = not_background)

	fruit_bw = cv2.cvtColor(fruit, cv2.COLOR_BGR2GRAY)
	fruit_bin = cv2.inRange(fruit_bw, 10, 255) #binary of fruit

	#erode before finding contours
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
	erode_fruit = cv2.erode(fruit_bin,kernel,iterations = 1)

	#cv2.imshow('erode_fruit',erode_fruit)
	#cv2.waitKey(0)

	#find largest contour since that will be the fruit
	img_th = cv2.adaptiveThreshold(erode_fruit,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
	contours, hierarchy = cv2.findContours(img_th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	mask_fruit = np.zeros(fruit_bin.shape, np.uint8)
	largest_areas = sorted(contours, key=cv2.contourArea)
	if (len(largest_areas)==1):
		fruit_contour = largest_areas[-1]
	else:
		fruit_contour = largest_areas[-2]
	cv2.drawContours(mask_fruit, [fruit_contour], 0, (255,255,255), -1)

	#cv2.imshow('mask_fruit',mask_fruit)
	#cv2.waitKey(0)

	#dilate now
	kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13))
	mask_fruit2 = cv2.dilate(mask_fruit,kernel2,iterations = 1)
	res = cv2.bitwise_and(fruit_bin,fruit_bin,mask = mask_fruit2)
	fruit_final = cv2.bitwise_and(img1,img1,mask = mask_fruit2)

	#cv2.imshow('fruit_final',fruit_final)
	#cv2.waitKey(0)

	#find area of fruit
	img_th = cv2.adaptiveThreshold(mask_fruit2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
	contours, hierarchy = cv2.findContours(img_th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	largest_areas = sorted(contours, key=cv2.contourArea)
	#fruit_contour = largest_areas[-2]
	if (len(largest_areas)==1):
		fruit_contour = largest_areas[-1]
	else:
		fruit_contour = largest_areas[-2]
	fruit_area = cv2.contourArea(fruit_contour)

	mask_fruit3 = np.zeros(img_th.shape, np.uint8)
	cv2.drawContours(img1, fruit_contour, -3, (255,255,0), 3)
	
	#cv2.imshow('img1',img1)
	#cv2.waitKey(0)

	
	skin_area=0
	pix_to_cm_multiplier=0.005

	return fruit_area, mask_fruit2, fruit_final, skin_area, fruit_contour, pix_to_cm_multiplier


if __name__ == '__main__':
	img1 = cv2.imread(sys.argv[1])
	area, bin_fruit, img_fruit, skin_area, fruit_contour, pix_to_cm_multiplier = getAreaOfFood(img1)

	cv2.waitKey()
	cv2.destroyAllWindows()

