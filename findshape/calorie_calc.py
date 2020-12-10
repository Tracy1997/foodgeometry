import cv2
import numpy as np
import sys

#density - gram / cm^3
density_dict = { 1:0.609, 2:0.94, 3:0.577, 4:0.641, 5:1.151, 6:0.482, 7:0.513, 8:0.641, 9:0.481, 10:0.641, 11:0.521, 12:0.881, 13:0.228, 14:0.650 }
#kcal
calorie_dict = { 1:52, 2:89, 3:92, 4:41, 5:360, 6:47, 7:40, 8:158, 9:18, 10:16, 11:50, 12:61, 13:31, 14:30 }
#skin of photo to real multiplier
skin_multiplier = 0.00003

def getCalorie(label, volume): #volume in cm^3
	'''
	Inputs are the volume of the foot item and the label of the food item
	so that the food item can be identified uniquely.
	The calorie content in the given volume of the food item is calculated.
	'''
	calorie = calorie_dict[int(label)]
	if (volume == None):
		return None, None, calorie
	density = density_dict[int(label)]
	mass = volume*density*1.0
	calorie_tot = (calorie/100.0)*mass
	return mass, calorie_tot, calorie #calorie per 100 grams

def getVolume(label, area, skin_area, pix_to_cm_multiplier, fruit_contour):
	'''
	Using callibration techniques, the volume of the food item is calculated using the
	area and contour of the foot item by comparing the foot item to standard geometric shapes
	'''
	area_fruit = area*skin_multiplier #area in cm^2	
	fruit_rect = cv2.minAreaRect(fruit_contour)
	height = max(fruit_rect[1])*pix_to_cm_multiplier
	radius = area_fruit/(2.0*height)
	print(height, radius)
	volume = np.pi*radius*radius*height
	
	
	return volume
