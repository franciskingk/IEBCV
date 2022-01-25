import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.image as mpimg
import pytesseract
import pickle
import cvzone
from sre_constants import SUCCESS
import pyqrcode
from pyzbar.pyzbar import decode
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\angie\Tesseract-OCR\tesseract.exe'

import time
start_time = time.time()

# specifying the folder path containing the images
image_path = r'C:\Users\angie\Documents\Moringa School\Final Project - Computer Vision\ballot papers\GVnairobidagoballot-1_LI (88).jpg'

# loading the images
# the image in PIL format
image = Image.open(image_path)
# the image in array format
img_array = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

# function for greyscaling an image
def grayscale(img_array):
  return cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

# function for preprocessing before contouring
def my_prep_cont(image):
  # blurring image
  img_blur = cv2.GaussianBlur(image, (7,7),0 )
  img_blur = img_blur.astype(np.uint8)
  # setting threshold and kernel
  thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10,12))
  # dilating the image
  dilate = cv2.dilate(thresh, kernel, iterations =1)
  return dilate


# function for processing the image that will be fed into the mark recognition function
def process_mark_reg(img_array):
  # grayscaling the image
  imgGray = cv2.cvtColor(img_array,cv2.COLOR_BGRA2GRAY)
  # appying blur
  imgBlur = cv2.GaussianBlur(imgGray,(3,3),1)
  imgThreshhold = cv2.adaptiveThreshold(imgBlur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
  cv2.THRESH_BINARY_INV,25,16)
  imgmedialblur = cv2.medianBlur(imgThreshhold,3)
  # setting the kernel for dilation
  kernel = np.ones((3,3),np.uint8)
  # dilating the image after blurring
  imgdialate = cv2.dilate(imgmedialblur,kernel, iterations=1)
  return imgdialate


# dilation to make text thicker
def thick_font(image):
  image = cv2.bitwise_not(image)
  kernel = np.ones((2,2), np.uint8) # increasing the kernel size makes the font thicker
  image = cv2.dilate(image, kernel, iterations=1) #increasing the number of iterations makes the font thicker as well 
  image = cv2.bitwise_not(image)
  return image

# specifying the coordinates for the positions of the bounding boxes for candidate names and the marks

# storing the data for their coordinates in a list
# the arrangement is typically in tuple format in the following order(left, top, right, bottom)
candi_coordinates = [(244,250,474,320),(244,320,474,380),(244,390,474,470),(244,480,474,550),(244,550,474,620),(244,625,474,660)]
pos_marks = [(502, 259), (502, 331), (502, 401), (502, 478), (501, 553), (502, 626)]
# out_of_bounds = [(584, 256), (584, 328), (582, 404), (584, 480), (583, 553), (585, 626)]
# after_name = [(384, 261), (427, 330), (356, 403), (356, 481), (383, 554), (418, 627)]
# before_name = [(226, 259), (229, 330), (237, 403), (239, 480), (238, 557), (236, 625)]


# defining the function for checking the marks on the ballot papers
def checkformark(imgpro):
  thicker = thick_font(grayscale(imag))
  thicker = Image.fromarray(np.uint8(thicker))
  i = 0
  boxcounter = 0
  for pos in pos_marks:
    #specifying the coordinates for our mark bounding boxes
    x,y = pos

    # cropping the image and computing the number of pixels in each
    imgcrop = imgpro[y:y+height, x:x+width]

    # cropping the candidate name
    cropped  = thicker.crop(candi_coordinates[i])
    # cropping the ballot paper elective position
    b_cropped = thicker.crop((0,0,660,55))

    # counting number of pixels
    count = cv2.countNonZero(imgcrop)
    cvzone.putTextRect(imag, str(count),(x,y+height-10), scale=1,thickness=1,offset=0)

    
    # to determine the where the ballot is marked the count for pixels is investigated.
    # we set a condition that where the number of pixels exceed 700, the ballot is considered marked
    if count > 100:
      color =(255,0,0)
      thickness = 4
      cv2.rectangle(imag,pos,(pos[0]+width-350,pos[1]+height),(255,0,255),1)
      read_name = pytesseract.image_to_string(cropped)
      boxcounter += 1

    else:
      color = (0,0,255)
      thickness = 2

    cv2.rectangle(imag,pos,(pos[0]+width,pos[1]+height),color,thickness)
    i +=1

  if boxcounter == 1:
    countable = "countable vote"
  else:
    countable = "spoilt vote"
    read_name = "spoilt"
  
  office = pytesseract.image_to_string(b_cropped)

  if "SENATOR" in office:
    position_name = "senate"
  elif "WOMAN" in office:
    position_name = "women rep"
  elif "PARLIAMENT" in office:
    position_name = "member of parliament"
  elif "GOVERNOR" in office:
    position_name = "governor"
  elif "PRESIDENT" in office:
    position_name = "president"
  elif "WARD" in office:
    position_name = "mca"

  
  return imag, read_name, countable, position_name


# resizing the image
imgwidth = 660
imgheight = 750
imag = cv2.resize(img_array,(imgwidth,imgheight))

# setting dimensions for bounding boxes
width, height = 70, 60

# creating another copy of the image
clone = imag.copy()

# processing the image for contouring
processed_imag = process_mark_reg(imag)

# reading the qr code
qr_image = Image.fromarray(np.uint8(clone))

qr_cropping = qr_image.crop((490, 50, 603,150))

d = decode(qr_cropping)
data = d[0].data.decode('ascii')
data_split = data.split(",")

# obtaining the the markchecked image, name of the candidate chosen, 
# whether the vote is spolit or not and the elective position the ballot paper is for
markchecked, name, countable, position  = checkformark(processed_imag)

# printing output
print("This vote was:", data_split[0])

print("County of origin:", data_split[1])
print("Constituency of origin:", data_split[2])
print("Polling station:", data_split[3])
print("Position:", position)
print("Candidate chosen:",name)
print("This vote was a",countable ,".")

cv2.imwrite(r"C:\Users\angie\Documents\Moringa School\Final Project - Computer Vision\counted votes\markchecked.jpg", markchecked)


print("--- %s seconds ---" % (time.time() - start_time))