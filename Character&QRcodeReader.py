import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import pytesseract
import pickle
import cvzone
from sre_constants import SUCCESS
import pyqrcode
from pyzbar.pyzbar import decode
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\angie\Tesseract-OCR\tesseract.exe'

# specifying the folder path containing the images
folder_path = r'C:\Users\angie\Documents\Moringa School\Final Project - Computer Vision\ballot papers'

# defining the function we are going to make use of 

# function for greyscaling an image
def grayscale(img_array):
  return cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)


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


# creating an empty dictionary for the text to be read
data_dict = {}

# function for creating values in my dictionary
def add_vals(dic,key, values):
  if key not in dic:
    dic[key] = list()
  dic[key].extend(values)


# specifying the coordinates for the positions of the bounding boxes for candidate names and the marks

# storing the data for their coordinates in a list
# the arrangement is typically in tuple format in the following order(left, top, right, bottom)
candi_coordinates = [(244,250,474,320),(244,320,474,380),(244,390,474,470),(244,480,474,550),(244,550,474,620),(244,620,474,670)]
pos_marks = [(497, 257), (497, 328), (499, 401), (498, 478), (499, 553), (499, 625)]


# defining the function for checking the marks on the ballot papers
def checkformark(imgpro):
  thicker = thick_font(grayscale(imag))
  thicker = Image.fromarray(np.uint8(thicker))
  i = 0
  boxcounter = 0
  for pos in pos_marks:
    #specifying the coordinates for our mark bounding boxes
    x,y = pos

    # cropping the image for computing the number of pixels in each marking box
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
    if count > 700:
      color =(255,0,0)
      thickness = 5
      cv2.rectangle(imag,pos,(pos[0]+width-350,pos[1]+height),(255,0,255),thickness)
      boxcounter += 1
      read_name = pytesseract.image_to_string(cropped)

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
  
  position_name = pytesseract.image_to_string(b_cropped)

  
  return imag, read_name, countable, position_name


# using a for loop to iterate through the folder

m = 0
for img_path in os.listdir(folder_path):

    # writing the image path
    path = os.path.join(folder_path, img_path)

    # loading the images
    # the image in PIL format
    image = Image.open(path)
    # the image in array format
    img_array = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    # making a copy of the ballot paper
    cont_img = img_array.copy()
    # preprocessing my image for contouring
    gray_img = grayscale(img_array)
    to_contour = my_prep_cont(gray_img)

    # drawing contours onto my image to pinpoint the bounding box containing the qr code
    cnts = cv2.findContours(to_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0 if len(cnts) == 2 else cnts[1]]
    cnts = sorted(cnts, key =lambda x: cv2.boundingRect(x)[0])
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if h > 150 and w >  150 and h < 300 and w < 300:
            roi = img_array[y:y+h, x:x+w]
    d = decode(roi)
    data = d[0].data.decode('ascii')
    data_split = data.split(",")

    # resizing the image
    imgwidth = 660
    imgheight = 750
    imag = cv2.resize(img_array,(imgwidth,imgheight))

    # setting dimensions for bounding boxes
    width, height = 80, 65

    # creating another copy of the image
    clone = imag.copy()

    # processing the image for contouring
    processed_imag = process_mark_reg(clone)

    # obtaining the the markchecked image, name of the candidate chosen, 
    # whether the vote is spolit or not and the elective position the ballot paper is for
    markchecked, name, countable, position  = checkformark(processed_imag)

    # adding the values into our dictionary
    add_vals(data_dict, "Status", [data_split[0]])
    add_vals(data_dict, "County", [data_split[1]])
    add_vals(data_dict, "Constituency", [data_split[2]])
    add_vals(data_dict, "Polling Station", [data_split[3]])
    add_vals(data_dict, "Candidate chosen", [name])
    add_vals(data_dict, "Validity", [countable])
    add_vals(data_dict, "Elective position", [position])

    # adding the images with their bounding boxes drawn into a directory
    # cv2.imwrite(r"C:\Users\angie\Documents\Moringa School\Final Project - Computer Vision\counted votes" +"\\"+"image"+f'{m}'+'.jpg', markchecked)
    # m += 1
    

# converting my dictionary into a dataframe
df = pd.DataFrame.from_dict(data_dict)
df.head()

# saving my csv file to my directory
df.to_csv(r'C:\Users\angie\Documents\Moringa School\Final Project - Computer Vision\data\draft_data.csv', encoding='utf-8')

   
    
    




    

    

