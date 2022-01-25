import os
import cv2
import time
import pickle
import cvzone
import pyqrcode
import pytesseract
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pyzbar.pyzbar import decode
from sre_constants import SUCCESS
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\angie\Tesseract-OCR\tesseract.exe'

# starting the time counter
start_time = time.time()

# specifying the folder path containing the images
folder_path = r'C:\Users\angie\Documents\Moringa School\Final Project - Computer Vision\ballot papers'

# specifying the coordinates for the positions of the bounding boxes for candidate names and the marks

# storing the data for their coordinates in a list
# the arrangement of candidate coordinates is typically in tuple format in the following order(left, top, right, bottom)
candi_coordinates = [(244,250,474,320),(244,320,474,380),(244,390,474,470),(244,480,474,550),(244,550,474,620),(244,625,474,660)]
pos_marks = [(502, 259), (502, 331), (502, 401), (502, 478), (503, 553), (502, 626)]

# setting image height and width
imgwidth = 660
imgheight = 750

# setting dimensions for bounding boxes
width, height = 70, 60




# defining the functions we are going to make use of 

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


# defining the function for checking the marks on the ballot papers
def checkformark(imgpro):
  thicker = thick_font(grayscale(imag))
  thicker = Image.fromarray(np.uint8(thicker))
  i = 0
  boxcounter = 0
  for pos in pos_marks:
    #specifying the coordinates for our mark bounding boxes
    x,y = pos

    # cropping the image and computing the number of pixels in each mark box
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
    if count > 50:
      color =(255,0,0)
      thickness = 4
      cv2.rectangle(imag,pos,(pos[0]+width-350,pos[1]+height),(255,0,255),thickness)
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

# creating an empty dictionary for the text to be read
data_dict = {}

# function for creating values in my dictionary
def add_vals(dic,key, values):
  if key not in dic:
    dic[key] = list()
  dic[key].extend(values)




# using a for loop to iterate through the folder

m = 0
v = 0
s = 0
c = 0
for img_path in os.listdir(folder_path):

    # writing the image path
    path = os.path.join(folder_path, img_path)

    # loading the images
    # the image in PIL format
    image = Image.open(path)
    # the image in array format
    img_array = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    # resizing the image
    imag = cv2.resize(img_array,(imgwidth,imgheight))

    # reading the qr code
    qr_image = Image.fromarray(np.uint8(imag))

    qr_cropping = qr_image.crop((490, 50, 603,150))

    d = decode(qr_cropping)
    data = d[0].data.decode('ascii')
    data_split = data.split(",")

    # setting  condition that if the ballot paper is invalid, the for loop breaks
    if data_split[0] != "Valid":
      v += 1
      pass
    else:
      # creating another copy of the image
      clone = imag.copy()

      # processing the image for contouring
      processed_imag = process_mark_reg(clone)

      # # obtaining the the markchecked image, name of the candidate chosen, 
      # # whether the vote is spolit or not and the elective position the ballot paper is for
      markchecked, name, countable, position  = checkformark(processed_imag)

      # # adding the values into our dictionary
      add_vals(data_dict, "Status", [data_split[0]])
      add_vals(data_dict, "County", [data_split[1]])
      add_vals(data_dict, "Constituency", [data_split[2]])
      add_vals(data_dict, "Polling Station", [data_split[3]])
      add_vals(data_dict, "Candidate chosen", [name])
      add_vals(data_dict, "Validity", [countable])
      add_vals(data_dict, "Elective position", [position])

      # # adding the images with their bounding boxes drawn into a directory
      if name == "spoilt":
        cv2.imwrite(r"C:\Users\angie\Documents\Moringa School\Final Project - Computer Vision\counted votes\spoilt" +"\\"+"image"+f'{m}'+'.jpg', markchecked)
        s += 1
      else:
        cv2.imwrite(r"C:\Users\angie\Documents\Moringa School\Final Project - Computer Vision\counted votes\countable" +"\\"+"image"+f'{m}'+'.jpg', markchecked)
        c += 1
    m += 1
    

# converting my dictionary into a dataframe
df = pd.DataFrame.from_dict(data_dict)
df.head()
senate = df[df['Elective position'] == "senate"]
wrep = df[df['Elective position'] == "women rep"]
pres = df[df['Elective position'] == "president"]
mp = df[df['Elective position'] == "member of parliament"]
mca = df[df['Elective position'] == "mca"]
gov = df[df['Elective position'] == "governor"]



# # saving my csv file to my directory
df.to_csv(r'C:\Users\angie\Documents\Moringa School\Final Project - Computer Vision\data\draft_data.csv', encoding='utf-8')
senate.to_csv(r'C:\Users\angie\Documents\Moringa School\Final Project - Computer Vision\data\sen_data.csv', encoding='utf-8')
wrep.to_csv(r'C:\Users\angie\Documents\Moringa School\Final Project - Computer Vision\data\womrep_data.csv', encoding='utf-8')
pres.to_csv(r'C:\Users\angie\Documents\Moringa School\Final Project - Computer Vision\data\pres_data.csv', encoding='utf-8')
mp.to_csv(r'C:\Users\angie\Documents\Moringa School\Final Project - Computer Vision\data\parl_data.csv', encoding='utf-8')
mca.to_csv(r'C:\Users\angie\Documents\Moringa School\Final Project - Computer Vision\data\ward_data.csv', encoding='utf-8')
gov.to_csv(r'C:\Users\angie\Documents\Moringa School\Final Project - Computer Vision\data\govn_data.csv', encoding='utf-8')

# manipulating the data
# cleaning function
def clean(df):
    df["Candidate chosen"] = df['Candidate chosen'].str.replace("\n","").str.replace(" \ ", "").str.replace("XX","").str.replace("\\","")
    df['Elective position'] = df['Elective position'].str.replace("\n","").str.replace("=", "").str.replace("pe","")
    df.columns = df.columns.str.replace("Unnamed: 0", "Vote index")

# function for computing results
def compute_res(df):
    result = pd.DataFrame(df.groupby("County")["Candidate chosen"].value_counts())
    result.columns = result.columns.str.replace("Candidate chosen", "count")
    result = result.reset_index("Candidate chosen")
    return result

def plot_results(df,office):
    # plotting the results for the different counties
    l = len(df.index.unique())
    x = 1
    plt.figure(figsize=(16,10))
    plt.subplots_adjust(hspace=0.5)
    plt.suptitle("Election Results", fontsize=18, y=0.95)
    for i in df.index.unique():
        data_f = df[df.index==i]
        plt.subplot(l,l,x)
        plt.title(i + " county results: " + office)
        ax = sns.barplot(x = data_f["Candidate chosen"], y= data_f['count'], palette='prism_r')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment="right")

        x += 1

clean(senate)
sen_res = compute_res(senate)
plot_results(sen_res,"Senate")
plt.show()

clean(wrep)
wom_res = compute_res(wrep)
plot_results(wom_res,"Women Representative")
plt.show()

clean(mp)
parl_res = compute_res(mp)
plot_results(parl_res)
plt.show()

clean(mca)
mca_res = compute_res(mca)
plot_results(mca_res,"MCA")
plt.show()

clean(gov)
gov_res = compute_res(gov)
plot_results(gov_res,"Governor")
plt.show()

clean(pres)
pres_res = pd.DataFrame(pres["Candidate chosen"].value_counts())
plt.title("Presidential county results")
ax = sns.barplot(x = pres_res.index, y= pres_res['Candidate chosen'], palette='prism_r')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment="right")
plt.show()

print("The data has been collected from "+ str(m) + " images.")
print("Your data contains "+ str(v) + " fake ballot papers.")
print("There were "+ str(c) +" votes classified as countable and "+ str(s) +" classified as spoilt.")
print("--- %s seconds ---" % (time.time() - start_time))
    
    




    

    

