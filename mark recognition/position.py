import cv2
import pickle

# importing and resizing the image
im = cv2.imread('new_ballot-papers/prnairobidagoballot.jpg')
imgwidth = 860
imgheight = 950
img = cv2.resize(im,(imgwidth,imgheight))


# dimension for bounding boxes
width, height = 105, 90

try:
    with open('ballotboxposition','rb') as f:
        positionlist= pickle.load(f)
except:
    positionlist =[]



# function to create and delete the boxes
def mouseclick(events,x,y,flags,params):
    if events == cv2.EVENT_LBUTTONDOWN:
        positionlist.append((x,y))
    if events == cv2.EVENT_RBUTTONDOWN:
        for i, pos in enumerate(positionlist):
            x1,y1 = pos
            if x1 < x < x1 + width and y1 < y < y1 + height:
                positionlist.pop(i)

    with open('ballotboxposition','wb') as f:
        pickle.dump(positionlist,f)


while True:
    img = cv2.imread('new_ballot-papers/prnairobidagoballot.jpg')
    imgwidth = 860
    imgheight = 950
    img = cv2.resize(im,(imgwidth,imgheight))

    for pos in positionlist:
        #cv2.rectangle(img,pos)
        cv2.rectangle(img,pos,(pos[0]+width,pos[1]+height),(255,0,255),2)


    cv2.imshow('image',img)
    cv2.setMouseCallback('image',mouseclick)
    cv2.waitKey(1)