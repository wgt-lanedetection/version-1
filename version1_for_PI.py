from picamera.array import PiRGBArray
from picamera import PiCamera
import os
import numpy as np
import math
import cv2
import time


#Set Frame Size
#frame_x = 416
#frame_y = 256
#---
frame_x = 640
frame_y = 368
#---

# Parameters for the Camera
camera = PiCamera()
camera.resolution = (frame_x,frame_y)
camera.framerate = 15
camera.iso = 800
camera.saturation = 35
camera.sharpness = 10
camera.video_stabilization = True
rawCapture = PiRGBArray(camera, size = (frame_x,frame_y))
#rawCapture = PiRGBArray(camera, size = (480,272))
cv2.namedWindow("Version 1")
cv2.setWindowProperty("Version 1", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
#cv2.resizeWindow("Version 1", 960, 544)
time.sleep(1)

#Parameters for the ROI
point_ul =    int(0) , int(frame_y)                     #unten links
point_ur =    int(frame_x), int(frame_y)                #unten rechts
point_or  =   int(frame_x * 0.65) , int(frame_y *0.6)   #oben  rechts
point_ol  =   int(frame_x * 0.35) , int(frame_y * 0.6)  #oben links

# Parameter for Text
font              = cv2.FONT_HERSHEY_SIMPLEX
LeftCornerText    = (10,20)
fontScale         = 0.5
fontColor         = (255,255,255)
lineType          = 1

#Parameters for Camera Calibration
#def undistort(img):
#    cam_mtx = np.array([    
#                         [168.12940519,   0,           162.07114665],
#                         [  0.0,         168.27516637 ,100.44213185],
#                         [  0,            0,           1           ]
#                         ])
#
#    cam_dst = np.array([-0.34068385,  0.14733747,  0.00066994,  0.00060006, -0.03411863])
#    img = cv2.undistort(img, cam_mtx, cam_dst, None, cam_mtx)
#    return img

def drawpoints(img):
    img = cv2.circle(img,(point_ul) ,5,(0,0,255)) #Punkt unten Links
    img = cv2.circle(img,(point_ol) ,5,(0,0,255),-1) #Punkt oben Links
    img = cv2.circle(img,(point_or) ,5,(0,0,255),-1) #Punkt oben Rechts
    img = cv2.circle(img,(point_ur) ,5,(0,0,255)) #Punkt unten rechts
    return img

def wirteText(img, c_fps):
    text = "fps:" + str(int(c_fps))
    cv2.putText(img, text , 
        LeftCornerText, 
        font, 
        fontScale,
        fontColor,
        lineType)
    return img

#-------------------------------------------------------------------------------------#

#PIPLINE STARTS HERE

#-------------------------------------------#

# Color Filtering
def color_filter(image):
    #convert to HLS to mask based on HLS
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    lower = np.array([0,190,0])
    upper = np.array([255,255,255])
    
    yellower = np.array([10,0,90])
    yelupper = np.array([50,255,255])
    
    yellowmask = cv2.inRange(hls, yellower, yelupper)    
    whitemask = cv2.inRange(hls, lower, upper)
    
    mask = cv2.bitwise_or(yellowmask, whitemask)  
    masked = cv2.bitwise_and(image, image, mask = mask)    
    
    return masked


#-------------------------------------------#

# REGION OF INTEREST 
def roi(img):
    x = int(img.shape[1])
    y = int(img.shape[0])
   
    shape = np.array([
                      [point_ul], 
                      [point_ur], 
                      [point_or], 
                      [point_ol]
                      ])

#    shape = np.array([
#                      [int(0), int(y)], 
#                      [int(x), int(y)], 
#                      [int(0.65*x), int(0.6*y)], 
#                      [int(0.35*x), int(0.6*y)]
#                      ])

    #define a numpy array with the dimensions of img, but comprised of zeros
    mask = np.zeros_like(img)

    #Uses 3 channels or 1 channel for color depending on input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #creates a polygon with the mask color
    cv2.fillPoly(mask, np.int32([shape]), ignore_mask_color)

    #returns the image only where the mask pixels are not zero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

#-------------------------------------------#
#
#Edge detection
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def canny(img):
    return cv2.Canny(grayscale(img), 50, 120)

#-------------------------------------------#

# Finding Lines
rightSlope, leftSlope, rightIntercept, leftIntercept = [],[],[],[]
def draw_lines(img, lines, thickness=5):
    global rightSlope, leftSlope, rightIntercept, leftIntercept
    rightColor=[0,0,255]
    leftColor=[255,0,0]
    
    #this is used to filter out the outlying lines that can affect the average
    #We then use the slope we determined to find the y-intercept of the filtered lines by solving for b in y=mx+b
    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = (y1-y2)/(x1-x2)
            if slope > 0.3:
                if x1 > 500 :
                    yintercept = y2 - (slope*x2)                    
                    rightSlope.append(slope)
                    rightIntercept.append(yintercept)
                else: None                
            elif slope < -0.3:
                if x1 < 600:
                    yintercept = y2 - (slope*x2)                    
                    leftSlope.append(slope)
                    leftIntercept.append(yintercept)    
                    
                    
    #We use slicing operators and np.mean() to find the averages of the 30 previous frames
    #This makes the lines more stable, and less likely to shift rapidly
    leftavgSlope = np.mean(leftSlope[-30:])
    leftavgIntercept = np.mean(leftIntercept[-30:])
    
    rightavgSlope = np.mean(rightSlope[-30:])
    rightavgIntercept = np.mean(rightIntercept[-30:])
    
    
    #Here we plot the lines and the shape of the lane using the average slope and intercepts
    try:
        left_line_x1 = int((0.65*img.shape[0] - leftavgIntercept)/leftavgSlope)
        left_line_x2 = int((img.shape[0] - leftavgIntercept)/leftavgSlope)
    
        right_line_x1 = int((0.65*img.shape[0] - rightavgIntercept)/rightavgSlope)
        right_line_x2 = int((img.shape[0] - rightavgIntercept)/rightavgSlope)

        pts = np.array([[left_line_x1, int(0.65*img.shape[0])],[left_line_x2, int(img.shape[0])],[right_line_x2, int(img.shape[0])],[right_line_x1, int(0.65*img.shape[0])]], np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.fillPoly(img,[pts],(0,255,0))      
        
        
        cv2.line(img, (left_line_x1, int(0.65*img.shape[0])), (left_line_x2, int(img.shape[0])), leftColor, 10)
        cv2.line(img, (right_line_x1, int(0.65*img.shape[0])), (right_line_x2, int(img.shape[0])), rightColor, 10)
    except ValueError:
            #I keep getting errors for some reason, so I put this here. Idk if the error still persists.
        pass
       
                
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def linedetect(img):
    return hough_lines(img, 1, np.pi/180, 10, 20, 100)

#-------------------------------------------#

# Overlaying the Image and the Lines
def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def weightSum(input_set):
    img = list(input_set)
    return cv2.addWeighted(img[0], 1, img[1], 0.8, 0)


#-------------------------------------------#

# Camera Stream
for frame in camera.capture_continuous(rawCapture, format="rgb", use_video_port=True):

   image1 = frame.array
   start = time.time()            #start timer for fps calculation
   image1 = drawpoints(image1)
#   image1 = undistort(image1)
#   cv2.imshow("Version 1", image1)

   interest  = roi(image1)
   filterimg = color_filter(interest)
   canny = cv2.Canny(grayscale(filterimg), 50, 120)
   myline = hough_lines(canny, 1, np.pi/180, 10, 20, 5)
   weighted_img = cv2.addWeighted(myline, 0.5, image1, 0.8, 0)
   end = time.time()               #end timer for fps calculation
   c_fps = (1 / (end - start))      #calculate fps in second
   final = wirteText(weighted_img, c_fps)
   final = cv2.resize(final, (int(frame_x *2), int(frame_y*2)))
   cv2.imshow("Version 1", final)
   
   key = cv2.waitKey(1)

   rawCapture.truncate(0)
   if key == 27:
      camera.close()
      cv2.destroyAllWindows()
      break
