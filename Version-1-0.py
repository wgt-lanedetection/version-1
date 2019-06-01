from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import time
import numpy as np

frame_width = 1280
frame_height = 720

camera = PiCamera()
camera.resolution = (frame_width, frame_height)
camera.framerate = 30

rawCapture = PiRGBArray(camera, size=(frame_width, frame_height))
display_window = cv2.namedWindow("Version 1")
time.sleep(1)

point_ul =   150,700
point_o  =   600,100
point_ur =  1150,700

cam_mtx = np.array([    
                    [168.12940519,   0,           162.07114665],
                    [  0.0,         168.27516637 ,100.44213185],
                    [  0,            0,           1           ]
                  ])

cam_dst = np.array([-0.34068385,  0.14733747,  0.00066994,  0.00060006, -0.03411863])

# Undisttort Image
def undistort(img):
    img = cv2.undistort(img, cam_mtx, cam_dst, None, cam_mtx)
    return img

# Draw Points to mark the RIO
def drawpoints(img):
    img = cv2.circle(img,(point_ul),10,(0,0,255)) #Punkt unten Links
    img = cv2.circle(img,(point_o) ,10,(0,0,255)) #Punkt oben
    img = cv2.circle(img,(point_ur),10,(0,0,255)) #Punkt unten rechts
    return img

def make_points(image, line):
    slope, intercept = line
    y1 = int(image.shape[0])    # bottom of the image
    y2 = int(y1*3/5)            # slightly lower than the middle
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return [[x1, y1, x2, y2]]

def average_slope_intercept(image, lines):
    left_fit    = []
    right_fit   = []
    if lines is None:
        return None
    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1,x2), (y1,y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:       # y is reversed in image
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
    # add more weight to longer lines
    left_fit_average  = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line  = make_points(image, left_fit_average)
    right_line = make_points(image, right_fit_average)
    averaged_lines = [left_line, right_line]
    return averaged_lines

def canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = 5
    blur = cv2.GaussianBlur(gray,(kernel, kernel),0)
    canny = cv2.Canny(gray, 50, 150)
    return canny

def display_lines(img,lines):
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
    return line_image

def region_of_interest(canny):  #define RIO
    height = canny.shape[0]
    width = canny.shape[1]
    mask = np.zeros_like(canny)

    triangle = np.array([[   #RIO Triangle
    (point_ul),
    (point_o ),
    (point_ur),]], np.int32)

    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(canny, mask)
    return masked_image



for frame in camera.capture_continuous(rawCapture, format="rgb", use_video_port=True):

    image1 = frame.array
    image1 = drawpoints(image1)
#   image1 = undistort(image1)

    canny_image = canny(image1)  #Canny Filter on single frames
    cropped_canny = region_of_interest(canny_image) #Canny Filter with RIO: Street + RIO
    lines = cv2.HoughLinesP(cropped_canny, 2, np.pi/180, 100, np.array([]), minLineLength=40,maxLineGap=5)
    averaged_lines = average_slope_intercept(image1, lines) #Draw blue lines
    line_image = display_lines(image1, averaged_lines)
    combo_image = cv2.addWeighted(image1, 0.8, line_image, 1, 1) #Lines + Street
    cv2.imshow("Version 1", combo_image)
    key = cv2.waitKey(1)
    #print(height)
    rawCapture.truncate(0)
    if key == 27:
        camera.close()
        cv2.destroyAllWindows()
        break
