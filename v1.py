from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import time
import numpy as np

framesize = 1280, 720

camera = PiCamera()
camera.resolution = (framesize)
camera.framerate = 30

rawCapture = PiRGBArray(camera, size=(framesize))
display_window = cv2.namedWindow("Version 1")
time.sleep(1)

point_ul =   150,700
point_o  =   600,100
point_ur =  1150,700

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
    img = cv2.circle(img,(point_ul),10,(0,0,255)) #Punkt unten Links
    img = cv2.circle(img,(point_o) ,10,(0,0,255)) #Punkt oben
    img = cv2.circle(img,(point_ur),10,(0,0,255)) #Punkt unten rechts
    return img

#-------------------------------------------#

def do_canny(frame):
    # Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
    gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    # Applies a 5x5 gaussian blur with deviation of 0 to frame - not mandatory since Canny will do this for us
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    # Applies Canny edge detector with minVal of 50 and maxVal of 150
    canny = cv.Canny(blur, 50, 150)
    return canny


def do_segment(frame):
    # Since an image is a multi-directional array containing the relative intensities of each pixel in the image, we can use frame.shape to return a tuple: [number of rows, number of columns, number of channels] of the dimensions of the frame
    # frame.shape[0] give us the number of rows of pixels the frame has. Since height begins from 0 at the top, the y-coordinate of the bottom of the frame is its height
    height = frame.shape[0]
    # Creates a triangular polygon for the mask defined by three (x, y) coordinates
    triangle = np.array([[   #RIO Triangle
        (point_ul),
        (point_o ),
        (point_ur),]], np.int32)

    # Creates an image filled with zero intensities with the same dimensions as the frame
    mask = np.zeros_like(frame)
    # Allows the mask to be filled with values of 1 and the other areas to be filled with values of 0
    cv.fillPoly(mask, triangle, 255)
    # A bitwise and operation between the mask and frame keeps only the triangular area of the frame
    segment = cv.bitwise_and(frame, mask)
    return segment


def calculate_lines(frame, lines):
    # Empty arrays to store the coordinates of the left and right lines
    left = []
    right = []
    # Loops through every detected line
    for line in lines:
        # Reshapes line from 2D array to 1D array
        x1, y1, x2, y2 = line.reshape(4)
        # Fits a linear polynomial to the x and y coordinates and returns a vector of coefficients which describe the slope and y-intercept
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        y_intercept = parameters[1]
        # If slope is negative, the line is to the left of the lane, and otherwise, the line is to the right of the lane
        if slope < 0:
            left.append((slope, y_intercept))
        else:
            right.append((slope, y_intercept))
    # Averages out all the values for left and right into a single slope and y-intercept value for each line
    left_avg = np.average(left, axis = 0)
    right_avg = np.average(right, axis = 0)
    # Calculates the x1, y1, x2, y2 coordinates for the left and right lines
    left_line = calculate_coordinates(frame, left_avg)
    right_line = calculate_coordinates(frame, right_avg)
    return np.array([left_line, right_line])


def calculate_coordinates(frame, parameters):
    slope, intercept = parameters
    # Sets initial y-coordinate as height from top down (bottom of the frame)
    y1 = frame.shape[0]
    # Sets final y-coordinate as 150 above the bottom of the frame
    y2 = int(y1 - 150)
    # Sets initial x-coordinate as (y1 - b) / m since y1 = mx1 + b
    x1 = int((y1 - intercept) / slope)
    # Sets final x-coordinate as (y2 - b) / m since y2 = mx2 + b
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def visualize_lines(frame, lines):
    # Creates an image filled with zero intensities with the same dimensions as the frame
    lines_visualize = np.zeros_like(frame)
    # Checks if any lines are detected
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            # Draws lines between two coordinates with green color and 5 thickness
            cv.line(lines_visualize, (x1, y1), (x2, y2), (0, 255, 0), 5)
    return lines_visualize




for frame in camera.capture_continuous(rawCapture, format="rgb", use_video_port=True):

    image1 = frame.array
    image1 = drawpoints(image1)
#   image1 = undistort(image1)

    canny = do_canny(image1)
    cv.imshow("canny", canny)

    # plt.imshow(frame)
    # plt.show()
    segment = do_segment(canny)
    hough = cv.HoughLinesP(segment, 2, np.pi / 180, 100, np.array([]), minLineLength = 100, maxLineGap = 50)
    # Averages multiple detected lines from hough into one line for left border of lane and one line for right border of lane
    lines = calculate_lines(image1, hough)
    # Visualizes the lines
    lines_visualize = visualize_lines(image1, lines)
    cv.imshow("hough", lines_visualize)
    # Overlays lines on frame by taking their weighted sums and adding an arbitrary scalar value of 1 as the gamma argument
    output = cv.addWeighted(image1, 0.9, lines_visualize, 1, 1)
    # Opens a new window and displays the output frame
    cv.imshow("output", output)
    # Frames are read by intervals of 10 milliseconds. The programs breaks out of the while loop when the user presses the 'q' key
    if cv.waitKey(10) & 0xFF == ord('q'):
        break

    rawCapture.truncate(0)
    
    if key == 27:
        camera.close()
        cv2.destroyAllWindows()
        break
