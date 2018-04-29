---
layout: post
title: Coin Counting with the JeVois in Python
---

This tutorial will help explain how to identify U.S. coins from a video stream with Python and OpenCV.  

I found the exercise of identifying U.S. coins to be a good introduction to some of the basic functions in OpenCV and some important computer vision algorithms.  The final code uses blob detection, but we'll explore other algorithms such as Hough Circles and the Watershed algorithm along the way.  While this post explains how to use OpenCV with the [JeVois](http://jevois.org/), a smart machine vision camera, you can use any video stream you can extract images from.  At the very least, you'll need to have OpenCV, numpy, and an image of some coins.

My first goal was to see how well I could identify the four most common U.S. coins, the penny, nickel, dime, and quater, by detecting just the circle sizes and coin colors.  We'll have five main steps:
1. Read in our image
2. Pre-process our image 
3. Identify coins with blob detection
4. Create a calibration file with values for coin heuristics
5. Identify each ROI as a particular coin using calibration data
6. Compute the total value

If you run into any issues, scroll down to check out some tourbleshooting tips!

### Step 1: Read in our image
Step 1 is pretty simple if you are using the JeVois. If you want to learn more about the built-in JeVois functions, check out the helpful [JeVois Programming Tutorials](http://jevois.org/tutorials/ProgrammerTutorials.html).  If you are using a different video stream, read in a single image and store it as "img".

Here's the following code for reading in our image.  

``` python
    def process(self, inframe, outframe):
        # Get the next camera image
        # May block until it is captured
        # Convert it to OpenCV BGR (for color output):
        img = inframe.getCvBGR()
```
Here's an image I read in on my video stream.  I'm using a resolution of 640X480.  We'll see how this image changes as we go through each step.

<p align="center">
  <img src= "https://raw.githubusercontent.com/Me-ghana/Coin-Counter/master/CoinImages/CoinStep1.png" width = "450">
</p>

### Step 2: Pre-process our image
In Step 2, we get our first taste of OpenCV algorithms. This pre-processing was adapted from the [JeVois Python Dice Tutorial](http://jevois.org/tutorials/ProgrammerPythonDice.html).  

Let's continue our above code:

``` python
    def process(self, inframe, outframe):
        # Step 1A 
        # Get the next camera image
        # May block until it is captured
        # Convert it to OpenCV BGR (for color output):
        img = inframe.getCvBGR()

        # Step 2A
        # Also convert it to grayscale for processing:
        grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Step 2B
        # Get image width, height:
        height, width = grayImage.shape

        # Step 2C
        # Filter noise
        grayImage = cv2.GaussianBlur(grayImage, (5, 5), 0, 0)
        
	# Step 2D
        # Apply automatic threshold
        ret, grayImage = cv2.threshold(grayImage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        
```
**Step 2A** 
In the first step of pre-processing, we use the OpenCV function convert color function to make our image gray-scale.  [Here](https://docs.opencv.org/3.2.0/de/d25/imgproc_color_conversions.html) is a list of all the possible color converstions - this will come in helpful later if you want to use a different model, like HSV, for identifying coin types. In OpenCV, the channels are in Blue, Green, Red (BGR) order, not the more common RGB. We'll talk more about HSV vs BGR later on. 

**Step 2B** 
The .shape function will return only two values for a gray-scale image, the number of rows (height) and columns (width).  If you have a color image, .shape will return an additional channel value.

**Step 2C**
In order to remove noise from our image, we use a 5 X 5 kernel to apply a Gaussian Blur to the image.  This smoothing process recomputes a pixel's value by taking a weighted sum of the neighboring pixels.  Since we are using the Gaussian blur, spatially more distant pixels are weighted less.  If you are not familiar with kernel convolution, [here](https://docs.opencv.org/2.4/doc/tutorials/imgproc/gausian_median_blur_bilateral_filter/gausian_median_blur_bilateral_filter.html) is a tutorial on OpenCV's image smoothing operations.

**Step 2D**
In this step, we make our gray-scale image black and white by choosing a threshold using Otsu's Method.  This algorithm finds the optimal threshold value in a bimodal image. In this example, we are using a white background and the coins are darker.  We want to coins to be black and the background to be white, so we use the inverse method provided by OpenCV.  This is because the Simple Blob Detector typically detects dark spots as blobs.  (Later on, we can add a simple function to determine the background color to see if the inverse method should be used or not automatically.)

Here's what our image looks like after thresholding:
<p align="center">
  <img src= "https://raw.githubusercontent.com/Me-ghana/Coin-Counter/master/CoinImages/Coins2D.png" width = "450">
</p>

### Step 3: Identify coins with blob detection
We'll instantiate a simple blob detector in our constructor in order to identify coins in our image.  The simple blob detector algorithm works by:
   1. thresholding the image into not just one, but several binary images, and then extracting connected pixels in each binary image with findContours, and finally merging blobs that have centers close to one another.  You can specify specific thresholds and filters to have increased control over how blobs should be extracted. 

``` python
# Step 3A
## Constructor
    def __init__(self):
        # Instantiate a circular blob detector:
        params = cv2.SimpleBlobDetector_Params()
        params.filterByCircularity = True
        params.filterByArea = True
        params.minArea = 200.0
        self.detector = cv2.SimpleBlobDetector_create(params)
        
    def process(self, inframe, outframe):
        # Step 1A 
        # Step 2A
        # Step 2B
        # Step 2C
        
        # Step 3B
        # Blob detection
        keypoints = self.detector.detect(invBack2)
        nrOfBlobs = len(keypoints)
        
        # Step 3C
        # Draw keypoints
        im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (255, 0, 0),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

```
**Step 3A & 3B**
Instantiate a simple blob detector in the constructor and filter by circularity. Set a minimum area to limit inclusion of artifacts. Use the gray-scaled and de-noised image to detect the keypoints, and save the number of blobs found.

**Step 3C**
Draw the keypoints as a blue line on the original image.  You can see that we have successfully identified each coin. 

<p align="center">
  <img src= "https://raw.githubusercontent.com/Me-ghana/Coin-Counter/master/CoinImages/Coins3.png" width = "450">
</p>

While we chose to use the simple blob detector, there are several other algorithms that we can have used instead.  If you're not interested in hearing about the alternatives, skip ahead to Step 4.  

Alternative #1: Find Contours

An alternative is to use OpenCV's find contour algorithm to detect the coin outlines.  Instead of using the blob detector to apply multiple thresholds, we can automatically apply just one threshold with Otsu's Binarization, which finds the optimal thershold value in a bimodal distribution.  From there, we can find and draw the minimum enclosing circle.
 
 ```python
 # Apply automatic threshold
ret, grayImage2 = cv2.threshold(grayImage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Find contours    
im2, contours, hierarchy = cv2.findContours(grayImage2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

# Draw tightest enclosing circle for each contour
for c in contours:
	(x,y),radius = cv2.minEnclosingCircle(c)
	center = (int(x),int(y))
	radius = int(radius)
	cv2.circle(img,center,radius,(255, 0, 0),1)
```

However, we can see this does not give us as tight as a circle:
<p align="center">
  <img src= "https://raw.githubusercontent.com/Me-ghana/Coin-Counter/master/CoinImages/Coins4.png" width = "450">
</p>

Alternative #2: Hough Circles
Alternative #2: The Watershed Algorithm



### Step 4: Create a calibration file with current values for coin heuristics
In this project, we are attempting to use basic image processing to identify U.S. Coins.  Two simple heuristics we can use are coin size and color.  First, let's check to see if it is possible to use these heuristics to tell U.S. Coins apart.



Since the size and the color will change depending on the distance the camera is mounted from the coins and the lighting conditions, we'll first create a calibration program that should always be run prior to the actual coin counting algorithm.    






 HSV is often used in computer vision for color based operations since it maps specific colors to a hue channel, while keeping the saturation (how white the color is) and the value (how dark the channel is) in two other channels.  For example, in HSV, all types of red color, regardless of illumination, will have the same hue value.  
