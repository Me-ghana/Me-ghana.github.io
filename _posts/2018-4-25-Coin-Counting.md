---
layout: post
title: Coin Counting with the JeVois in Python
---

This tutorial will help explain how to identify U.S. coins from a video stream with Python and OpenCV.  

I found the exercise of identifying U.S. coins to be a good introduction to some of the basic functions in OpenCV and some important computer vision algorithms.  The final code uses blob detection, but we'll explore other algorhtms such as Hough Circles and the Watershed algorithm, along the way.  While this post explains how to use OpenCV with the [JeVois](http://jevois.org/), a smart machine vision camera, to count U.S. coins, you can use any video stream that is read image by image.  At the very least, you'll need to have OpenCV, numpy, and an image of some coins.

My first goal was to see how well I could identify the four most common U.S. coins, the penny, nickel, dime, and quater, by detecting just the circle sizes and coin colors.  We'll have five main steps:
1. Read in our image
2. Pre-process our image 
3. Identify the coins in the image as our ROIs
4. Create a calibration file with current values for coin heuristics
5. Identify each ROI as a particular coin using calibration data
5. Compute the total value

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
Here's an image I read in on my video stream.  I'm using a resolution of 640 X 480.  We'll see how it changes as we go through each step.

<p align="center">
  <img src= "https://raw.githubusercontent.com/Me-ghana/Coin-Counter/master/CoinImages/CoinStep1.png" width = "50em">
</p>

### Step 2: Pre-process our image
In Step 2, we we get our first taste of OpenCV algorithms. This pre-processing was adapted from the [JeVois Python Dice Tutorial](http://jevois.org/tutorials/ProgrammerPythonDice.html).  

Let's continue our above code:

``` python
    def __init__(self):
        # Step 2
        self.kernel = np.ones((5,5), np.uint8)
        
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
        
        # Apply automatic threshold
        ret, grayImage = cv2.threshold(grayImage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Background area
        # grayImage = cv2.dilate(grayImage, self.kernel, iterations = 1) #self.morphBNo2)
        invBack2 = 255 - grayImage
        
```
**Step 2A** 
In the first step of pre-processing, we use the OpenCV function convert color function to make our image gray-scale.  [Here](https://docs.opencv.org/3.2.0/de/d25/imgproc_color_conversions.html) is a list of all the possible color converstions - this will come in helpful later if you want to use a different model, like HSV, for identifying coin types. In OpenCV, the channels are in Blue, Green, Red (BGR) order, nor the more common RGB. 

**Step 2B** 
The .shape function will return only two values for a gray-scale image, the number of rows (height) and columns (width).  If you have a color image, .shape will return an additional channel value.

**Step 2C**
In order to remove noise from our image, we use a 5 X 5 kernel to apply a Gaussian Blur to the image.  This smoothing process recomputes a pixel's value by taking a weighted sum of the neighboring pixels.  Since we are using the Gaussian blur, as spatially more distant pixels are weighted less.  If you are not familiar with kernel convolution, [here](https://docs.opencv.org/2.4/doc/tutorials/imgproc/gausian_median_blur_bilateral_filter/gausian_median_blur_bilateral_filter.html) is a tutorial on OpenCV's image smoothing operations.

**Step 2D**

