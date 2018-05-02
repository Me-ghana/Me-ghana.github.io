---
layout: post
title: Coin Counting with the JeVois in Python - Programmer Tutorial
---

This tutorial will help explain how to identify U.S. coins from a video stream with Python and OpenCV, like in the video below. 

<div align="center">
  <a href="https://www.youtube.com/watch?v=R4LO0sgfBmU"><img src="https://img.youtube.com/vi/R4LO0sgfBmU/0.jpg" ></a>
	<div align = "center"><figcaption>Click image to watch video</figcaption></div>
</div>

I found the exercise of identifying U.S. coins to be a good introduction to some of the basic functions in OpenCV and some important computer vision algorithms.  The final code uses blob detection, but we'll explore other algorithms such as Hough Circles and the Watershed algorithm along the way.  While this post explains how to use OpenCV with the [JeVois](http://jevois.org/), a smart machine vision camera, you can use any video stream you can extract images from.  At the very least, you'll need to have OpenCV, numpy, and an image of some coins.

My first goal was to see how well I could identify the four most common U.S. coins, the penny, nickel, dime, and quater, by detecting just the circle sizes and coin colors.  We'll have nine steps:
1. <a href = "https://me-ghana.github.io/Coin-Counting/#P1">Read in our image</a>
2. <a href = "https://me-ghana.github.io/Coin-Counting/#P2">Pre-process our image</a>
3. <a href = "https://me-ghana.github.io/Coin-Counting/#P3">Identify coins with blob detection</a>
4. <a href = "https://me-ghana.github.io/Coin-Counting/#P4">Determine the heuristics and color space</a> 
5. <a href = "https://me-ghana.github.io/Coin-Counting/#P5">Write calibration program body (optional)</a>
6. <a href = "https://me-ghana.github.io/Coin-Counting/#P6">Write calibration program helper functions (optional)</a>
7. <a href = "https://me-ghana.github.io/Coin-Counting/#P7">Write coin detection program helper functions</a>
8. <a href = "https://me-ghana.github.io/Coin-Counting/#P8">Write coin detection program program body</a>
9. <a href = "https://me-ghana.github.io/Coin-Counting/#P9">Count coins using the calibration and coin detection programs!</a>

The big picture is we will be creating two programs, a calibration program and a coin counting program.  The calibration program will be run first and generate files with data about each coin type.  The main program will then open these files and use this data to identify coin types, and will then calculate the total value of all the coins.  (Users can skip the calibration step, and write in the parameters manually in the coin counting program.)

You can follow along by [downloading the code](https://github.com/Me-ghana/Coin-Counter) for both the calibration and counting program. If you run into any issues, scroll down to check out some <a href = "https://me-ghana.github.io/Coin-Counting/#PT">touble-shooting tips!</a> If you want to skip ahead and see how this project turned out, check out the <a href = "https://me-ghana.github.io/Coin-Counting/#PTA">take-aways</a>.
<div id = "P1"> </div>
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
<div id = "P2"> </div>
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
        grayImage = cv2.GaussianBlur(grayImage, (self.kernelWidth, self.kernelWidth), 0, 0)
        
	# Step 2D
        # Apply automatic threshold
        ret, threshImage = cv2.threshold(grayImage, 0, 255, self.threshWhiteBackground)
        # Determine if the background is white or black, and then apply appropriate threshold
        whitePixels = cv2.countNonZero(threshImage)
        blackPixels = height*width - whitePixels
        
```
**Step 2A** 
In the first step of pre-processing, we use the OpenCV function convert color function to make our image gray-scale.  [Here](https://docs.opencv.org/3.2.0/de/d25/imgproc_color_conversions.html) is a list of all the possible color converstions - this will come in helpful later if you want to use a different model, like HSV, for identifying coin types. In OpenCV, the channels are in Blue, Green, Red (BGR) order, not the more common RGB. We'll talk more about HSV vs BGR later on. 

**Step 2B** 
The .shape function will return only two values for a gray-scale image, the number of rows (height) and columns (width).  If you have a color image, .shape will return an additional channel value.

**Step 2C**
In order to remove noise from our image, we use a 5 X 5 kernel to apply a Gaussian Blur to the image. The size is specified in the constructor as a modifiable parameter, which we will see later. This smoothing process recomputes a pixel's value by taking a weighted sum of the neighboring pixels.  Since we are using the Gaussian blur, spatially more distant pixels are weighted less.  If you are not familiar with kernel convolution, [here](https://docs.opencv.org/2.4/doc/tutorials/imgproc/gausian_median_blur_bilateral_filter/gausian_median_blur_bilateral_filter.html) is a tutorial on OpenCV's image smoothing operations.

**Step 2D**
In this step, we make our gray-scale image black and white by choosing a threshold using Otsu's Method. (This is set as another parameter in the constructor.)  This algorithm finds the optimal threshold value in a bimodal image. We compute wether we have more white or black pixels, to determiene what the background is.  If we have a black background, we use the thresholding inverse.  In this example, we are using a white background and the coins are darker.  We want the coins to be black and the background to be white.  This is because the simple blob detector typically detects dark spots as blobs.  We set the threshold method as a parameter, so if you choose to use another circle detection method that detects connected white pixels as circles, the thresholding can be easily flipped.  

Here's what our image looks like after thresholding:
<p align="center">
  <img src= "https://raw.githubusercontent.com/Me-ghana/Coin-Counter/master/CoinImages/Coins2D.png" width = "450">
</p>
<div id = "P3"> </div>
### Step 3: Identify coins with blob detection
We'll instantiate a simple blob detector in our constructor in order to identify coins in our image.  The simple blob detector algorithm works by:
   1. Thresholding the image into not just one, but several binary images
   2. Extracting connected pixels in each binary image with findContours 
   3. Finally merging blobs that have centers close to one another. You can specify threshold and filter values to have increased control over how blobs should be extracted. 

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

          # Parameters: pre-processing 
          self.threshWhiteBackground = cv2.THRESH_BINARY + cv2.THRESH_OTSU           # threshold type 
          self.threshBlackBackground = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU       # threshold type 
          self.kernelWidth = 5                                                       # width for blur
          self.kernelHeight = 5                                                      # height for blur

          # Parameters: instructions and marker placement 
          self.xVal = 80                               # x coordinate for first marker
          self.yVal = 300                              # y coordinate for first marker
          self.xDelta = 140                            # The distance between coins 
          self.yDelta = -100                           # The distance between coins and above text
        
    def process(self, inframe, outframe):
        # Step 1A 
        # Step 2A
        # Step 2B
        # Step 2C
        # Step 2D
	
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
Draw the keypoints as a blue line on the original image.  You can see that we have successfully identified each coin region. 

<p align="center">
  <img src= "https://raw.githubusercontent.com/Me-ghana/Coin-Counter/master/CoinImages/Coins3.png" width = "450">
</p>

While we chose to use the simple blob detector, there are several other algorithms that we can have used instead.  If you're not interested in hearing about the alternatives, skip ahead to Step 4.  

###### Alternative #1: Find Contours
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

###### Alternative #2: Hough Circles
Hough Circles is a specialization of the Hough Transform, and is a popular choice for detecting circles in images.  OpenCV uses the Hough Gradient Method for detection.  It is similar to the Hough Line Transform OpenCV operation.  The algorithm is explained [here](https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/hough_lines/hough_lines.html) and a python tutorial for Hough Circles is provided by OpenCV [here](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghcircles/py_houghcircles.html).

We can see that we get better performance with Hough Circles than with the find contours algorithm.  You can choose to replace the circle detection code with Hough Circles if you want.  Here is the video of a final coin counting algorithm that uses Hough Circles. 

<div align="center">
  <a href="https://www.youtube.com/watch?v=lPb4vpTNWcI"><img src="https://img.youtube.com/vi/lPb4vpTNWcI/0.jpg" alt="IMAGE ALT TEXT"></a>
	<div align = "center"><figcaption>Click image watch video</figcaption></div>
</div>

As you can see, it does a pretty good job and is even able to detect coins that are overlapping.  In fact, if you watch the final video using blob detection at the top of this page, you'll notice that at 00:31 seconds, I place a quarter too closely to another coin, and have to move it away for it to be detected.  However, the size of the circles are undulating.  This may be due to the fan on the JeVois, which causes vibrations, which Hough Circles may be more sensitive to than other methods.  If we compare the final coin video with Simple Blob detection, we can see the coin outline is much steadier. If you using static images, or have a very steady video stream, Hough Circles is a great option. 

###### Alternative #2: The Watershed Algorithm
The watershed algorithm is an intuitive method to identify circles.  The idea is that an image can be treated like a topographic map, with valleys separating low points.  If  the center of each coin is the low point in the map, we can fill the valley of each lowpoint with a unique color of water.  The boundaries of each coin are indicated where different water colors merge.  

<p align="center">
	<img src= "https://raw.githubusercontent.com/Me-ghana/Coin-Counter/master/CoinImages/CoinsAltWater.png" width = "450"><div align = "center"><figcaption>Watershed Boundaries</figcaption></div>
</p>
<p align="center">
	<img src= "https://raw.githubusercontent.com/Me-ghana/Coin-Counter/master/CoinImages/CoinsAltWater2.png" width = "450"><div align = "center"><figcaption>Final Circles</figcaption></div>
</p>

This is a great way to identify overlapping regions.  However, it  doesn't give us the tightest circles we've seen so far. This implemetnation of the watershed algorithm relies on a series of dilations and erosions to identify the low points, and this also reduces the accuracy of the final circle.    
<div id = "P4"> </div>
### Step 4: Determine the heuristics and color space
**Note: You do not need to do this step yourself if you merely want to use the algorithm.  However, if you're interested in finding your own heuristics, you can read this and follow the method I outline below**

In this project, we are attempting to use basic image processing to identify U.S. Coins.  Two simple heuristics we can use are coin size and color.  First, let's check to see if it is possible to use these heuristics to tell U.S. Coins apart.

**Step 4A**
I wanted to create probability distrbiutions for coin color and size.  The most similar in size is the penny and the dime. If we can tell these two coins apart with a combination of size and color heuristics, we can be fairly confident that we can tell the nickel and quarter apart as well.

**Step 4B**
I went to the bank and got about 400 of each type of coin. Using the JeVois, I used the "screen capture" to capture images of coins in the video screen.  I imaged about 30 coins at a time, and used the simple blob detector method to extract each coin area.  This method had a 96% accuracy of correctly finding the coin boundaries.  (Sometimes coins near the edge of the camera, and coins that I had not completely separated, were not identified correctly).  I recorded the radius, R, G, and B values for each coin.

<p align = "center">
<img src= "https://raw.githubusercontent.com/Me-ghana/Coin-Counter/master/CoinImages/PennyData.jpg" width = "600"><div align = "center"><figcaption>An example of an image used for penny data gathering</figcaption></div>
</p>

<p align = "center">
<img src= "https://raw.githubusercontent.com/Me-ghana/Coin-Counter/master/CoinImages/chart.png" width = "600"><div align = "center"><figcaption>Probability Distribution for Penny and Dime Radii</figcaption></div>
</p>

From this graph, it's clear we cannot rely on size alone to differentiate pennies from dimes.  

**Step 4C**
If you do a similar exercise with R, G, and B channels, you'll again find the overlap is quite large.  However, here's the distribution for the R:G ratio for pennies and dimes:

<p align = "center">
<img src= "https://raw.githubusercontent.com/Me-ghana/Coin-Counter/master/CoinImages/chartRG.png" width = "450"><div align = "center"><figcaption>Probability Distribution for Penny and Dime Radii</figcaption></div>
</p>

We can see a clear separation between the distribution of R:G dime and the penny values!  We can assume this would also be true of pennies vs nickels or quarters. This is a good time to discuss RGB vs HSV.  HSV is often used in computer vision for color based operations since it maps specific colors to a hue channel, while keeping the saturation (how white the color is) and the value (how dark the channel is) in two other channels.  For example, in HSV, all types of red color, regardless of illumination, will have the same hue value. I first tried HSV values for this exercise, but found the R:B separation was the best.  As a result, I'll proceed with working the in the RGB space.  


With similar distributions, I was able to find adequate separation between (1) pennies and nickels radii, and (2) nickels and quarters radii.  While there was some overlap (<15%) in both cases, let's proceed and see how well our coin counter does.
<div id = "P5"> </div>
### Step 5: Write calibration code body (optional)
Since the size and the color will change depending on the distance the camera is mounted from the coins and the lighting conditions, we'll first create a calibration program that should always be run prior to the actual coin counting algorithm.  This is optional, because you can just go into the main program and manually set whatever parameters you want.  We'll direct the user to place a single penny, nickel, dime, and quarter in front of the camera in designated space using a red "x".  Once the coin is placed on the x, the coin's R:G value and radius will be written out to a file each time a new frame is loaded, and displayed on the screen. Once the values are written out above each coin, the user knows they have all the calibration data and can start running the main program.  As an example, I've also chosen to include the R:B values.  You can add as many heuristics as you want!  Here's what the final calibration product will look like:

<div align="center">
  <a href="https://www.youtube.com/watch?v=kTZLyB5hBIY"><img src="https://img.youtube.com/vi/kTZLyB5hBIY/0.jpg" ></a>
	<div align = "center"><figcaption>Click image to watch video</figcaption></div>
</div>

Let's continue the code from earlier:


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
        # Step 3C
        # Draw keypoints
        im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (255, 0, 0),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
					      
					  
        # Step 5A
	# Instantiate the coordinates of where you will be directing the user to place coins
	xVal = 80
        yVal = 300
	# Instantiate the distances between coins
	xDelta = 140
	# Instantiate the distances between the coins and the text directing the user
	yDelta = -100
	
	# Step 5B
	# Create a loop that iterates once for each detected blob
        for x in range(0,len(keypoints)):
            # Draw a green border around the detected blob
            im_with_keypoints=cv2.circle(im_with_keypoints, (np.int(keypoints[x].pt[0]),np.int(keypoints[x].pt[1])),
	    radius=np.int(keypoints[x].size/2), color=(0,0,255), thickness=2) 
	    
	    # Step 5C & 6A
	    # Call the "detectCoinType" function
            im_with_keypoints = self.detectCoinType(im_with_keypoints,
	    np.int(keypoints[x].pt[0]),np.int(keypoints[x].pt[1]), xVal + 20 , yVal + yDelta, xDelta,
	    np.int(keypoints[x].size/2))
	    
	    
	# Step 5D & 6B
	# Add text on the image to make more user-friendly
	im_with_keypoints = self.imageText(im_with_keypoints, xVal, yVal, xDelta, yDelta)

	# Step 5E
	# Convert our BGR image to video output format and send to host over USB
        outframe.sendCvBGR(im_with_keypoints)
```

**Step 5A** We want the user to place the four coins evenly spaced in front of the camera in a specific order.  We'll draw a marker where each coin should be placed starting at (xVal,yVal). We'll label the image with the name of the coin the user should place yDelta pixels above the marker.

**Step 5B** This loop should iterate four times, one for each coin.  A green circlular border will be drawn around each coin.

**Step 5C** The whole point of the calibration step is to store the heuristic data for each coin type in a file. The "detectCoinType" function will do that.   See part 6 for more.

**Step 5D** We add text to the image to make the calibration portion more user-friendly.  See part 6 for more.
 
**Step 5D** The program is complete.  At this point, if you are using a JeVois, you would send the altered image out to be displayed on the video interface.

### Step 6: Write calibration code helper functions (optional)
Let's take a look at all the helper functions we use in this code, starting with the "detectCoinType" function.  
<div id = "P6"> </div>
**Step 6A: detectCoinType**
The "detectCoinType" function determines which marker is closest to the coin.  Since we have already determined the order of the markers (leftmost marker being 0 and rightmost being 3), we will know which coins we have detected.  We then write out the color information and radius of the detected coin to the matching text file for that coin.  The main coin counting program will then use this file to help tell coins apart. 

```python
     ## Function determines if detected circle represents Penny, Nickel, Quarter or Dime
     ## This is based on the (X,Y) coordinate (center needs to be within 15% of (X,Y) target)
     ## Function returns the following values based on detected coin type:
     ## Dime - 0
     ## Penny - 1
     ## Nickel - 2
     ## Quarter - 3
    
    # Step 6A
    # Pass in the (x,y) coordinates of the detected coin center as (circleCenterXCoord,circleCenterYCoord)
    # Pass in the (x,y) coordinates of the left most marker as (targetXVal, targetYVal)
    # Pass in the target spacing between coins as xDelta 
    # Pass in the radius of the detected coin
    def detectCoinType(self, image, circleCenterXCoord,circleCenterYCoord, \
        targetXVal, targetYVal, xDelta, radius):  

	# Check each marker and see which the coin is closest too
        for i in range(0,4):
           # Calculate % difference b/w center of detected circle and center of coin target
            xDifference = abs(circleCenterXCoord-targetXVal)/targetXVal
            yDifference = abs(circleCenterYCoord-targetYVal)/targetYVal

            # If center of circle is within 30% of the target label the image with the detected coin type
	    # E.g. If a coin is detected to be near the Penny marker, label
            if (xDifference < 0.3) and (yDifference < 0.3):
               
	       	# Step 6B
                # Call the printCoinType function to get the coin type
		# The variable "i" is the number iteration of this loop
		# E.g. if i was 3, this is the third coin from the left, which is the Nickel 
                coin = self.printCoinType(i)
		
		# Continue with Step 6A
                # Get RGB values from coin
                # Radius isn't a great way to distinguish between the coins, looking at RGB values is more helpful
                # Specifically, looking at the R:G and R:B ratios helps distinguish pennies from the rest
                
		# We use a circular area of half the radius of the coin with the same center to get the average RGB value
		circle_img = np.zeros((height, width), np.uint8)
                cv2.circle(circle_img, (circleCenterXCoord, circleCenterYCoord), int(radius/2), [255,255,255], -1)
                circle_img = np.uint8(circle_img)
                
		# Compute the mean RGB value in this circle
                mean_val = cv2.mean(image, circle_img)[::-1]
		
		# Store these mean RGB values and use them to compute the ratios
                temp = mean_val[1:]
                red = temp[0]
                green = temp[1]
                blue = temp[2]
                ratioRG = red/green
                ratioRB = red/blue
		
		# If you want, you can add more heuristics, like the rgb squared value, which I've commented out.  
                # rgbSqrd =  math.sqrt(red*red + blue*blue + green*green)

		# Step 6C
		# Call the writeToFile function to write out the ratios and radii information for each coin
                self.writeToFile(ratioRG,ratioRB,radius,coin)

		# Write out the calculated values on the screen
		# This can help you troubleshoot if you notice the values are not what you expect
                cv2.putText(image, "RG " + str("%.2f" % ratioRG), (targetXVal, targetYVal - 105), \
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(image, "RB " + str("%.2f" % ratioRB), (targetXVal, targetYVal - 130), \
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

		# Write out the detected coin type values on the screen
		# This allows you to double check you are detecting the expected coin
                cv2.putText(image, coin, (targetXVal, targetYVal - 60), \
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
		# Write out the detected coin radius on the screen
                cv2.putText(image, "radius " + str("%.2f" % radius), (targetXVal, targetYVal - 80), \
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
		
		# We have detected the coin and added the necessary text to the image
		# Now we can exit this function
                return image
		
            # Every time the loops is incremented, we shift one marker to the left
            # As a result, we have to update the x coordinate of the marker
            targetXVal = targetXVal + xDelta
        
	# In this case, no coin was detected within 30% of a marker
        return image
```

**Step 6B: Function imageText**
This is a simple function which adds text to the image to help the user know where to place which coin.  For example, the user should place a dime on the red "x" with the word "dime" in red underneat the "x".  See the video in Part 5 to get an idea of what the end result should be.

```python
	
	# Add title and istruction
        cv2.putText(image, "COIN CALIBRATION", (20, 20), \
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(image, "Center the corresponding coin at each X",(20, 40), \
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
	# Add coin names from left to right on the screen
        cv2.putText(image, "Dime", (xVal, yVal), \
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(image, "Penny", (xVal + xDelta, yVal), \
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(image, "Nickel", (xVal + 2*xDelta, yVal), \
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(image, "Quarter", (xVal + 3*xDelta, yVal), \
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
          
	# Add "X"s where coins should be placed from left to right
        cv2.putText(image, "X", (xVal + 20, yVal + yDelta), \
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(image, "X", (xVal + xDelta + 20, yVal + yDelta), \
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(image, "X", (xVal + 2*xDelta + 20, yVal + yDelta), \
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(image, "X", (xVal + 3*xDelta + 20, yVal + yDelta), \
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
	       
        return image
```


**Step 6B: Function printCoinType**
This is another simple function which returns the type of the coin detected based on its closest marker.     

```python
    def printCoinType(self, type):
        if type == 0:
            coin = 'Dime'
        if type == 1:
            coin = 'Penny'
        if type == 2:
            coin = 'Nickel'
        if type == 3:
            coin = 'Quarter'
        return coin
```
**Step 6C: Function writeToFile**
This is our last helper function!  Here, we write out to three files for each coin, storing the R:G, R:B, and radius information.   Note that since we are appending infromation to the end of the file, the user will have to go in and delete the files completely if they want to start collecting calibration data for a new environment.  A future imrpovement may be to clear the file after reaching a certain number of lines, so that files can be completely updated by running the calibration program for a given amount of time.  On the JeVois, data can be stored in the /jevois/data folder. 

```python
    def writeToFile(self,ratioRG,ratioRB,radius,coin):
        fo = open("/jevois/data/" + coin+"_RG.txt","a")
        fo.write(str(ratioRG) + "\n")
        fo.close()

        fo = open("/jevois/data/" + coin+"_RB.txt","a")
        fo.write(str(ratioRB) + "\n")
        fo.close()

        fo = open("/jevois/data/" + coin+"_Radius.txt","a")
        fo.write(str(radius) + "\n")
        fo.close()  
```
We're finally done with the calibration program!  If you put steps 1-6 together, you should have a fully functional calibration program.  Now let's move on to the actual coin counting program.
<div id = "P7"> </div>
### Step 7: Write main coin counter program helper functions
The pre-processing of the main coin counter program will be very similar to the calibration program, so we'll get to that later when we put everything together.  For now, we will focus on the helper functions that will read in the data from the files the calibration program created.

**Step 7A: Function coinValues**
This function reads the data from the file corresponding to the coin name passed as a parameter.  The function then computes several statistics, like the average R:B value, that can be later used by the user to customize coin identification. 


```python
    def coinValues(self,coin):
    	# The number of coins for which data is stored (number of lines in file)
        coinNum = 0
	# A running sum of the R:G ratio
        rgSum = 0
	# A running square of sums of the R:G ratio
        rgSqSum = 0
	# A running sum of the R:B ratio
        rbSum = 0
	# A running square of sums of the R:B ratio
        rbSqSum = 0
	# A running sum of the radii
        radiusSum = 0
	# A running square of sums of the radii
        radiusSqSum = 0
	
	# Open file containing R:G data and compute related variables
        with open("/jevois/data/" + coin + "_RG.txt","r") as fo:
            for line in fo:
                line = line.strip()
                if line:
                     rgSum = rgSum + float(line)
                     rgSqSum = float(line) * float(line) + rgSqSum
                     coinNum = coinNum + 1
		     
	# Computed Statistics
        rgAvg = rgSum/coinNum
        rgSqAvg = rgSqSum/coinNum
        varianceRG = rgSqAvg - rgAvg*rgAvg
        standardDeviationRG = math.sqrt(varianceRG)

	# Open file containing R:B data and compute related variables
        with open("/jevois/data/" + coin + "_RB.txt","r") as fo:
            for line in fo: 
                line = line.strip()
                if line:
                     rbSum = rbSum + float(line)
                     rbSqSum = float(line) * float(line) + rbSqSum
		     
	# Computed Statistics
        rbAvg = rbSum/coinNum
        rbSqAvg = rbSqSum/coinNum
        varianceRB = rbSqAvg - rbAvg*rbAvg
        standardDeviationRB = math.sqrt(varianceRB)

	# Open file containing radius data and compute related variables
        with open("/jevois/data/" + coin + "_Radius.txt","r") as fo:
            for line in fo: 
                line = line.strip()
                if line:
                     radiusSum = radiusSum + float(line)
                     radiusSqSum = float(line) * float(line) + radiusSqSum

	# Computed Statistics
        radiusAvg = radiusSum/coinNum
        radiusSqAvg = radiusSqSum/coinNum
        variance = radiusSqAvg - radiusAvg*radiusAvg
        standardDeviation = math.sqrt(variance)


	# Create an array storing this data and return array
        coinValues = np.array([radiusAvg,rgAvg,rbAvg,standardDeviation,standardDeviationRG,standardDeviationRB])
        return coinValues
```

**Step 7B: Function addCoinStats**
This function is to help with troubleshooting, and I do not actually call this function in my final code.  What it does is display all the calculated statistics for each type of coin.  This can help you notice differences between coins, and manually set the parameters.

```python
    # Adds statistics text to the screen
    # Can be used for seeing real-time values of coin data
    def addCoinStats(self,inimg,values,Coin,initialX,deltaX):
        if Coin == 'Penny' or Coin == 'Dime':
            cv2.putText(inimg, str(Coin) + " Radius " + str("%.2f" % values[0]), (20, initialX+deltaX), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(inimg, str(Coin) + " RG " + str("%.2f" % values[1]), (20, initialX+deltaX*2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(inimg, str(Coin) + " RB " + str("%.3f" % values[2]), (20, initialX+deltaX*3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(inimg, str(Coin) + " Radius Standard Dev " + str("%.3f" % values[3]), (20, initialX+deltaX*4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(inimg, str(Coin) + " RG Standard Dev" + str("%.3f" % values[4]), (20, initialX+deltaX*5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(inimg, str(Coin) + " RB Standard Dev " + str("%.3f" % values[5]), (20, initialX+deltaX*6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        else:
            cv2.putText(inimg, str(Coin) + " Radius " + str("%.2f" % values[0]), (20, initialX+deltaX), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(inimg, str(Coin) + " Standard Dev " + str("%.3f" % values[3]), (20, initialX+deltaX*2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        return inimg
```
<div id = "P8"> </div>
### Step 8: Write main coin detection code program body
In this portion, we'll replicate the pre-processing code from Steps 1-4, with a few differences in the constructor.  Using the helper functions from Step 7, we'll read in the data files and store the values in an numpy array for each coin.  Then, just like we did in the calibration portion, we'll extract the average RGB values and the radius for each detected blob.  The last step is to compare the data from the files to the detected blob, and determine what coin the blob is.  

**Step 8A**
First, we create variables to kep track of the number of each type of coin.
```python
	# Step 8A
        # Create variables to store number of each type of coin
        pennyNum = 0
        nickelNum = 0
        dimeNum = 0
        quarterNum = 0
```

**Step 8B**
We'll iterate though each detected blob and get the average RGB and radius values. Since most of this code is replicated, I won't include it again, but you can [download it off of github](https://github.com/Me-ghana/Coin-Counter). 

**Step 8C**
Next, we compare the values of the detected coins to the averaged values from the files. Below is the code doing the comparison.  Remember the radius data and R:G data is stored in the 0th and 1st element of the data array, respectively. It looks long, but it's just a series of comparison operations!


```python
    # Step 8C
    # To distinguish between nickels and quarters, we rely solely on radius
    # If the radius is greater than the average quarter radius, label this coin as a quarter
    if (radius > quarterValues[0]):
	cv2.putText(im_with_keypoints, "Q", center , cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
	quarterNum += 1
    # Assign the coin type based on whichever coin radius is closest in value
    elif (radius > nickelValues[0]):
	# For example: 
	# If the radius is greater than the average nickel radius, and closer to the average 
	# nickel radius than quarter radius, label this as as nickel
	if ((quarterValues[0] - radius) > (radius - nickelValues[0])):
	    cv2.putText(im_with_keypoints, "N", center , cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
	    nickelNum += 1
	# For example: 
	# If the radius is greater than the average nickel radius,
	# but closer to the average quarter radius, label as a quarter                
	else:
	    cv2.putText(im_with_keypoints, "Q", center , cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
	    quarterNum += 1


    # To distinguish between nickels and pennies, we rely largely on the R:G ratio
    # If you want, you can add in other dependencies, such as using the R:B ratio or the squared RGB values 
    elif (radius > pennyValues[0]):
	# If the average nickel and penny radius are within 15%, 
	# they are too similar, so we rely solely on the R:G ratio
	# Assign the coin type based on whichever coin has the closest R:G ratio value
	if (abs(nickelValues[0]-pennyValues[0])/pennyValues[0] < 0.15):
	    if ((abs(nickelValues[1]-ratioRG)) > (abs(pennyValues[1]-ratioRG)) ):
		cv2.putText(im_with_keypoints, "P", center , cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
		pennyNum += 1
	    else:
		cv2.putText(im_with_keypoints, "N", center , cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
		nickelNum += 1
	# If the average nickel and penny radius are greater than 15%, we rely on the radius
	elif ((nickelValues[0] - radius) > (radius - pennyValues[0])):
	    cv2.putText(im_with_keypoints, "P", center , cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
	    pennyNum += 1
	else:
	    cv2.putText(im_with_keypoints, "N", center , cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
	    nickelNum += 1

    # If the radius is less the the average dime radius, label as dime
    elif(radius < dimeValues[0]):
	cv2.putText(im_with_keypoints, "D", center , cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
	dimeNum += 1

    # To distinguish between dimes and pennies, we rely largely on the R:G ratio
    elif (abs(dimeValues[0]-pennyValues[0])/pennyValues[0] < 0.15):
	# If the average nickel and penny radius are within 15%, we rely solely on the R:G ratio
	# Assign the coin type based on whichever coin has the closest R:G ratio value
	if ((abs(pennyValues[1]-ratioRG)) > (abs(dimeValues[1]-ratioRG)) ):
		cv2.putText(im_with_keypoints, "D", center , cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
		dimeNum += 1
	else:
		cv2.putText(im_with_keypoints, "P", center , cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
		pennyNum += 1

    # If the average dime and penny radius are greater than 15%, we rely on the radius
    elif ((radius - dimeValues[0]) > (pennyValues[0]-radius)):
	cv2.putText(im_with_keypoints, "P", center , cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
	pennyNum += 1
    else:
	cv2.putText(im_with_keypoints, "D", center , cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
	dimeNum += 1
```

**Step 8D**
We calculate the total sum and display it on screen.  And we're done!

```python
    	# Step 8D
        # Compute the total value of all coins on screen
        totalVal = pennyNum*0.01 + dimeNum*0.1 + nickelNum*0.05 + quarterNum*0.25

        # Write out the total value and the number of each type of coin on the screen
        cv2.putText(im_with_keypoints, "!Total Value of Coins: $" + str("%.2f" % totalVal), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        cv2.putText(im_with_keypoints, "Pennies: " + str(pennyNum), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        cv2.putText(im_with_keypoints, "Nickels: " + str(nickelNum), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        cv2.putText(im_with_keypoints, "Dimes: " + str(dimeNum), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        cv2.putText(im_with_keypoints, "Quarters: " + str(quarterNum), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
```
<div id = "P9"> </div>
### Step 9: Count coins using the calibration and coin detection programs!

We're ready to test our programs!  Let's summarize what we're going to do to run the program, and also point out a few important points.

A. Secure the JeVois (or whatever device you are using) above your workspace.  Make sure your workspace has consistent lighting and a white or black background. 

B. Run the JeVois program "Coin Calibration" at resolution 640x480 and 34fps.  Line up your coins according to the instructions on the screen.  Stop the program and turn off the video interface.  if you are using the JeVois, connect to the USB.  This step should look like this: 

<div align="center">
  <a href="https://www.youtube.com/watch?v=kTZLyB5hBIY"><img src="https://img.youtube.com/vi/kTZLyB5hBIY/0.jpg" ></a>
	<div align = "center"><figcaption>Click image to watch video</figcaption></div>
</div>

C. Open the JeVois USB and change directory into "/jevois/data", or wherever you store your data files on your machine. There should be twelve files (three for each coin).  Delete all the files to remove the stale data.

D. Eject the JeVois USB and then wait for the camera to blink red.  Once the JeVois is ready, run the calibration program again.  Your coins should already be set up in the right spots, just like in this video below.

<div align="center">
  <a href="https://www.youtube.com/watch?v=cIm6demdQMY"><img src="https://img.youtube.com/vi/cIm6demdQMY/0.jpg" ></a>
	<div align = "center"><figcaption>Click image to watch video</figcaption></div>
</div>

E. After a few seconds, more than enough calibration data should be collected.  Switch programs to the "CoinCounter".  You should see a total value at the top left corner.  Add more coins, and see how it works!

<div id = "PT"> </div>
### Trouble-Shooting
If you've followed steps A-E and are getting too many errors, try modifying the Coin Counter file by adding the "addCoinStats" function, or simply look at your textfile data.  Here are some specific ways you may be able to improve your performance:
* My quarters and nickels are too similar &/or my dimes and nickels are too similar!
  
  How close is your camera?  In this example, the camera was only 17cm away from the coins.  If your camera is too far away, it will not be able to detect the difference in size between the silver coins.
  
* My pennies and nickels are too similar &/or my pennies and dimes are too similar!
  
  Are you relying too heavily on radius to distinguish between these coins?  Try using color information.

* I'm using color information and still can't distinguish between pennies and other coins!
  
  Make sure you have even lighting, and that you are working either a white or black surface. If the lighting is coming at an angle, it could cause shadows that increase the radius of some of your coins.  Also make sure your camera is secured and not moving when you take your calibration data and when you run the main program. If you're still having issues, try graphing probability distributions for the different heuristics and see if what you are trying to achieve is possible.  You may be using a machine with too low a resolution, or perhaps your coins are too weathered to distinguish.

<div id = "PTA"> </div>
### Take-Aways
This is a great project to learn about different computer vision algorithms, and also get used to using OpenCV.  Clearly, this is not the most robust way to count coins.  There are lots of issues - it requires calibration, it's limited to US coins, and it won't work on dirty or very dark coins.  The good news is there are a lot of ways we can improve on this project - here are some ideas:

1. Coins have very specific weights - use a sensitive scale that interacts with an arduino to add confidence values to the computed sum

2. Take the data we used earlier to create probability density functions and use them to train a convolution neural net. Use the CNN to identify coins instead of relying on color spaces and coin sizes.  The plus side to this is you can keep adding to your library of images, and start identifying coins from different countries.

3. Explore other OpenCV functions, such as SIFT, for coin identification
