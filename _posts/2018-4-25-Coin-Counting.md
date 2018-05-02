---
layout: post
title: Coin Counting with the JeVois in Python
---

This tutorial will help explain how to identify U.S. coins from a video stream with Python and OpenCV, like in the video below. I found the exercise of identifying U.S. coins to be a good introduction to some of the basic functions in OpenCV and some important computer vision algorithms.  The final code uses blob detection, but we'll explore other algorithms such as Hough Circles and the Watershed algorithm along the way.  While this post explains how to use OpenCV with the [JeVois](http://jevois.org/), a smart machine vision camera, you can use any video stream you can extract images from.  At the very least, you'll need to have OpenCV, numpy, and an image of some coins.

<div align="center">
  <a href="https://www.youtube.com/watch?v=R4LO0sgfBmU"><img src="https://img.youtube.com/vi/R4LO0sgfBmU/0.jpg" ></a>
	<div align = "center"><figcaption>Click image to watch video</figcaption></div>
</div>

My first goal was to see how well I could identify the four most common U.S. coins, the penny, nickel, dime, and quater, by detecting just the circle sizes and coin colors.  We'll have nine steps:
<a href = "P1"> 1. Read in our image </a>
2. Pre-process our image 
3. Identify coins with blob detection
4. Determine the heuristics and color space 
5. Write calibration program body (optional)
6. Write calibration program helper functions (optional)
7. Write coin detection program helper functions
8. Write coin detection program program body
9. Count coins using the calibration and coin detection programs!

The big picture is we will be creating two programs, a calibration program and a coin counting program.  The calibration program will be run first and generate files with data about each coin type.  The main program will then open these files and use this data to identify coin types, and finally to find the total value of all the coins.  (Users can skip the calibration step, and write in the parameters manually in the coin counting program.)

You can follow along by [downloading the code](https://github.com/Me-ghana/Coin-Counter) for both the Calibration and Counting program. If you run into any issues, scroll down to check out some tourbleshooting tips!

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

<div id = "P1">### Step 2: Pre-process our image</div>
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
In this step, we make our gray-scale image black and white by choosing a threshold using Otsu's Method.  This algorithm finds the optimal threshold value in a bimodal image. In this example, we are using a white background and the coins are darker.  We want to coins to be black and the background to be white, so we use the inverse method provided by OpenCV.  This is because the simple blob detector typically detects dark spots as blobs.  (Later on, we can add a simple function to determine the background color to see if the inverse method should be used or not automatically.)

Here's what our image looks like after thresholding:
<p align="center">
  <img src= "https://raw.githubusercontent.com/Me-ghana/Coin-Counter/master/CoinImages/Coins2D.png" width = "450">
</p>

### Step 3: Identify coins with blob detection
We'll instantiate a simple blob detector in our constructor in order to identify coins in our image.  The simple blob detector algorithm works by:
   1. Thresholding the image into not just one, but several binary images
   2. Extracting connected pixels in each binary image with findContours 
   3. Finally merging blobs that have centers close to one another. You can specify specific thresholds and filters to have increased control over how blobs should be extracted. 

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

We can see that we get better performance with Hough Circles than with the find contours algorithm.  You can choose to replace the circle detection code with Hough Circles if you want.  Here is the video of a final coin counting algorithm that uses Hough Circles.  As you can see, it does a pretty good job and is even able to detect coins that are overlapping. However, the size of the circles are undulating.  This may be due to the fan on the JeVois, which causes vibrations.  If we compare the final coin video with Simple Blob detection, we can see the coin outline is much steadier.  If you using static images, or have a very steady video stream, Hough Circles is a great option. 

<div align="center">
  <a href="https://www.youtube.com/watch?v=lPb4vpTNWcI"><img src="https://img.youtube.com/vi/lPb4vpTNWcI/0.jpg" alt="IMAGE ALT TEXT"></a>
	<div align = "center"><figcaption>Click image watch video</figcaption></div>
</div>

###### Alternative #2: The Watershed Algorithm
The watershed algorithm is an intuitive method to identify circles.  The idea is that an image can be treated like a topographic map, with valleys separating low points.  If  the center of each coin is the low point in the map, we can fill the valley of each lowpoint with a unique color of water.  The boundaries of each coin are indicated where different water colors merge.  

<p align="center">
	<img src= "https://raw.githubusercontent.com/Me-ghana/Coin-Counter/master/CoinImages/CoinsAltWater.png" width = "450"><div align = "center"><figcaption>Watershed Boundaries</figcaption></div>
</p>
<p align="center">
	<img src= "https://raw.githubusercontent.com/Me-ghana/Coin-Counter/master/CoinImages/CoinsAltWater2.png" width = "450"><div align = "center"><figcaption>Final Circles</figcaption></div>
</p>

This is a great way to identify overlapping regions.  However, it  doesn't give us the tightest circles we've seen so far. This implemetnation of the watershed algorithm relies on a series of dilations and erosions to identify the low points, and this also reduces the accuracy of the final circle.    

### Step 4: Determine the heuristics and color space
**Note: You do not need to do this step yourself if you merely want to use the algorithm.  However, if you're interested in finding your own heuristics, you can read this and follow the method I outline below**

In this project, we are attempting to use basic image processing to identify U.S. Coins.  Two simple heuristics we can use are coin size and color.  First, let's check to see if it is possible to use these heuristics to tell U.S. Coins apart.

**Step 4A**
I wanted to create probability distrbiutions for coin color and size.  The most similar in size is the penny and the dime. I went to the bank and got about 400 of each type of coin (I also got a few weird looks from the bank tellers). If we can tell these two coins apart with a combination of size and color heuristics, we can be fairly confident that we can tell the nickel and quarter apart as well.  

**Step 4B**
Let's first look at the radius data.  Using the JeVois, I used the "screen capture" to capture images of coins in the video screen.  I imaged about 30 coins at a time, and used the simple blob detector method to extract each coin area.  This method had a 96% accuracy of correctly finding the coin boundaries.  (Sometimes coins near the edge of the camera, and coins that I had not completely separated, were not identified correctly).  I recorded the radius, R, G, and B values for each coin.

<p align = "center">
<img src= "https://raw.githubusercontent.com/Me-ghana/Coin-Counter/master/CoinImages/PennyData.jpg" width = "600"><div align = "center"><figcaption>An example of an image used for penny data gathering</figcaption></div>
</p>

<p align = "center">
<img src= "https://raw.githubusercontent.com/Me-ghana/Coin-Counter/master/CoinImages/chart.png" width = "600"><div align = "center"><figcaption>Probability Distribution for Penny and Dime Radii</figcaption></div>
</p>

From this graph, it's clear we cannot rely on size alone to differentiate pennies from dimes.  

**Step 4C**
If you do a similar exercise with R, G, and B channels, you'll again find the overlap is quite large.  However, here's the distribution for the R:B ratio for pennies and dimes:

<p align = "center">
<img src= "https://raw.githubusercontent.com/Me-ghana/Coin-Counter/master/CoinImages/chart2.png" width = "450"><div align = "center"><figcaption>Probability Distribution for Penny and Dime Radii</figcaption></div>
</p>

We can see a clear separation between the distribution to R:B values in the dimes and the pennies!  We can assume this would also be true of pennies vs nickels or quarters. This is a good time to discuss RGB vs HSV.  HSV is often used in computer vision for color based operations since it maps specific colors to a hue channel, while keeping the saturation (how white the color is) and the value (how dark the channel is) in two other channels.  For example, in HSV, all types of red color, regardless of illumination, will have the same hue value. I first tried HSV values for this exercise, but found the R:B separation was the best.  As a result, I'll proceed with working the in the RGB space.  


With similar distributions, I was able to find adequate separation between (1) pennies and nickels radii, and (2) nickels and quarters radii.  While there was some overlap (<15%) in both cases, let's proceed and see how well our coin counter does.

### Step 5: Write calibration code body (optional)
Since the size and the color will change depending on the distance the camera is mounted from the coins and the lighting conditions, we'll first create a calibration program that should always be run prior to the actual coin counting algorithm.  This is optional, because you can just go into the main program and manually set whatever parameters you want.  We'll direct the user to place a single penny, nickel, dime, and quarter in front of the camera in designated space using a red "x".  Once the coin is placed on the x, the coin's R:B value and radius will be written out to a file each time a new frame is loaded, and displayed on the screen. Once the values are written out above each coin, the user knows they have all the calibration data and can start running the main program.  As an example, I've also chosen to include the R:G values.  You can add as many heuristics as you want!  Here's what the final calibration product will look like:




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

### Step 7: Write main coin counter program helper functions
The pre-processing of te main coin counter program will be very similar to the calibration program, so we'll get to that later when we put everything together.  For now, we will focus on the helper functions that will read in the data from the files the calibration program created.

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
### Step 8: Write main coin detection code program body

**Step 8A**

### Step 9: Count coins using the calibration and coin detection programs!

We're ready to test our programs!  Let's summarize what we're going to do to run the program, and also point out a few nuances.

1. Secure the JeVois (or whatever device you are using) above your workspace.  Make sure your workspace has consistent lighting and a light (preferably white) background. 
2. Run the JeVois program "Coin Calibration" at resolution 480 X 600 and 20fps.  Line up your coins according to the instructions on the screen.  Stop the program and turn off the video interface.  if you are using the JeVois, connect to the USB.
3. Open the JeVois USB and change directory into "/jevois/data", or wherever you store your data files on your machine. There should be twelve files (three for each coin).  Delete all the files to remove the stale data.
4. Eject the JeVois USB and then wait for the camera to blink red.  Once the JeVois is ready, run the calibration program.  Your coins should already be set up in the right spots.  
5. After a few seconds, more than enough calibration data should be collected.  Switch programs to the "Coin Counter".  You should see a total value at the top left corner.  Add more coins, and see how it works!

Trouble-Shooting
If you've followed steps 1-5 and are getting too many errors, try modifying the Coin Counter file by adding the "addCoinStats" function, or simply look at your textfile data.  Here are some specific ways you may be able to improve your performance:
⋅⋅* My quarters and nickels are too similar &/or my dimes and nickels are too similar!
...How close is your camera?  In this example, the camera was only 17cm away from the coins.  If your camera is too far away, it will not be able to detect the difference in size between the silver coins.
..* My pennies and nickels are too similar &/or my pennies and dimes are too similar!
...Are you relying too heavily on radius to distinguish between these coins?  Try using color information.
..* I'm using color information and still can't distinguish between pennies and other coins!
...Make sure you have even lighting, and that you are working off a white or light surface. Also make sure your camera is secured and not moving when you take your calibration data and when you run the main program. If you're still having issues, try graphing probability distributions for the different heuristics and see if what you are trying to achieve is possible.  You may be using a machine with too low a resolution, or perhaps your coins are too weathered to distinguish.

Take-Aways
This is a great project to learn about different computer vision algorithms, and also get used to using OpenCV.  Clearly, this is not the most robust way to count coins.  There are lots of issues - it requires calibration, it's limited to US coins, and it won't work on dirty or very dark coins.  The good news is there are a lot of ways we can improve on this project - here are some ideas:
1. Coins have very specific weights - use a sensitive scale that interacts with an arduino to add confidence values to the computed sum
2. Take the data we used earlier to create probability density functions and use them to train a convolution neural net. Use the CNN to identify coins instead of relying on color spaces and coin sizes.  The plus side to this is you can keep adding to your library of images, and start identifying coins from different countries.
3. Explore other OpenCV functions, such as SIFT, for coin identification
