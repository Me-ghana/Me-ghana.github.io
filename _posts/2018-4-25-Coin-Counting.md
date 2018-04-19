---
layout: post
title: Coin Counting with the JeVois
---

This tutorial will help explain how to use OpenCV with a video stream to identify U.S. coins using image segmentation and blob detection.  

I found the exercise of identifying U.S. coins to be a good introduction to some of the basic functions in OpenCV and some important computer vision algorithms.  While this post explains how to use OpenCV with the [JeVois](http://jevois.org/), a smart machine vision camera, to count U.S. coins, you can use any video stream that is read image by image.

My first goal was to see how well I could identify the four most common U.S. coins, the penny, nickel, dime, and quater, by detecting just circles and coin colors.  Let's breakdown what the order of what we're going to do into five main steps:
1. Read in our image
2. Pre-process our image 
3. Identify the regions of interest (ROI) which are the coins
4. Identify each ROI as a particular coin 
5. Computer the total value


###Step 1: Read in our image
Step 1 is pretty simple, especially if you are using the JeVois. If you want to learn more about the built-in JeVois functions, check out the very helpful [JeVois Programming Tutorials](http://jevois.org/tutorials/ProgrammerTutorials.html).  If you are using a different video stream, read in a single image and store it as "img".

Here's the following code for reading in our image.  

```python 
    def process(self, inframe, outframe):
        # Get the next camera image (may block until it is captured) and convert it to OpenCV BGR (for color output):
        img = inframe.getCvBGR()
```



