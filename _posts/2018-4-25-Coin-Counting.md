---
layout: post
title: Coin Counting with the JeVois
---

This tutorial will help explain how to use OpenCV with a video stream to identify U.S. coins using image segmentation and blob detection.  

I found the exercise of identifying U.S. coins to be a good introduction to some of the basic functions in OpenCV and some important computer vision algorithms.  While this post explains how to use OpenCV with the [JeVois](http://jevois.org/), a smart machine vision camera, to count U.S. coins, you can use any video stream that is read image by image.

My first goal was to see how well I could identify the four most common U.S. coins, the penny, nickel, dime, and quater, by detecting just circles and coin colors.  Let's take a look at the initial code:

```python
/* Some pointless Python */
import libjevois as jevois
import cv2
import numpy as np
```

