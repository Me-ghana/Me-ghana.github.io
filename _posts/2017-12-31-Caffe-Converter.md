This tutorial will explain how to use the Caffe to Tiny-DNN Converter

 The Caffe deep learning framework is often used in vision applications. However, if you have limited computation resources, you may want to use a different deep learning framework, such as tinyDNN. In my case, I would like to convert <a href = "https://github.com/CSAILVision/places365" target = "_blank"> MIT's Places365-CNN</a> caffemodel to tiny-dnn. This will allow MIT's model to run on the <a href = "http://jevois.org/" target="_blank">JeVois, a Smart Machine Vision Camera,</a> which has tiny-dnn support. I'll outline the steps I take below:


**My System:**
I chose to use Ubuntu 17.10 (Artful ardvark).  I <a href="IntelNucSetUp.html" target = "_blank">assembled an Intel NUC Mini-PC</a> so that I was not constrained by my virtual machine. 

**Step 1: Setup**
 Follow the instructions on <a href = "https://github.com/tiny-dnn/tiny-dnn/wiki/A-beginner's-guide-to-build-examples" target = "_blank">"A beginner's guide to build examples"</a> from the tiny-dnn GitHub.
 ```c++
  git clone https://github.com/tiny-dnn/tiny-dnn.git<br>
  cd tiny-dnn<br>
  cmake -DBUILD_EXAMPLES=ON .<br>
  make
```
**Step 2A: Prerequisites**

Follow the <a href="https://github.com/tiny-dnn/tiny-dnn/tree/master/examples/caffe_converter" target = "_blank">"Import Caffe Model to tiny-dnn"</a> instructions from the tiny-dnn caffe converter git repository.  Before you can continue, you will first need <a href="https://developers.google.com/protocol-buffers/" target ="_blank">Google protobuf</a> (a Google data structure similar to XML) and <a href ="https://opencv.org/" target = "_blank">OpenCV</a>.  I used the Conda package manager to install both.  Skip ahead to <a href="#3">Step 3</a> if you already have these prerequisites.  

**Step 2B: Anaconda Installation**
