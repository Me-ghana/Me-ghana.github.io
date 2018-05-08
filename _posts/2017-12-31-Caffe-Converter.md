This tutorial will explain how to use the Caffe to Tiny-DNN Converter

 The Caffe deep learning framework is often used in vision applications. However, if you have limited computation resources, you may want to use a different deep learning framework, such as tinyDNN. In my case, I would like to convert <a href = "https://github.com/CSAILVision/places365" target = "_blank"> MIT's Places365-CNN</a> caffemodel to tiny-dnn. This will allow MIT's model to run on the <a href = "http://jevois.org/" target="_blank">JeVois, a Smart Machine Vision Camera,</a> which has tiny-dnn support. I'll outline the steps I take below:


**My System:**

I chose to use Ubuntu 17.10 (Artful ardvark).  I assembled an Intel NUC Mini-PC so that I was not constrained by my virtual machine. 


**Step 1: Setup**

 Follow the instructions on <a href = "https://github.com/tiny-dnn/tiny-dnn/wiki/A-beginner's-guide-to-build-examples" target = "_blank">"A beginner's guide to build examples"</a> from the tiny-dnn GitHub.
 ```c++
  git clone https://github.com/tiny-dnn/tiny-dnn.git
  cd tiny-dnn
  cmake -DBUILD_EXAMPLES=ON .
  make
```

**Step 2A: Prerequisites**

Follow the <a href="https://github.com/tiny-dnn/tiny-dnn/tree/master/examples/caffe_converter" target = "_blank">"Import Caffe Model to tiny-dnn"</a> instructions from the tiny-dnn caffe converter git repository.  Before you can continue, you will first need <a href="https://developers.google.com/protocol-buffers/" target ="_blank">Google protobuf</a> (a Google data structure similar to XML) and <a href ="https://opencv.org/" target = "_blank">OpenCV</a>.  I used the Conda package manager to install both.  Skip ahead to <a href="#3">Step 3</a> if you already have these prerequisites.  


**Step 2B: Anaconda Installation**

Install the <a href ="https://www.anaconda.com/download/#macos" target ="_blank">Anaconda</a> package manager with the default settings. I chose to use the Python 3.6 version since support for Python 2.7 will end in 2020. <!--As a result, we will have to make a few changes to Caffe models later on.--> You can check your installation by typing "python". You should have a Python 3 version using the Anaconda distribution. You can also check the list of installed packages you now have in your active environment by typing "conda list".

<p align = "center">
	<img src= "https://raw.githubusercontent.com/Me-ghana/Me-ghana.github.io/master/images/CaffeConverter/condaPython.png">
<!--		<div align = "center">
			<figcaption></figcaption>
		</div>-->
</p>


**Step 2C: Protobuf Installation**

Use Conda to <a href = "https://anaconda.org/anaconda/protobuf" target = "_blank">install</a> Google Protocol Buffers: 
```c++ 
conda install -c anaconda protobuf
```
You can check the version of your install with:
```c++
protoc --version
```


**Step 2D: OpenCV Installation**

The last prerequiste you need now is OpenCV.  Use Conda to <a href = "https://anaconda.org/menpo/opencv" target = "_blank">install:</a>
```c++
conda install -c menpo opencv
```
You can check the version of your install with:
```c++
pkg-config --modversion opencv
```


**Step 3: Run Compiler Protoc**

Change directories into the “caffe” folder and use the Protocol Buffer “protoc” command.  
```c++
cd tiny_dnn/io/caffe
protoc caffe.proto --cpp_out=./
```
If all goes well, this compilation will produce an implementation file (caffe.pb.cc) and a header file (caffe.pb.h). In the above code, the "-cpp_out" option specifies C++ classes.

<p align = "center">
<img  src = "https://raw.githubusercontent.com/Me-ghana/Me-ghana.github.io/master/images/CaffeConverter/compile.png" >
</p>


**Step 4: Compilation**

Compile and link the following two files: 

    1. tiny_dnn/io/caffe/caffe.pb.cc
    2. examples/caffe_converter/caffe_converter.cpp

using the following code (as a miniumum for Ubuntu). The "-I" flag adds include directories of header files.  Use the "-l" flag to link to the pthread and protobuf libraries.  
```c++
      g++ \
      -I/YOUR PATH TO TINY-DNN FOLDER FROM HOME DIRECTORY\
      -std=c++14  -o caffe_converter.bin \
      caffe.pb.cc \
      <b>YOUR PATH TO CAFFE_CONVERTER.CPP FROM CURRENT WORKING DIRECTORY \
      -lprotobuf -lpthread 
```
You should now have an executable **caffe_converter.bin**
<div class = "im-center">
    <img  src = "https://raw.githubusercontent.com/Me-ghana/Me-ghana.github.io/master/images/CaffeConverter/compile.png">
</div> 

**Step 5: Execution with CaffeNet Model**
