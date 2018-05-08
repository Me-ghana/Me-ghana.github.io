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
<p align = "center">
    <img  src = "https://raw.githubusercontent.com/Me-ghana/Me-ghana.github.io/master/images/CaffeConverter/compile.png">
</p> 


**Step 5: Execution with CaffeNet Model**

You can execute with the following command:
```c++
/caffe_converter.bin [model-file] [trained-file] [mean-file] [label-file] [img-file]
```
Check that everything is working properly by using the files from the <a href = "https://github.com/BVLC/caffe/tree/master/examples/cpp_classification" target = "_blank">pre-trained CaffeNet model</a>.  When you install and compile Caffe, the following files will be automaticaly built in the "examples" folder.  However, if you would just like to download these files directly, click on the links below.  Save these files in your current working directory, the path name should end in tiny_dnn/io/caffe.

<a href = "https://github.com/Me-ghana/old-site/blob/master/caffeConvert/deploy.prototxt" download>deploy.prototxt</a> 
<a href = "http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel" download>bvlc_reference_caffenet.caffemodel </a>
<a href = "https://github.com/Me-ghana/old-site/blob/master/caffeConvert/imagenet_mean.binaryproto" download>imagenet_mean.binaryproto</a>
<a href = "https://github.com/Me-ghana/old-site/blob/master/caffeConvert/synset_words.txt" download>synset_words.txt</a> 
<a href="https://raw.githubusercontent.com/Me-ghana/old-site/master/caffeConvert/cat.jpg" download>cat.jpg</a> 

```c++
./caffe_converter.bin \ 
deploy.prototxt \ 
bvlc_reference_caffenet.caffemodel \ 
imagenet_mean.binaryproto \ 
synset_words.txt \ 
cat.jpg 
```

<p align = "center">
<img  src = "https://raw.githubusercontent.com/Me-ghana/Me-ghana.github.io/master/images/CaffeConverter/run.png">
</p> 

You should see the final output: <br>
```c++
---------- Prediction for examples/images/cat.jpg ----------
0.3134 - "n02123045 tabby, tabby cat"
0.2380 - "n02123159 tiger cat"
0.1235 - "n02124075 Egyptian cat"
0.1003 - "n02119022 red fox, Vulpes vulpes"
0.0715 - "n02127052 lynx, catamount"
```
<p align = "center">    
	<img  src = "https://raw.githubusercontent.com/Me-ghana/old-site/master/caffeConvert/results.png">
</p> 

You can have some fun testing CaffeNet's accuracy on different images.  The CaffeNet model has a <a href = "https://github.com/BVLC/caffe/tree/master/models/bvlc_reference_caffenet" target = "_blank">80% top-5 accuracy</a> on their validation set.  Take a look at their categories in their synset.txt file and try testing the model on various photos.  Here's how CaffeNet did on this photo of a sloth bear: 

<p align = "center">
    <img  src = "https://raw.githubusercontent.com/Me-ghana/old-site/master/caffeConvert/slothbear.png"> <img  src = "https://raw.githubusercontent.com/Me-ghana/old-site/master/caffeConvert/Sloth-Bear.png">
</p> 


**Step 6: Execution with Places365-CNN Model**
The next step is to use the converter on the actual Caffe model we want to convert.  I am using <a href = "https://github.com/CSAILVision/places365" target = "_blank">MIT's Places365-CNN Caffe model</a>. There are several CNNs - we'll try out Alexnet. You will need the following files (from Places365 GitHub):

<a href = "https://github.com/Me-ghana/old-site/tree/master/caffeConvert/deploy_alexnet_places365.prototxt" download>deploy_alexnet_places365.prototxt</a> <br>
<a href = "http://places2.csail.mit.edu/models_places365/googlenet_places365.caffemodel" download>alexnet_places365.caffemodel </a> <br>
<a href = "https://github.com/Me-ghana/old-site/tree/master/caffeConvert/places365CNN_mean.binaryproto" download>places365_mean.binaryproto</a><br>
<a href = "https://github.com/Me-ghana/old-site/tree/master/caffeConvert/labels_sunattribute.txt" download>labels_sunattribute.txt</a> <br>
<a href="https://github.com/Me-ghana/old-site/tree/master/caffeConvert/beach.jpg" download>beach.jpg</a> <br>

When we attempt to use these files, we get an "input shape not found in caffemodel" error message.
<p align = "center">
<img  src = "https://raw.githubusercontent.com/Me-ghana/old-site/master/caffeConvert/inputshape.png"> 
</p>
Currently, the input layer in the deploy_alexnet_places365.prototxt file looks like:
<p align = "center">
<img  src = "https://raw.githubusercontent.com/Me-ghana/old-site/master/caffeConvert/beforeLayerChange.png"> 
</p> 
We can fix this error by changing the input layer to this:   
<p align = "center">
<img  src = "https://raw.githubusercontent.com/Me-ghana/old-site/master/caffeConvert/afterLayerChange.png"> 
</p> 

When running this file, the next error we get is "root layer not found".
<p align = "center">
<img  src = "https://raw.githubusercontent.com/Me-ghana/old-site/master/caffeConvert/error1.png"> 
</p>
    

If we look at the source code, we see the error is thrown in layer_factory_impl.h Line 955, when the root is unable to find the layer node.  Working backwards, this happens when the function reload_weight_from_caffe_protobinary() from the file layer_factory.h executes reload_weight_from_caffe_net(). This calls the function caffe_layer_vector() in the file layer_factory_impl.h. (Function arguments omitted for clarity.)  Since we changed the prototxt file, we may need to recreate the binary file... more to come! (Please let me know if someone has a solution for this in the meantime) 


Any feedback to improve this material is much appreciated! Please email meghanak@usc.edu.
