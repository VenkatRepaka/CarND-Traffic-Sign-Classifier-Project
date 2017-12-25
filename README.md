## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, you will use what you've learned about deep neural networks and convolutional neural networks to classify traffic signs. You will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, you will then try out your model on images of German traffic signs that you find on the web.

We have included an Ipython notebook that contains further instructions 
and starter code. Be sure to download the [Ipython notebook](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb). 

We also want you to create a detailed writeup of the project. Check out the [writeup template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.

To meet specifications, the project will require submitting three files: 
* the Ipython notebook with the code
* the code exported as an html file
* a writeup report either as a markdown or pdf file 

Creating a Great Writeup
---
A great writeup should include the [rubric points](https://review.udacity.com/#!/rubrics/481/view) as well as your description of how you addressed each point.  You should include a detailed description of the code used in each step (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup.

The Project
---
The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

### Dataset and Repository

1. Download the data set. The classroom has a link to the data set in the "Project Instructions" content. This is a pickled dataset in which we've already resized the images to 32x32. It contains a training, validation and test set.
2. Clone the project, which contains the Ipython notebook and the writeup template.
```sh
git clone https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project
cd CarND-Traffic-Sign-Classifier-Project
jupyter notebook Traffic_Sign_Classifier.ipynb
```

### Requirements for Submission
Follow the instructions in the `Traffic_Sign_Classifier.ipynb` notebook and write the project report using the writeup template as a guide, `writeup_template.md`. Submit the project code and writeup document.

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).


### Data Set Summary & Exploration

1. ##### Basic summary of available traffic signs classifier data:
    - The size of training set is 34799
    - The size of the validation set is 4410
    - The size of test set is 12630
    - The shape of a traffic sign image is 32, 32, 3
    - The number of unique classes/labels in the data set is 43
    
2. ##### Exploratory visualization of the dataset.

    Randomly picked image for each class from training dataset
![](https://github.com/VenkatRepaka/CarND-Traffic-Sign-Classifier-Project/blob/master/documentation/sample_images.png)

    Below histograms show the number of samples given for each class in the dataset
![](https://github.com/VenkatRepaka/CarND-Traffic-Sign-Classifier-Project/blob/master/documentation/Original_Training_Data.png)
![](https://github.com/VenkatRepaka/CarND-Traffic-Sign-Classifier-Project/blob/master/documentation/Validation_Data.png)
![](https://github.com/VenkatRepaka/CarND-Traffic-Sign-Classifier-Project/blob/master/documentation/Test_Data.png)


### Design and Test a Model Architecture

##### Preprocessing images

1. I have converted all images to gray scale. It is mentioned in the technical paper that they could achieve accuracy above 99%. 
    Randomly picked image for each class from training dataset which are changed to gray scale images using open cv
![](https://github.com/VenkatRepaka/CarND-Traffic-Sign-Classifier-Project/blob/master/documentation/gray_sale_images.png)

2. The distribution of number of images for each class is disproportionate. So I have generated augmented data till the samples reach numbers near 1500 for
each class. For data augmentation I have applied 4 steps. I have used open cv for augmentation.
    - Rotation of image
    - Change of perspective of image
    - Shift image centre
    - Add nois using np.random.normal
    
    Sample of the augmented data
![](https://github.com/VenkatRepaka/CarND-Traffic-Sign-Classifier-Project/blob/master/documentation/augmented_sample.png)

Histogram of training data after generating augmented data.
![](https://github.com/VenkatRepaka/CarND-Traffic-Sign-Classifier-Project/blob/master/documentation/Augmented_Training_Data.png)

##### Model Architecture

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 grayscale image   							|
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 2x2 stride, valid padding, outputs 10x10x16    |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Convolution 1x1	    | 2x2 stride, valid padding, outputs 1x1x412    |
| RELU					|												|
| Fully connected		| input 412, output 122        									|
| RELU					|												|
| Dropout				| 50% keep        									|
| Fully connected		| input 122, output 84        									|
| RELU					|												|
| Dropout				| 50% keep        									|
| Fully connected		| input 84, output 43        									|
