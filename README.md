# **Traffic Sign Recognition** 

## Writeup
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

**Dependencies**
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.


[//]: # (Image References)

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/grayscale.png "Grayscaling"
[image3]: ./examples/original.png "Original"
[image4]: ./test_images/13_yield.jpg "Traffic Sign: Yield"
[image5]: ./test_images/3_speed_limit_60.jpg "Traffic Sign: Speed 60"
[image6]: ./test_images/14_stop.jpg "Traffic Sign: Stop"
[image7]: ./test_images/19_dangerous_curve_to_left.jpg "Traffic Sign: Dangerous curve (left)"
[image8]: ./test_images/35_ahead_only.jpg "Traffic Sign: Ahead Only"
[image9]: ./examples/table.png "Data Distribution"

## Rubric Points

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](./Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distribute according to each lable. The table is showing the detailed specification of the dataset.

![alt text][image1]


![alt text][image9]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale and normalize to reduce calculation cost by simpliying the features.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image3]
![alt text][image2]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:----------------------|:----------------------------------------------| 
| Input         		| 32x32x1 grayscaled image   				    | 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6		    		|
| Convolution 5x5	    | 1x1 stride, same padding, outputs 10x10x16    |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16		    		|
| Flatten   	      	| outputs 400               		    		|
| Dropout       		|         							    		|
| Fully connected		| outputs 120        							|
| RELU					|												|
| Dropout       		|         								    	|
| Fully connected		| outputs 84        							|
| RELU					|												|
| Dropout       		|         								    	|
| Fully connected		| outputs 43        							|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer. The hyperparameters are:

| Parameter                             |     Value	        | 
|:--------------------------------------|:------------------| 
| Epoch                              	| 50        	    | 
| Batch size                            | 128        		| 
| Learning rate                         | 0.001        		| 
| mu for `tf.truncated_normal`          | 0        			| 
| sigma for `tf.truncated_normal`       | 0.1        		| 
| Keep probability for dropout layer    | 0.5        		| 


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.00
* validation set accuracy of 0.954
* test set accuracy of 0.937

Here's some history on how I reached out the current design.
* I chosed the LeNet model from the Udacity course material as a starting point simply because I am familiarized with it through the course.
* Initial problem was the low accuracy with around 0.800. I tried to finetune the hyperparameters but it didn't work well and improved only about 0.01 to 0.03.
* I decided to introduce dropout layers and tweaked the layout (where to insert) and keep probablity parameter. It improved the accuracy up to 0.850.
* What brought the significant improvement was change of the grayscale methodology in preprocessing. I used cv2.`cvtColor(image, cv2.COLOR_BGR2GRAY)` first. When I started to use `np.sum(images/3, axis=3, keepdims=True)` instead according to a [Q&A in Udacity Knowledgebase](https://knowledge.udacity.com/questions/358946), it improved the accuracy with above 0.930. Old implementation is left in the 5th cell with comment-out.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:----------------------|:----------------------------------------------| 
| Yield					| Yield											| 
| 60 km/h	      		| 60 km/h					     				|
| Stop Sign      		| Stop sign   									| 
| Dangerous curve (left)| Dangerous curve (left)                    	|
| Keep Straight			| Keep Straight      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 18th cell of the Ipython notebook. The model is very accurate and confident with the provided test images with almost 100% of probabilities.


