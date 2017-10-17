**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_ex.jpg
[image2]: ./output_images/noncar_ex.jpg

[image3]: ./output_images/HOG_car_ex.jpg
[image4]: ./output_images/HOG_noncar_ex.jpg

[image5]: ./output_images/not_thresholded0.jpg
[image6]: ./output_images/not_thresholded1.jpg
[image7]: ./output_images/not_thresholded2.jpg
[image8]: ./output_images/not_thresholded3.jpg
[image9]: ./output_images/not_thresholded4.jpg
[image10]: ./output_images/not_thresholded5.jpg

[image11]: ./output_images/thresholded0.jpg
[image12]: ./output_images/thresholded1.jpg
[image13]: ./output_images/thresholded2.jpg
[image14]: ./output_images/thresholded3.jpg
[image15]: ./output_images/thresholded4.jpg
[image16]: ./output_images/thresholded5.jpg

[video1]: ./test_video_out_without_frame.mp4
[video2]: ./test_video_out_with_frame.mp4
[video3]: ./project_video_out.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first 12 code cells of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]
![alt text][image2]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `Y` color space of `YUV` and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image3]
![alt text][image4]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and found that `YUV` color space is the best choice to detect a car. Using YUV color space makes fewer false positives when detection.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVC (Support Vector Classifier). 
When the test set was applied, the accuracy was 98.56%.
All data were shuffled when learning and flipped left and right to augment the data twice.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Only the area where y is between 400 and 600, which is the area where the car is likely to be, is performed.
From the small boxes to the large boxes, I performed a window search with 12 different sized boxes.

I used only five types of boxes to reduce the amount of computation, but I could not distinguish between cars. So I use 12 different boxes to increase the amount of computation and perform proper detection.

####2. Show some examples of test images to demonstrate how your pipeline is working.

These pictures are the output of the pictures obtained from the window search. There are many large and small boxes around the car.
![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]

These boxes draw out more than two overlapping areas at the same time and draw the area again.
![alt text][image11]
![alt text][image12]
![alt text][image13]
![alt text][image14]
![alt text][image15]
![alt text][image16]

### Video Implementation

####1. Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

The output of test video without frame operation : [link to my video result](./test_video_out_without_frame.mp4)
 
The output of test video with frame operation : [link to my video result](./test_video_out_with_frame.mp4)
 
The output of project video : [link to my video result](./project_video_out.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each last 3 frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected.  

In the example images, when detecting the vehicle, we found the overlap of two or more boxes, but in the video, the positive detection values of the previous 3 frames were entered, so the threshold value of 7 was the optimal value.


###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The vehicle is well detected, but there is too much computation. And, it can not detect other objects.
It would be better to use deep_learning to use classifier in autonomous driving. If I use deep_learning, I will be able to identify more objects and be more accurate and faster.
tiny yolo v2, squeezedet, PVANet seems to be light and good.
