# Cityscapes

## Data
http://www.mimuw.edu.pl/~ciebie/cityscapes.tgz

## Task
You are supposed to implement a fully convolutional U-net like neural network for segmenting
pixels into 30 categories. The data contains 3475 images of __256 * 256 size__ together with
annotations. Your are not allowed to copy code from online resources. You should implement your
solution in __PyTorch__ (if you insist on using __Tensorflow__ you may ask your lab teacher
for permission).

Requested features (to reach max score):
* split the training data into train and validation randomly,
* training done on GPU,
* data augmentation: horizontal flips,
* when doing predictions on the validation set, average augmented versions of an image and rate
 the averaged predictions.

A necessary condition to get half of points is to reach __50% accuracy__ on your validation set.
Note that 50% is not a great score, if your program does not reach 50% it most likely means you
have done something wrong.

It would be nice to have a visualization of learning process (eg. using TensorBoard).

## Objective Function
For each image in your validation set compute the average accuracy of prediction over all pixels,
then compute the mean score over all images.

## Logs
You should save the logs from your run, containing train and validation error for subsequent epochs
as well as timestamps. Also, you should store a checkpoint to show the validation error during code
inspection. Finally, you should be able to visualize predictions on the validation set.

## Deadline
You should submit your solution by email by 23:59 on 14.05.2019 (Tuesday) to your lab teacher with
email title Assignment 2 - Deep neural networks. Your code will be inspected during the lab session
following the deadline. Note that even if you are one minute late after the deadline, your solution
will not be inspected. We have no mercy whatsover so you better not count on that.