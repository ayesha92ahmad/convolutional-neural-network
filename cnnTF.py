import os
import time
import math
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load the CIFAR10 dataset
from keras.datasets import cifar10
baseDir = os.path.dirname(os.path.abspath('__file__')) + '/'
classesName = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
(xTrain, yTrain), (xTest, yTest) = cifar10.load_data()
xVal = xTrain[49000:, :].astype(np.float)
yVal = np.squeeze(yTrain[49000:, :])
xTrain = xTrain[:49000, :].astype(np.float)
yTrain = np.squeeze(yTrain[:49000, :])
yTest = np.squeeze(yTest)
xTest = xTest.astype(np.float)

# Show dimension for each variable
print ('Train image shape:    {0}'.format(xTrain.shape))
print ('Train label shape:    {0}'.format(yTrain.shape))
print ('Validate image shape: {0}'.format(xVal.shape))
print ('Validate label shape: {0}'.format(yVal.shape))
print ('Test image shape:     {0}'.format(xTest.shape))
print ('Test (label shape:     {0}'.format(yTest.shape))

# Pre processing data
# Normalize the data by subtract the mean image
meanImage = np.mean(xTrain, axis=0)
xTrain -= meanImage
xVal -= meanImage
xTest -= meanImage

# Select device
deviceType = "/cpu:0"

# Simple Model
tf.reset_default_graph()
with tf.device(deviceType):
    x = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y = tf.placeholder(tf.int64, [None])
def simpleModel():
    with tf.device(deviceType):
        wConv = tf.get_variable("wConv", shape=[7, 7, 3, 32])
        bConv = tf.get_variable("bConv", shape=[32])
        w = tf.get_variable("w", shape=[5408, 10]) # Stride = 2, ((32-7)/2)+1 = 13, 13*13*32=5408
        b = tf.get_variable("b", shape=[10])

        # Define Convolutional Neural Network
        a = tf.nn.conv2d(x, wConv, strides=[1, 2, 2, 1], padding='VALID') + bConv # Stride [batch, height, width, channels]
        h = tf.nn.relu(a)
        hFlat = tf.reshape(h, [-1, 5408]) # Flat the output to be size 5408 each row
        yOut = tf.matmul(hFlat, w) + b

        # Define Loss
        totalLoss = tf.losses.hinge_loss(tf.one_hot(y, 10), logits=yOut)
        meanLoss = tf.reduce_mean(totalLoss)


        # Define Optimizer
        optimizer = tf.train.AdamOptimizer(5e-4)
        trainStep = optimizer.minimize(meanLoss)

        # Define correct Prediction and accuracy
        correctPrediction = tf.equal(tf.argmax(yOut, 1), y)
        accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))

        return [meanLoss, accuracy, trainStep]

def train(Model, xT, yT, xV, yV, xTe, yTe, batchSize=1000, epochs=100, printEvery=10):
    # Train Model
    trainIndex = np.arange(xTrain.shape[0])
    np.random.shuffle(trainIndex)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(epochs):
            # Mini-batch
            losses = []
            accs = []
            # For each batch in training data
            for i in range(int(math.ceil(xTrain.shape[0] / batchSize))):
                # Get the batch data for training
                startIndex = (i * batchSize) % xTrain.shape[0]
                idX = trainIndex[startIndex:startIndex + batchSize]
                currentBatchSize = yTrain[idX].shape[0]

                # Train
                loss, acc, _ = sess.run(Model, feed_dict={x: xT[idX, :], y: yT[idX]})

                # Collect all mini-batch loss and accuracy
                losses.append(loss * currentBatchSize)
                accs.append(acc * currentBatchSize)

            totalAcc = np.sum(accs) / float(xTrain.shape[0])
            totalLoss = np.sum(losses) / xTrain.shape[0]
            if e % printEvery == 0:
                print('Iteration {0}: loss = {1:.3f} and training accuracy = {2:.2f}%,'.format(e, totalLoss, totalAcc * 100), end='')
                loss, acc = sess.run(Model[:-1], feed_dict={x: xV, y: yV})
                print(' Validate loss = {0:.3f} and validate accuracy = {1:.2f}%'.format(loss, acc * 100))

        loss, acc = sess.run(Model[:-1], feed_dict={x: xTe, y: yTe})
        print('Testing loss = {0:.3f} and testing accuracy = {1:.2f}%'.format(loss, acc * 100))

# Start training simple model
print("\n################ Simple Model #########################")
train(simpleModel(), xTrain, yTrain, xVal, yVal, xTest, yTest)

# Complex Model
tf.reset_default_graph()
with tf.device(deviceType):
    x = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y = tf.placeholder(tf.int64, [None])
def complexModel():
    with tf.device(deviceType):
        #############################################################################
        # TODO: 40 points                                                           #
        # - Construct model follow below architecture                               #
        #       7x7 Convolution with stride = 2                                     #
        #       Relu Activation                                                     #
        #       2x2 Max Pooling                                                     #
        #       Fully connected layer with 1024 hidden neurons                      #
        #       Relu Activation                                                     #
        #       Fully connected layer to map to 10 outputs                          #
        # - Store last layer output in yOut                                         #
        #############################################################################

        #       7x7 Convolution with stride = 2
        wConv = tf.get_variable("wConv", shape=[7, 7, 3, 64])
        bConv = tf.get_variable("bConv", shape=[64])

        # Define Convolutional Neural Network
        a = tf.nn.conv2d(x, wConv, strides=[1,2,2,1], padding='VALID') + bConv # Stride [batch, height, width, channels]
        #       Relu Activation
        h = tf.nn.relu(a)
        #       2x2 Max Pooling
        max_pool_output = tf.nn.max_pool(value=h, ksize=[1,2,2,1], strides=[1,2,2,1],padding='VALID')
        hFlat = tf.reshape(max_pool_output, [-1, 6*6*64]) # Flat the output to be size 6*6*64 each row

        #       Fully connected layer with 1024 hidden neurons
        w_fully_h = tf.get_variable("w_fully_h", shape=[6*6*64, 1024])
        b_fully_h = tf.get_variable("b_fully_h", shape=[1024])

        fully_connected_output = tf.matmul(hFlat, w_fully_h) + b_fully_h
        #       Relu Activation
        relu_out =  tf.nn.relu(fully_connected_output)
        relu_outFlat = tf.reshape(relu_out, [-1, 1024]) # Flat the output to be size 1024 each row
        #       Fully connected layer to map to 10 outputs
        w = tf.get_variable("w", shape=[1024,10])
        b = tf.get_variable("b", shape=[10])
        # - Store last layer output in yOut
        yOut = tf.matmul(relu_outFlat, w) + b
        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################

        # Define Loss
        totalLoss = tf.losses.hinge_loss(tf.one_hot(y, 10), logits=yOut)
        meanLoss = tf.reduce_mean(totalLoss)

        # Define Optimizer
        optimizer = tf.train.AdamOptimizer(5e-4)
        trainStep = optimizer.minimize(meanLoss)

        # Define correct Prediction and accuracy
        correctPrediction = tf.equal(tf.argmax(yOut, 1), y)
        accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))

        return [meanLoss, accuracy, trainStep]

# Start training complex model
print("\n################ Complex Model #########################")
train(complexModel(), xTrain, yTrain, xVal, yVal, xTest, yTest)

# Your Own Model
tf.reset_default_graph()
with tf.device(deviceType):
    x = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y = tf.placeholder(tf.int64, [None])
def yourOwnModel():
    with tf.device(deviceType):
        #############################################################################
        # TODO: 60 points                                                           #
        # - Construct your own model to get validation accuracy > 70%               #
        # - Store last layer output in yOut                                         #
        #############################################################################
        #       7x7 Convolution with stride = 2
        wConv1 = tf.get_variable("wConv1", shape=[7, 7, 3, 32])
        bConv1 = tf.get_variable("bConv1", shape=[32])

        # Define Convolutional Neural Network
        a = tf.nn.conv2d(x, wConv1, strides=[1,2,2,1], padding='SAME') + bConv1 # Stride [batch, height, width, channels]
        #       Relu Activation
        h = tf.nn.relu(a)

        #       2x2 Max Pooling
        max_pool_output = tf.nn.max_pool(value=h, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')
        #       5x5 Convolution with stride = 2
        wConv2 = tf.get_variable("wConv2", shape=[5, 5, 32, 64])
        bConv2 = tf.get_variable("bConv2", shape=[64])

        # Define Convolutional Neural Network
        a = tf.nn.conv2d(max_pool_output, wConv2, strides=[1,1,1,1], padding='SAME') + bConv2 # Stride [batch, height, width, channels]
        #       Relu Activation
        h = tf.nn.relu(a)
        #       2x2 Max Pooling
        max_pool_output = tf.nn.max_pool(value=h, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')
        hFlat = tf.reshape(max_pool_output, [-1, 4*4*64]) # Flat the output to be size 6*6*64 each row

        #       Fully connected layer with 1024 hidden neurons
        w_fully_h = tf.get_variable("w_fully_h", shape=[ 4*4*64, 1024])
        b_fully_h = tf.get_variable("b_fully_h", shape=[1024])

        fully_connected_output = tf.matmul(hFlat, w_fully_h) + b_fully_h
        #       Relu Activation
        relu_out =  tf.nn.relu(fully_connected_output)
        relu_outFlat = tf.reshape(relu_out, [-1, 1024]) # Flat the output to be size 1024 each row
        #       Fully connected layer to map to 128 outputs
        w1 = tf.get_variable("w1", shape=[1024,128])
        b1 = tf.get_variable("b1", shape=[128])

        fully_connected_output= tf.matmul(relu_outFlat, w1) + b1
        relu_out =  tf.nn.relu(fully_connected_output)
        relu_outFlat = tf.reshape(relu_out, [-1, 128]) # Flat the output to be size 128 each row
        relu_outFlat = tf.nn.dropout(relu_outFlat, 0.5) #To avoid overfitting
        #       Fully connected layer to map to 10 outputs
        w2 = tf.get_variable("w2", shape=[128,10])
        b2 = tf.get_variable("b2", shape=[10])
        # - Store last layer output in yOut
        yOut = tf.matmul(relu_outFlat, w2) + b2
        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################

        # Define Loss
        totalLoss = tf.losses.hinge_loss(tf.one_hot(y, 10), logits=yOut)
        meanLoss = tf.reduce_mean(totalLoss)

        # Define Optimizer
        optimizer = tf.train.AdamOptimizer(5e-4)
        trainStep = optimizer.minimize(meanLoss)

        # Define correct Prediction and accuracy
        correctPrediction = tf.equal(tf.argmax(yOut, 1), y)
        accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))

        return [meanLoss, accuracy, trainStep]

# Start your own Model model
print("\n################ Your Own Model #########################")
#########################################################################
# TODO: 0 points                                                        #
# - You can set your own batchSize and epochs                           #
#########################################################################
train(yourOwnModel(), xTrain, yTrain, xVal, yVal, xTest, yTest, batchSize=256, epochs=200, printEvery=10)
#########################################################################
#                       END OF YOUR CODE                                #
#########################################################################
