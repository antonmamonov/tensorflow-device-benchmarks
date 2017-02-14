import tensorflow as tf
import numpy as np
import time

#diffWeights 98.7484
#time 14.6911480427

totalTrainingSet = 10000
inputLength = 1000
outputLength = 100
learningRate = 0.0001
epoch = 5

# setup the actual weights
#actualWeights = np.ones((1,outputLength,inputLength)) * 0.25

#for n,i in enumerate(actualWeights):
#    for nn,ii in enumerate(actualWeights[n]):
#        if nn % 2 == 0:
#            actualWeights[n][nn] = -0.25

with tf.device('/gpu:1'):

    # Generate training data
    
    actualWTensor = tf.multiply(tf.ones((outputLength,inputLength), dtype=tf.float32),0.25)
    #tf.constant(actualWeights,'float32')
    
    trX = tf.random_normal([totalTrainingSet,inputLength,1],stddev=0.5)
    #np.random.rand(totalTrainingSet,inputLength,1) - 0.5
    
    trY = tf.map_fn(lambda singleX: tf.matmul(actualWTensor,singleX),trX)    
    #np.array(map(lambda x: np.matmul(actualWeights,x)[0],trX))

    # Set model weights
    W = tf.Variable(tf.random_normal([outputLength,inputLength], stddev=0.75, name="weights"))

    PredY = tf.map_fn(lambda singleX: tf.multiply(tf.multiply(tf.matmul(W,singleX),tf.matmul(W,singleX)),tf.multiply(tf.matmul(W,singleX),tf.matmul(W,singleX))),trX)
    loss = tf.reduce_sum(tf.square(PredY-trY))
                
    optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(loss)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    init = tf.global_variables_initializer()
    sess.run(init)

    def diffWeights():
        return sess.run(tf.reduce_sum(tf.abs(tf.subtract(actualWTensor, W))))

    print "diffWeights",diffWeights()
    startTime = time.time()
    for i in range(epoch):
        _, loss_val = sess.run([optimizer,loss])
        if i % 5 == 0:
            print loss_val
    print "diffWeights",diffWeights()
    print "time",(time.time() - startTime)
