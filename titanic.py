# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 15:24:55 2017

@author: E605416
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer

gender_submission = pd.read_csv('gender_submission.csv')
test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')
y = train['Survived']

#imp = Imputer(missing_values='NaN', strategy ='mean', axis=0)
X = train.copy()
del X['Survived']
del X['Embarked']

#%% Preprocessing
def preprocess(X):
    del X['Name']
    del X['Ticket']
    #set nan to -1
    X = X.fillna(value =-1)

    #Sets cabins to 1,2,...,7 depending on class
    for index, entry in X['Cabin'].iteritems():
        entry = str(entry)
        if 'A' in entry:
            X['Cabin'][index] = 1
        elif 'B' in entry:
            X['Cabin'][index] = 2
        elif 'C' in entry:
            X['Cabin'][index] = 3
        elif 'D' in entry:
            X['Cabin'][index] = 4
        elif 'E' in entry:
            X['Cabin'][index] = 5 
        elif 'F' in entry:
            X['Cabin'][index] = 6
        elif 'G' in entry:
            X['Cabin'][index] = 7  
        elif 'T' in entry:
            X['Cabin'][index] = 8         
         
    #Changes sex to binary 0 = male, 1 = female
    for index, entry in X['Sex'].iteritems():   
        if 'male' in entry:
          X['Sex'][index] = 0
        if 'female' in entry:
          X['Sex'][index] = 1   
    return(X)
#%% preprocess training and test.
X = preprocess(X)
t = test.copy()
t = preprocess(t)
#%% Random forest classifier


#imp = imp.fit(X)

clf = RandomForestClassifier(max_depth = 5, random_state =0)

X_imp = imp.transform(X)
clf.fit(X_imp,y)

print(clf.feature_importances_) 
print(clf.predict(X_imp))
final = clf.predict(X_imp)

final_series = pd.DataFrame(final)
final_series.columns = ['Survived']
df = test['PassengerId']
df = df[:-1]
df['Survived'] = final
final = df.assign(final_series)

a = pd.DataFrame()
a['PassengerId'] = df
a['Survived'] = final_series['Survived']

predicitons = a

predicitons.to_csv('predictions.csv', index = False)


#%% Neural Network TensorFlow
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
training_data_df = pd.read_csv("train.csv")

X_training = training_data_df.drop('Survived',axis=1)
Y_training = training_data_df[['Survived']]

del X_training['Embarked']
X_training = preprocess(X_training)

test_data_df = pd.read_csv("test.csv")
X_testing = test_data_df.copy()
del X_testing['Embarked']
X_testing = preprocess(X_testing)

X_scaler = MinMaxScaler(feature_range=(0,1))
Y_scaler = MinMaxScaler(feature_range=(0,1))

X_scaled_training = X_scaler.fit_transform(X_training)
Y_scaled_training = Y_scaler.fit_transform(Y_training)

X_scaled_testing = X_scaler.transform(X_testing)

# Define model parameters
learning_rate = 0.001
training_epochs = 100

# Define how many inputs and outputs are in our neural network
number_of_inputs = 8
number_of_outputs = 1

# Define how many neurons we want in each layer of our neural network
layer_1_nodes = 50
layer_2_nodes = 100
layer_3_nodes = 50

# Section One: Define the layers of the neural network itself

# Input Layer
with tf.variable_scope('input'):
    X = tf.placeholder(tf.float32, shape=(None, number_of_inputs))
# Layer 1
with tf.variable_scope('layer_1'):
    weights = tf.get_variable("weights1", shape=[number_of_inputs, layer_1_nodes], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases1", shape=[layer_1_nodes], initializer=tf.zeros_initializer())
    layer_1_output = tf.nn.relu(tf.matmul(X, weights) + biases)

# Layer 2
with tf.variable_scope('layer_2'):
    weights = tf.get_variable("weights2", shape=[layer_1_nodes, layer_2_nodes], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases2", shape=[layer_2_nodes], initializer=tf.zeros_initializer())
    layer_2_output = tf.nn.relu(tf.matmul(layer_1_output, weights) + biases)

# Layer 3
with tf.variable_scope('layer_3'):
    weights = tf.get_variable("weights3", shape=[layer_2_nodes, layer_3_nodes], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases3", shape=[layer_3_nodes], initializer=tf.zeros_initializer())
    layer_3_output = tf.nn.relu(tf.matmul(layer_2_output, weights) + biases)

# Output Layer
with tf.variable_scope('output'):
    weights = tf.get_variable("weights4", shape=[layer_3_nodes, number_of_outputs], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases4", shape=[number_of_outputs], initializer=tf.zeros_initializer())
    prediction = tf.matmul(layer_3_output, weights) + biases

# Section Two: Define the cost function of the neural network that will measure prediction accuracy during training

with tf.variable_scope('cost'):
    Y = tf.placeholder(tf.float32, shape=(None, 1))
    cost = tf.reduce_mean(tf.squared_difference(prediction, Y))

# Section Three: Define the optimizer function that will be run to optimize the neural network

with tf.variable_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Initialize a session so that we can run TensorFlow operations
with tf.Session() as session:

    # Run the global variable initializer to initialize all variables and layers of the neural network
    session.run(tf.global_variables_initializer())

    # Run the optimizer over and over to train the network.
    # One epoch is one full run through the training data set.
    for epoch in range(training_epochs):

        # Feed in the training data and do one step of neural network training
        session.run(optimizer, feed_dict={X: X_scaled_training, Y: Y_scaled_training})

        # Every 5 training steps, log our progress
        if epoch % 5 == 0:
            training_cost = session.run(cost, feed_dict={X: X_scaled_training, Y: Y_scaled_training})
            print(epoch, training_cost)
    # Training is now complete!
    print("Training is complete!")
    final_training_cost = session.run(cost, feed_dict={X: X_scaled_training, Y: Y_scaled_training})
    
    print('Final Training cost: {}'.format(final_training_cost))      


    # Now that the neural network is trained, let's use it to make predictions for our test data.
    # Pass in the X testing data and run the "prediciton" operation
    Y_predicted_scaled = session.run(prediction, feed_dict={X: X_scaled_testing})
    # Unscale the data back to it's original units (dollars)
    Y_predicted = Y_scaler.inverse_transform(Y_predicted_scaled)

res = np.round(Y_predicted)
df = pd.DataFrame(res)
df.columns=['Survived']
df2 = X_testing[['PassengerId']]

final = pd.concat([df2,df],axis=1)


final.to_csv('tensorFlow_prediction.csv',index=False)


#Defining all columns of interest
feature_columns = [tf.feature_column.numeric_column('PassengerId',shape=[1]), 
                   tf.feature_column.numeric_column('Pclass', shape = [1]),
                   tf.feature_column.numeric_column('Sex', shape = [1]),
                   tf.feature_column.numeric_column('Age', shape = [1]),
                   tf.feature_column.numeric_column('SibSp', shape = [1]),
                   tf.feature_column.numeric_column('Parch', shape = [1]),
                   tf.feature_column.numeric_column('Fare', shape = [1]),
                   tf.feature_column.numeric_column('Cabin', shape = [1])]

estimator = tf.estimator.DNNClassifier(feature_columns = feature_columns, hidden_units = [1024, 512, 256])

x_train = X
y_train = y
x_eval = t

input_fn = tf.estimator.inputs.pandas_input_fn({'PassengerId': x_train['PassengerId'],
                                               'Pclass': x_train['Pclass'],
                                               'Sex': x_train['Sex'],
                                               'Age': x_train['Age'],
                                               'SibSp': x_train['SibSp'],
                                               'Parch': x_train['Parch'],
                                               'Fare': x_train['Fare'],
                                               'Cabin': x_train['Cabin'],},
                                                y_train, 
                                                batch_size = 4, 
                                                num_epochs=None, 
                                                shuffle=True)

input_fn = tf.estimator.inputs.pandas_input_fn(X, y_train, batch_size = 4, num_epochs=None, shuffle=True)
input_fn = tf.estimator.inputs.pandas_input_fn(train, train['Survived'], batch_size = 4, num_epochs=None, shuffle=True)
estimator.train(input_fn=input_fn, steps=1000)

print("test")
print("branching")
