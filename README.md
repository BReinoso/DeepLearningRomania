# DeepLearningRomania
Report and Implementation for Methauristic Algorithms Project in Romania

This project is only a report for a subject in Romania and some test applications.
In this 'project' I am doing my first steps in theano and Deep Learning implementations.
To test the model I used the dataset: http://archive.ics.uci.edu/ml/datasets/Heart+Disease

There are probably a lot of mistakes in my code and report but, since I am starting in this topic, I hope you can forgive me for that.

# Model
The model is a simple Neural Network with 3 layers. The first one with 10 neurons, the second one with 5 neurons and the last one with 2 neurons. In order to predict two classes.

The update of the Thetas is doing like in Stochastic Gradient Descent, we made an update with each exmple. Because we are using the sigmoid function and we expect values 0 or 1, the cost function is calculated with the same formula that is used in Logistic Regression.
# Results
The training is done very quickly, from 8 to 20 Epochs. The better succes rate obtained with the architecture was 83% of hit rate in the test set.

That's all. I'll be adding more information and, probably, more implementations.
Greetings.
