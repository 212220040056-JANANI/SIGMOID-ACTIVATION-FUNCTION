### EX NO: 05
### DATE: 18/04/2022
# <p align= "center">SIGMOID ACTIVATION FUNCTION</p>
## AIM:
  To develop a python code that creates a simple feed-forward neural networks or perception with the Sigmoid activation function. The neuron has to be trained such that it can predict the correct output value when provided with a new set of input data.
  
 ![image](https://user-images.githubusercontent.com/93023609/162692440-f59e7ad2-0414-4ddb-8640-fede7a0655f2.png)

## EQUIPMENTS REQUIRED:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner / Google Colab

## RELATED THEORETICAL CONCEPT:
Sigmoid activation function:

σ(x) = 1/1 + e−κx 

The main reason why we use sigmoid function is because it exists between (0 to 1). Therefore, it is especially used for models where we have to predict the probability as an output.Since probability of anything exists only between the range of 0 and 1, sigmoid is the right choice.The function is differentiable.That means, we can find the slope of the sigmoid curve at any two points.The function is monotonic but function’s derivative is not.The logistic sigmoid function can cause a neural network to get stuck at the training time. Sigmoid Function acts as an activation function in machine learning which is used to add non-linearity in a machine learning model, in simple words it decides which value to pass as output and what not to pass

## ALGORITHM:
1. Import the required modules for the sigmoid function.
2. Create a class and define the functions  
3. Give the derivatie function for training the existing neuron to create a feed forward network.
4. Based on sigmoid activation function the weights are adjusted.
5. Give the  set of inputs for new neuron so as to get the output.


## PROGRAM:
```
/*
Program to implement the sigmoid activation function in a feed forward ANN.
Developed by:
RegisterNumber:  
*/
import numpy as np
class NeuralNetwork():

   def __init__(self):
        # seeding for random number generation
        np.random.seed(1)

        #converting weights to a 3 by 1 matrix with values from -1 to 1 and mean of 0
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

   def sigmoid(self, x):
        #applying the sigmoid function
        return 1 / (1 + np.exp(-x))

   def sigmoid_derivative(self, x):
       #computing derivative to the Sigmoid function
       return x * (1 - x)

         def train(self, training_inputs, training_outputs, training_iterations):

       #training the model to make accurate predictions while adjusting weights continually
       for iteration in range(training_iterations):
       #siphon the training data via the neuron
           output = self.think(training_inputs)

           #computing error rate for back-propagation
           error = training_outputs - output

           #performing weight adjustments
           adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))

           self.synaptic_weights += adjustments

        def think(self, inputs):
         #passing the inputs via the neuron to get output
         #converting values to floats

          inputs = inputs.astype(float)
          output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
          return output

        if __name__ == "__main__":

        #initializing the neuron class
        neural_network = NeuralNetwork()

        print("Beginning Randomly Generated Weights: ")
        print(neural_network.synaptic_weights)

        #training data consisting of 4 examples--3 input values and 1 output
        training_inputs = np.array([[0,0,1],
        [1,1,1],
        [1,0,1],
        [0,1,1]])

        training_outputs = np.array([[0,1,1,0]]).T

        #training taking place
        neural_network.train(training_inputs, training_outputs, 15000)

        print("Ending Weights After Training: ")

        print(neural_network.synaptic_weights)

        user_input_one = str(input("User Input One: "))
        user_input_two = str(input("User Input Two: "))
        user_input_three = str(input("User Input Three: "))

        print("Considering New Situation: ", user_input_one, user_input_two, user_input_three)
        print("New Output data: ")
        print(neural_network.think(np.array([user_input_one, user_input_two, user_input_three])))
        print("Wow, we did it!")
```
<br>
<br>
<br>
<br>
<br>


## OUTPUT:
![image](https://user-images.githubusercontent.com/86832944/169003000-171e2cc2-b5e4-4feb-bb8d-a3dfb5efe02d.png)


## RESULT:
  Thus created a perception to employ the Sigmoid activation function. This neuron was successfully trained to predict the correct output value, when provided with a new set of input data.
