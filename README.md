# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import dataset and compute cost value
2. Calculate the gradient descent
3. Find H(x) equation
4. Plot cost function using gradient descent
5. Plot profit prediction graph
6. Check the prediction

## Program:
```
Program to implement the linear regression using gradient descent.
Developed by: SuriyaPrakash.B
RegisterNumber:  212220220048
```
```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```
```
data=pd.read_csv('/content/ex1.txt',header=None)
```
```
plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Predicted")
```
```
def compute(x,y,theta):
  m=len(y)
  h=x.dot(theta)
  square_err=(h-y)**2
  return 1/(2*m)*np.sum(square_err)
```
```
data_n=data.values
m=data_n[:,0].size
x=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))
compute(x,y,theta)
```
```
def gradientdescent(x,y,theta,alpha,num_iters):
  m=len(y)
  j_history=[]
  for i in range(num_iters):
    predictions=x.dot(theta)
    error=np.dot(x.transpose(),(predictions - y))
    descent=alpha * 1/m * error
    theta-=descent
    j_history.append(compute(x,y,theta))
  return theta,j_history
plt.plot(j_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)")
plt.title("Cost function using Gradient Descent")
plt.show()
```
```
plt.scatter(data[0],data[1],color="black")
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="red")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of city (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")
plt.show()
```
```
theta,j_history=gradientdescent(x,y,theta,0.01,1500)
print("h(x) = "+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")
```
```
def predict(x,theta):
  predictions=np.dot(theta.transpose(),x)
  return predictions[0]
predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000,we predict a profit of $"+str(round(predict1,0)))
predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000,we predict a profit of $"+str(round(predict2,0)))
```

## Output:
<img src="https://github.com/Username24112002/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/104640337/e6588547-a95c-4154-b83c-ffdd4ba71bfc" alt="alt text" width="200" height="200">

![Screenshot (20)](https://github.com/Username24112002/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/104640337/960aa6ee-4cad-47a7-9e99-e563d32ddff2)
![Screenshot (21)](https://github.com/Username24112002/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/104640337/a155b2f5-15c1-4d6f-a769-4b29a7f5f1f4)

![Screenshot (22)](https://github.com/Username24112002/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/104640337/87342bdf-4cbd-4b2a-8c5e-b14004afc0cc)
![Screenshot (23)](https://github.com/Username24112002/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/104640337/26425a72-4165-4be5-9bcf-61757ec13007)
![Screenshot (24)](https://github.com/Username24112002/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/104640337/1e6efc7c-75aa-47ea-a04e-e5a7701ba34a)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
