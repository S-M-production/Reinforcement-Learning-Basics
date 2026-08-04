import numpy as np
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)

#Cost function:
def cost_function(x, y, w, b):
    y_hat = (x @ w + b)
    return np.sum((y_hat-y)**2)/2,y_hat

#Gradient descent function:
def linear_regression(x,y,learning_rate = 0.001,cut_off = 0.01,max_iterations = 10000):
    w = np.random.uniform(-2,2,size=(x.shape[1]))
    b = np.random.uniform(-2,2,size=(1))[0]
    cost,y_hat = cost_function(x,y,w,b)
    for _ in range(max_iterations):
        error = y_hat - y
        new_w = w - learning_rate*np.average(error*x.T,axis=1)
        new_b = b - learning_rate*np.average(error)
        w,b = new_w,new_b

        new_cost,y_hat = cost_function(x, y, w, b)

        if (abs(new_cost-cost) < cut_off):
            return w,b
        cost = new_cost
    return w,b

#Training data:
scale = 1000
deviation = 10
w = np.array([20,30])
b = 30
x_input = np.random.uniform(-scale,scale,size=(scale,2))
y_output = x_input @ w + b + np.random.normal(0,scale*deviation,size=(scale))

#Testing linear_regression
w,b = linear_regression(x_input,y_output,0.000001,0.1)
print(f"Values: {w},{b:.2f}")
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(*x_input.T,y_output,color="blue")
ax.scatter(*x_input.T,x_input @ w + b,color="orange")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("y")
plt.show()

