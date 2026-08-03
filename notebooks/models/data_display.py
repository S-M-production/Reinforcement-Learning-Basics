import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)

#Training data:

scale = 100
deviation = 0.1
w = np.array([20,30])
b = 30
x_input = np.random.uniform(-scale,scale,size=(scale,2))
y_output = x_input @ w + b + np.random.normal(0,scale*deviation,size=(scale))
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(x_input[:,0],x_input[:,1],y_output)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("y")
plt.show()