import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

X = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5350]
y = [.664,.669, .693, .702, .726, .732, .740, .742, .75, .756, .759]
x2 = [500, 2500, 5350]
y2 = [.664,.726,.759]

plt.xlabel('hoeveelheid trainingsdata')
plt.ylabel('recall score')

plt.plot(X, y, x2, y2, marker = 'o')
plt.show()

# X = np.array(X).reshape(-1, 1)
# reg = LinearRegression().fit(X, y)
# print(reg.predict([[6000]]))

