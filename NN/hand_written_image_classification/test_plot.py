import matplotlib.pyplot as plt

x = range(100)
y = range(100,200)
a = range(50)
b = range(50,100)
fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(x, y, s=10, c='b', marker="s", label='first')
ax1.scatter(a,b, s=10, c='r', marker="o", label='second')
plt.legend(loc='upper left');
plt.show()