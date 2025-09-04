def f(x, y=0):
    return y + y + y + x


y = 0
for i in range(100):
    y = f(i, y)
    print(y)
