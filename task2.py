from numpy import array


#list of data 1D
print("1D")
data = [11,22,33,44,55]
#array of data 1D
data= array(data)
print(data)
print(type(data))

#2D example
print("2D")
data2= [[11,22] , [33,44], [55,66]]
data2 = array(data2)
print(data2)
print(type(data2))

print("Indexing data example")
print(data[0])
print(data[4])

print("negative indexes test")
# -1 refers  to lat item in array so -2 refers to the second last item
print(" data[-1] ")
print(data[-1])
print("data[-5]  ")
print(data[-5])

print("indexing 2D")
print(data2[0,0])
print("printing first row of 2d array")
print(data2[0,])

print("slicing arrays")
# data[from:to]
print("1D slicing")
print(data[0:1])
print(data[-2:])

print("2D")
data3 = array([[11,22,33], [44,55,66], [77,88,99]])
x_var , y_var = data3[:,:-1] , data3[:, -1]
print(x_var)
print(y_var)

print("split data")
split = 2
train,test= data3[:split,:],data3[split:,:]
print(train)
print(test)

#data reshaping
print("1D")
print(data.shape)
print("2D")
print(data2.shape)
print('Rows: %d' % data2.shape[0])
print('Cols: %d' % data2.shape[1])

#reshape 1D to 2D
print("reshape 1d to 2d")
data =data.reshape((data.shape[0], 1))
print(data.shape)

#reshape 2D to 3D
data2 = data2.reshape((data2.shape[0], data2.shape[1],1))
print(data2.shape)



