import torch
import torch.nn as nn	#neural network package
import torchvision.datasets as dsets #to get the MNIST data
import torchvision.transforms as transforms	 #
from torch.autograd import Variable


# Hyper parameters
num_epochs = 10
size_of_batch = 100
learning_rate = 0.0001	#learning rate

#-------------------- MNIST Data Set --------------------#
train_data = dsets.MNIST(		# TRAINING DATA
				root = './data/',	# Root directory of dataset
				train = True,		
				transform = transforms.ToTensor(),	# Make multidimensional matrix of single type out of data
				download = True)


testing_data = dsets.MNIST(		# TESTING DATA
				root = './data/',
				train = False,
				transform = transforms.ToTensor())	#Download is by default False

#--------------------------------------------------------#

#--------------------- Data Loader  ---------------------#
#utils.data is an abstract class representing a Dataset
train_loader = torch.utils.data.DataLoader(
				dataset = train_data,
				batch_size = size_of_batch,
				shuffle = True) 

testing_loader = torch.utils.data.DataLoader(
				dataset = testing_data,
				batch_size = size_of_batch,
				shuffle = False) 

#--------------------------------------------------------#

#--------------------- Create our CNN ---------------------#
class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()	#init nn.Module
		self.layer1 = nn.Sequential(	# "Sequential plugs layers together in a feed-forward fully connected manner"
						# We will use a 2D Convelutional layer.
						# 1 is the number of channels to input the image, 16 is number of channels produced by conv.
						# padding is on all sidesOther possible parameteres are 'stride', 'dilation', and 'bias'
						# http://pytorch.org/docs/master/nn.html#convolution-layers
						nn.Conv2d(1, 16, kernel_size=5, padding=2),		
		            	nn.BatchNorm2d(16),		# parameters (num_features, eps = 1e-5, momentum = 0.1, affine = True) batch normalization, calculated for each dimension
		            	nn.ReLU(),
		            	nn.MaxPool2d(2)) # 2 = kernal size
		
		self.layer2 = nn.Sequential(
						nn.Conv2d(16, 32, kernel_size=5, padding=2),
						nn.BatchNorm2d(32),
						nn.ReLU(),
						nn.MaxPool2d(2))
		
		self.fc = nn.Linear(7*7*32, 10)		# (features in, features out, bias = True). Take 1,568 features and output 10. Fully connected layer
		# 7 = kernal + padding. Input to fully connected layer is 32, 7x7 features 

	def forward(self, x):
		output = self.layer1(x)
		output = self.layer2(output)
		output = output.view(output.size(0), -1)	#If you know the number of rows but not columns, use -1
		output = self.fc(output)
		return output

myCnn = CNN()

print("Here is our CNN so far: " + str(myCnn))
	
#--------------------------------------------------------#

#------------ Define Loss and Optimizer functions -------------#

#criterion = nn.CrossEntropyLoss()	

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(myCnn.parameters(), lr=learning_rate)

#--------------------------------------------------------------#

# Loss and Optimizer
optimizer = torch.optim.Adam(myCnn.parameters(), lr=learning_rate)

# Train the Model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images)
        labels = Variable(labels)
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = myCnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
                   %(epoch+1, num_epochs, i+1, len(train_data)//size_of_batch, loss.data[0]))

# Test the Model
myCnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images)
    outputs = myCnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))


#---------------------------------------------------------------#

#Github CNN tutorial at: https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/convolutional_neural_network/main.py#L33-L53
