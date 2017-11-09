import torch
import torch.nn as nn	#neural network package
import torchvision.datasets as dsets #to get the dsets.MNIST data
import torchvision.transforms as transforms	 #
from torch.autograd import Variable
import torch.nn.functional as F



# Hyper parameters
epochs = 10
bs = 100
learning_rate = 0.0001	#learning rate

training_data = dsets.MNIST(
					root = './data/',
					train = True,
					transform = transforms.ToTensor())
					
testing_data = dsets.MNIST(
					root = './data/',
					train = False,
					transform = transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(
					dataset = training_data,
					batch_size = bs,
					shuffle = True)

test_loader = torch.utils.data.DataLoader(
					dataset = testing_data,
					batch_size = bs,
					shuffle = False)

#------------------- Create the model ---------------------#
class Dense_block(nn.Module):
	def __init__(self, inputChannel, outputChannel, growth_rate):
		super(Dense_block, self).__init__()
		
		interChannel = 4 * growth_rate		#given in paper in Bottleneck layers section
		self.b1 = nn.BatchNorm2d(inputChannel)
		self.r1 = nn.ReLU()
		self.c1 = nn.Conv2d(inputChannel, interChannel, kernel_size=1)
	
		self.b2 = nn.BatchNorm2d(interChannel)
		self.r2 = nn.ReLU()
		self.c2 = nn.Conv2d(interChannel, outputChannel, kernel_size=3, padding=1)

	def forward(self, x):
		out = self.c1(self.r1(self.b1(x)))
		out = self.c2(self.r2(self.b2(out)))
		out = torch.cat((x, out), 1)  # IMPORTANT - this step allows previous layer to be past forward in DenseNet
		return out

class Transition_block(nn.Module):
	def __init__(self, inputChannel, outputChannel):
		super(Transition_block, self).__init__()
		self.b1 = nn.BatchNorm2d(inputChannel)
		self.c1 = nn.Conv2d(inputChannel, inputChannel, kernel_size=1)

	def forward(self, x):
		out = self.c1(F.relu(self.b1(x)))
		out = F.avg_pool2d(out, 2) #pooling layer
		return out


class Classification_block(nn.Module):
	def __init__(self, inputChannel, outputChannel):
		super(Classification_block, self).__init__()
		#self.a1 = nn.AvgPool2d(7)		#image is already too small so leave out
		self.l1 = nn.Linear(inputChannel, outputChannel)
		self.s1 = nn.Softmax()

	def forward(self, x):
		#out = self.s1(self.l1(self.a1(x)))
		#out = self.s1(self.l1(x))
		out = self.l1(x)
		return out


class DenseNet(nn.Module):
	def __init__(self):
		super(DenseNet, self).__init__()
		#variables that change with each additional layer
		self.growth_rate = 12 #growth rate is number of feature maps. How much info each layer contributes
		self.inputChannel = 1	#imput image with 1d depth
		self.accounting = 1	#imput image with 1d depth
		self.layer_number = 1
		self.outputChannel = 1 + self.growth_rate*(self.layer_number - 1)	# = k_0 + k * (layer - 1)
		
		self.intro_layer = nn.Sequential(
						nn.BatchNorm2d(self.inputChannel),
						nn.ReLU(),
						nn.Conv2d(self.inputChannel, self.outputChannel, kernel_size=7, stride=2),
						nn.MaxPool2d(3, stride=2))

		self.update()	
	#---------------- Middle Layers ----------------#

		self.layer1 = []
		for x in range(6):			#Add corresponding number of layers per block
			self.layer1.append(Dense_block(self.inputChannel, self.outputChannel, self.growth_rate))
			self.update()
		self.layer1.append(Transition_block(self.inputChannel, self.outputChannel))
		self.layer1 = nn.Sequential(*self.layer1)	#Unpack the list and put it in a sequence

		self.layer2 = []
		for x in range(12):			#Add corresponding number of layers per block
			self.layer2.append(Dense_block(self.inputChannel, self.outputChannel, self.growth_rate))
			self.update()
		self.layer2.append(Transition_block(self.inputChannel, self.outputChannel))
		self.layer2 = nn.Sequential(*self.layer2)	#Unpack the list and put it in a sequence
		#----- Update for next iteration ---#

#		self.layer3 = []
#		for x in range(24):			#Add corresponding number of layers per block
#			self.layer3.append(Dense_block(self.inputChannel, self.outputChannel, self.growth_rate))
#			self.update()
#		self.layer3.append(Transition_block(self.inputChannel, self.outputChannel))
#		self.layer3 = nn.Sequential(*self.layer3)	#Unpack the list and put it in a sequence
#		#----- Update for next iteration ---#
#
#		self.layer4 = []
#		for x in range(16):			#Add corresponding number of layers per block
#			self.layer4.append(Dense_block(self.inputChannel, self.outputChannel, self.growth_rate))
#			self.update()
#		self.layer4 = nn.Sequential(*self.layer4)	#Unpack the list and put it in a sequence
#		#----- Update for next iteration ---#

	#------------- End of Middle Layers -------------------#

		self.final_layer = nn.Sequential(Classification_block(self.inputChannel, 10))
	
	def forward(self, x):		
		out = self.intro_layer(x) 
		#print("intro_layer out is : " + out)
		out = self.layer1(out) 
		#print("layer1 out is : " + out)
		out = self.layer2(out) 
		#print("layer 2 out is : " + out)
		#out = self.layer3(out) 
		#print("layer 3 out is : " + out)
		#out = self.layer4(out) 
		#print("layer 4 out is : " + out)
		out = self.final_layer(out)
		return out

	def update(self):
		self.inputChannel = self.accounting
		self.layer_number += 1
		self.outputChannel = 12
		#keep track of the growing input at each layer
		self.accounting =  1 + self.growth_rate*(self.layer_number - 1)	# = k_0 + k * (layer - 1)


dn = DenseNet()
print(str(dn))


#------------ Define Loss and Optimizer functions -------------#

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(dn.parameters(), lr=learning_rate)

##--------------------------------------------------------------#
## Train the Model
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images)
        labels = Variable(labels)
		
        print("Image size is " + str(images.size()))
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = dn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
                   %(epoch+1, epochs, i+1, len(train_data)//size_of_batch, loss.data[0]))

dn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images)
    outputs = dn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))



# ---------------- Questions ---------------#
# 1) If we don't include self. in the classes that make the blocks, when we instantiate them, the layers aren't implimented. Why? Is that cause of scope, so the non-self variables are not deep copied?
