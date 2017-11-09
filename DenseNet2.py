import torch
import torch.nn as nn	#neural network package
import torchvision.datasets as dsets #to get the dsets.MNIST data
import torchvision.transforms as transforms	 #
from torch.autograd import Variable


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
class Dense_block(self, inputChannel, outputChannel):
	def super(Dense_block, self).__init__():
		b1 = nn.BatchNorm2d(inputChannel)
		r1 = nn.ReLU()
		c1 = nn.Conv2d(inputChannel, outputChannel, kernel_size=1)]
	
		b2 = nn.BatchNorm2d(inputChannel)
		r2 = nn.ReLU()
		c2 = nn.Conv2d(inputChannel, outputChannel, kernel_size=3, padding=1)

	def forward(self, x):
		out = c1(r1(b1(x)))
		out = c2(r2(b2(out)))
		out = torch.cat((x, out), 1)  # IMPORTANT - this step allows previous layer to be past forward in DenseNet
		return out

class Transition_block(self, inputChannel, outputChannel):
	def super(Transition_block, self).__init__():
		b1 = nn.BatchNorm2d(inputChannel)
		r1 = nn.ReLU()
		c1 = nn.Conv2d(inputChannel, outputChannel, kernel_size=1)
		a1 = nn.AvgPool2d(2, stride=2)

	def forward(self, x):
		out = a1(c1(r1(b1(x))))
		out = torch.cat((x, out), 1)  # IMPORTANT - this step allows previous layer to be past forward in DenseNet
		return out


class Classification_block(self, inputChannel, outputChannel):
	def super(Classification_block, self).__init__():
		a1 = nn.AvgPool2d(7)
		l1 = nn.Linear(inputChannel, outputChannel)
		s1 = nn.SoftMax()

	def forward(self, x):
		out = s1(l1(a1(x)))
		out = torch.cat((x, out), 1)  # IMPORTANT - this step allows previous layer to be past forward in DenseNet
		return out


class DenseNet(nn.Module):
	def __init__(self):
		super(DenseNet, self).__init__()
		#variables that change with each additional layer
		self.growth_rate = 12 #growth rate is number of feature maps. How much info each layer contributes
		self.inputChannel = 1	#imput image with 1d depth
		self.layer_number = 1
		self.outputChannel = 1 + self.growth_rate*(self.layer_number - 1)	# = k_0 + k * (layer - 1)
		
		intro_layer = nn.Sequential(
						nn.BatchNorm2d(self.inputChannel),
						nn.ReLU(),
						nn.Conv2d(self.inputChannel, self.outputChannel, kernel_size=7, stride=2),
						nn.MaxPool2d(3, stride=2))

		self.inputChannel = self.outputChannel

		self.layer1 = []
		for x in range(6):			#Add corresponding number of layers per block
			self.layer1.append(self.make_dense_block())
			self.layer1.append(self.make_transition_block())
		self.inputChannel = self.outputChannel
		self.outputChannel = 1 + self.growth_rate*(self.layer_number - 1)	# = k_0 + k * (layer - 1)
		#print(self.layer1)
		print(nn.Sequential(*self.layer1))

		self.layer2 = []
		for x in range(12):			#Add corresponding number of layers per block
			self.layer2.append(self.make_dense_block())
			self.layer2.append(self.make_transition_block())
		self.inputChannel = self.outputChannel
		self.outputChannel = 1 + self.growth_rate*(self.layer_number - 1)	# = k_0 + k * (layer - 1)

		self.layer3 = []
		for x in range(24):			#Add corresponding number of layers per block
			self.layer3.append(self.make_dense_block())
			self.layer3.append(self.make_transition_block())
		self.inputChannel = self.outputChannel
		self.outputChannel = 1 + self.growth_rate*(self.layer_number - 1)	# = k_0 + k * (layer - 1)

		self.layer4 = []
		for x in range(16):			#Add corresponding number of layers per block
			self.layer4.append(self.make_dense_block())
		self.inputChannel = self.outputChannel
		self.outputChannel = 1 + self.growth_rate*(self.layer_number - 1)	# = k_0 + k * (layer - 1)

		final_layer = []
		final_layer.append(nn.Conv2d(self.inputChannel, self.outputChannel, kernel_size=7, stride=2)),
		final_layer.append(nn.Linear(self.inputChannel, self.outputChannel))
		final_layer.append(nn.Softmax())
		
		#self.network_layers.append(self.make_classification_block())

		return nn.Sequential(*[intro_layer, layer1, layer2, layer3, layer4, final_layer])

x = DenseNet()
x.make_network()
print(x)


#	def forward(self, x):
#		out = dense_net(x)
#		return out

