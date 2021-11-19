import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings( "ignore" )
from variable import Variable

# to compare our model's accuracy with sklearn model
from sklearn.linear_model import LogisticRegression
# Logistic Regression
class LogitRegression() :
	def __init__( self, learning_rate, iterations ) :		
		self.learning_rate = learning_rate
		self.bias = 0.001		
		self.iterations = iterations

		
	def predict( self, X) :	
		Z = []
		temp = 1
		for row in X:
			temporary = 0
			for m in row:
				temporary+= m*self.weights["m_{}".format(temp)]
			
			Z.append(1/(1+ np.exp((temporary + self.bias)/-1)))
		Y = []
		for lol in Z:
			Y.append(np.where( lol > 0.5, 1, 0 ))		
		return Y
		
	# Function for model training	
	def fit( self, X, Y ) :
		self.X = X
		self.Y = Y
		self.m, self.n = X.shape
		self.coefs = [Variable(name= "m_{}".format(i+1)) for i in range(self.n)]
		self.vars = [Variable(name= "m_{}".format(i+1)) for i in range(self.n)]
		self.weights = {"m_{}".format(i+1) : np.random.uniform(-1, 1) for i in range(self.n)}
		self.loss_val = []
		self.z = sum(coeff * var for coeff, var in zip(self.coefs[1:], self.vars)) + self.coefs[0]
		self.yhat = 1/1+math.e**(0-self.z)
		for j in range(self.iterations):
			self.update_weights()
				
		return sum(self.loss_val)/len(self.loss_val)

	def update_weights(self):
		for X, y in zip(self.X, self.Y):
				vars_val = {(i+1) : var for i, var in enumerate(X)}
				all_coeffs = vars_val
				all_coeffs.update(self.weights)
				loss = 0 - y * Variable.log(self.yhat) - (1 - y) * Variable.log(1 - self.yhat)
				self.loss_val.append(loss.evaluate(all_coeffs))
				loss_grad = loss.grad(all_coeffs)
				for i in range(len(self.weights) - 1):
					self.weights["m_{}".format(i+1)] -= loss_grad[i] * self.learning_rate
	def cost_function(self):
		temp = 1
		cost = 0
		for i, row in enumerate(self.X):
			temporary = 0
			for m in row:
				temporary+= m*self.weights["m_{}".format(temp)]
			
			yhat = 1/(1+ np.exp((temporary + self.bias)/-1))
			cost+= 0-(self.Y[i] * np.log(yhat) + (1-self.Y[i])*np.log(1-yhat))
			temp+=1
		
		return cost 

	



		# # no_of_training_examples, no_of_features
		# self.X = X
		# self.Y = Y	
		# self.m, self.n = X.shape
		# self.b = self.bias
		# # self.coefs = [Variable(evaluate= 1/(1 + np.exp( - (self.X.dot(self.weights) + self.b))), name=i) for i in self.n]
		
		# # yhat = Variable(name="yhat")
		# costs = []
		# print(self.coefs)
		# print(self.weights)
		# print("hi")

		# for i in range(self.iterations):
		# 	# create yhat using current weights
		# 	yhat = hypothesis
		# 	loss = 

		# 	for killme in range(len(self.coefs)):
		# 		for why in range(self.m):
		# 			self.coefs[killme] = 0 - Y[why] * Variable.log(self.yhat)
		# 	# take the gradient of the cost function and multiply by learning rate to subtract from weight


		# for i in range(self.iterations):
		# 	for s in range(len(self.coefs)):
		# 		self.weights["m_{}".format(s+1)] -= self.learning_rate*self.coefs[s].grad({self.weights})
		# 	costs.append(LogisticRegression.cost_function())
		
		# indep1 = Variable()
		# predictedweight = Variable()
		# costfunc = 0 - (indep1 * Variable.log(predictedweight) + (1 - indep1)* Variable.log(1 - predictedweight)) 
		# for i in range(self.iterations):

		# 	yhat = 1/(1+ np.exp((sum(self.weights["m_{}".format(z + 1)] * X[z] for z in range) + self.b)/-1))



		# 	for m in range(self.coefs):
		# 		self.coefs["m_{}".format(m+1)].grad({})
		# 	self.weights = self.weights - self.learning_rate * costfunc.grad({"yhat": hypothesis})
		
		# return self
	# 	# take gradient of cost function with given coefficients, update weights

	# 	# weight initialization		
	# 	self.W = np.zeros( self.n )		
	# 	self.b = 0		
	# 	self.X = X		
	# 	self.Y = Y
		
	# 	for i in range(self.iterations):
		
	# 	# gradient descent learning
				
	# 	# for i in range( self.iterations ) :			
	# 	# 	self.update_weights()			
	# 	# return self
	
	# # Helper function to update weights in gradient descent
	
	# # def update_weights( self ) :		
	# # 	A = 1 / ( 1 + Variable.exp( - ( self.X.dot( self.W ) + self.b ) ) )
		
	# # 	# calculate gradients		
	# # 	tmp = ( A - self.Y.T )		
	# # 	tmp = np.reshape( tmp, self.m )		
	# # 	dW = np.dot( self.X.T, tmp ) / self.m		
	# # 	db = np.sum( tmp ) / self.m
		
	# # 	# update weights	
	# # 	self.W = self.W - self.learning_rate * dW	
	# # 	self.b = self.b - self.learning_rate * db
		
	# # 	return self
	
	# # # Hypothetical function h( x )