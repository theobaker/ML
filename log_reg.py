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
		self.linear = sum(coeff*var for coeff, var in zip(self.coefs[1:], self.vars)) + self.coefs[0]
		self.yhat = 1/(1+math.e**(0-self.linear))

		
	# Function for model training	
	def fit( self, X, Y ) :		
		# no_of_training_examples, no_of_features		
		self.m, self.n = X.shape
		self.coefs = [Variable(evaluate= 1/(1 + np.exp( - (self.X.dot(self.weights) + self.b)))) for i in self.n]
		self.vars = [Variable(name= i) for i in range(self.n)]
		self.weights = {-i : np.random.uniform(-1, 1) for i in range(0, self.n)}
		
		for i in range(self.iterations):
			allcombined = {(i+1) : var for i, var in enumerate(X)}
			loss = 0-Y[i]*Variable.log(self.yhat) - (1-Y[i])*Variable.log(1-self.yhat) 

			# take gradient of cost function with given coefficients, update weights

		# weight initialization		
		self.W = np.zeros( self.n )		
		self.b = 0		
		self.X = X		
		self.Y = Y
		
		for i in range(self.iterations):
		
		# gradient descent learning
				
		for i in range( self.iterations ) :			
			self.update_weights()			
		return self
	
	# Helper function to update weights in gradient descent
	
	def update_weights( self ) :		
		A = 1 / ( 1 + Variable.exp( - ( self.X.dot( self.W ) + self.b ) ) )
		
		# calculate gradients		
		tmp = ( A - self.Y.T )		
		tmp = np.reshape( tmp, self.m )		
		dW = np.dot( self.X.T, tmp ) / self.m		
		db = np.sum( tmp ) / self.m
		
		# update weights	
		self.W = self.W - self.learning_rate * dW	
		self.b = self.b - self.learning_rate * db
		
		return self
	
	# Hypothetical function h( x )
	
	def predict( self, X ) :	
		Z = 1 / ( 1 + np.exp( - ( X.dot( self.W ) + self.b ) ) )		
		Y = np.where( Z > 0.5, 1, 0 )		
		return Y