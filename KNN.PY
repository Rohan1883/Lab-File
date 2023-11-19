import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor



class K_Nearest_Neighbors_Regressor() :
	
	def __init__( self, K ) :
		
		self.K = K
		
	def fit( self, X_train, Y_train ) :
		
		self.X_train = X_train
		
		self.Y_train = Y_train		
		
		self.m, self.n = X_train.shape
		
	def predict( self, X_test ) :
		
		self.X_test = X_test	
		
		self.m_test, self.n = X_test.shape
		
		Y_predict = np.zeros( self.m_test )
		
		for i in range( self.m_test ) :
			
			x = self.X_test[i]
			
			# find the K nearest neighbors from current test example
			
			neighbors = np.zeros( self.K )
			
			neighbors = self.find_neighbors( x )
			
			# calculate the mean of K nearest neighbors
			
			Y_predict[i] = np.mean( neighbors )
			
		return Y_predict
	
	# Function to find the K nearest neighbors to current test example
			
	def find_neighbors( self, x ) :
		
		euclidean_distances = np.zeros( self.m )
		
		for i in range( self.m ) :
			
			d = self.euclidean( x, self.X_train[i] )
			
			euclidean_distances[i] = d
		
		inds = euclidean_distances.argsort()
		
		Y_train_sorted = self.Y_train[inds]
		
		return Y_train_sorted[:self.K]
	
	# Function to calculate euclidean distance
			
	def euclidean( self, x, x_train ) :
		
		return np.sqrt( np.sum( np.square( x - x_train ) ) )

def main() :
	
	df = pd.read_csv( "salary_data.csv" )

	X = df.iloc[:,:-1].values

	Y = df.iloc[:,1].values
	X_train, X_test, Y_train, Y_test = train_test_split(
	X, Y, test_size = 1/3, random_state = 0 )

	model = K_Nearest_Neighbors_Regressor( K = 3 )

	model.fit( X_train, Y_train )
	
	model1 = KNeighborsRegressor( n_neighbors = 3 )
	
	model1.fit( X_train, Y_train )
	
	Y_pred = model.predict( X_test )
	
	Y_pred1 = model1.predict( X_test )
	
	print( "Predicted values by our model	 : ", np.round( Y_pred[:3], 2 ) )
	
	print( "Predicted values by sklearn model : ", np.round( Y_pred1[:3], 2 ) )
	
	print( "Real values					 : ", Y_test[:3] )

if __name__ == "__main__" :
	
	main()