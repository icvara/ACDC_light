from scipy.stats import gaussian_kde
import numpy as np

def getMAP(points):
	kernel_estimate = gaussian_kde(points) # generate a continuous kernel density from the data
	densities = kernel_estimate(points) # evaluate the data on the kernel
	return points.T[np.argmax(densities)] # return the parameter set that is more dense



# Example

# Creating a fake set of points to test the function, they should be centered at means
means = np.array([1,4,7])
cov = np.array([[3, -2, 0], [-2, 3.5, 0], [0,0, 1]])
pts = np.random.multivariate_normal(means, cov, size=10000).T # each column is a parameter set (point of the posterior)

print(pts.shape)

#Testing of getMAP()
print(" The estimnated MAP of the distribution centered at ",means, " is ", getMAP(pts))






