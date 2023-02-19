import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.stats import multivariate_normal

def get_gaussian_random():
    m = 0
    while m == 0:
        m = round(np.random.random() * 100)

    numbers = np.random.random(int(m))
    summation = float(np.sum(numbers))
    gaussian = (summation - m/2) / math.sqrt(m/12.0)

    return gaussian

def generate_known_gaussian(dimensions,count):
    
    ret = []
    for i in range(count):
        current_vector = []
        for j in range(dimensions):
            g = get_gaussian_random()
            current_vector.append(g)

        ret.append( tuple(current_vector) )

    return ret

def sample(count,c):
    
    known = generate_known_gaussian(2,count)
    target_mean = np.matrix([ [1.0], [5.0]])
    target_cov  = np.matrix([[  1.0, 0.7], 
                             [  0.7, 0.6]])
                         
    [eigenvalues, eigenvectors] = np.linalg.eig(target_cov)
    l = np.matrix(np.diag(np.sqrt(eigenvalues)))
    Q = np.matrix(eigenvectors) * l
    x1_tweaked = []
    x2_tweaked = []
    tweaked_all = []
    for i, j in known:
        original = np.matrix( [[i], [j]]).copy()
        tweaked = (Q * original) + target_mean
        x1_tweaked.append(float(tweaked[0]))
        x2_tweaked.append(float(tweaked[1]))
        tweaked_all.append( tweaked )

    return tweaked_all
           




def parzen_window_est(x_samples, h=1, center=[0,0,0]):
    '''
    Implementation of the Parzen-window estimation for hypercubes.

    Keyword arguments:
        x_samples: A 'n x d'-dimensional numpy array, where each sample
            is stored in a separate row.
        h: The length of the hypercube.
        center: The coordinate center of the hypercube

    Returns the probability density for observing k samples inside the hypercube.

    '''
    dimensions = np.asarray(x_samples).shape[1]

 #   assert (len(center) == dimensions)
 #           'Number of center coordinates have to match sample dimensions'
    k = 0
    for x in x_samples:
        is_inside = 1
        for axis,center_point in zip(x, center):
            if np.abs(axis-center_point) > (h/2):
                is_inside = 0
        k += is_inside
    return (k / len(x_samples)) / (h**dimensions)





def knn(x_samples, k, center=[0,0]):

  
    dimensions = np.asarray(x_samples).shape[1]

 #   assert (len(center) == dimensions)
 #           'Number of center coordinates have to match sample dimensions'
    n = 0
    kn_list=[]
    for x in x_samples:
        kn_list.append(np.sqrt((center[0]-x_samples[0])**2 + (center[1]-x_samples[1])**2))
        
    kn_list.sort()
    r = kn_list[k]
    return k/(len(x_samples)*np.pi*r**2)












def main():
    
    
#known = generate_known_gaussian(2,10)
#learn_mean_cov(tweaked_all)
    Mean_Matrix = []
    Cov_Matrix = []
    
    Mean_Matrix1 = []
    Cov_Matrix1 = []

    Mean_Matrix2 = []
    Cov_Matrix2 = []

    Mean_Matrix3 = []
    Cov_Matrix3 = []

    
    #for i in range(20):
   # known = generate_known_gaussian(2,10)
#    for i in range(5):
        
        # Mean_Matrix,Cov_Matrix = learn_mean_cov(sample(10,i))
        # Mean_Matrix1.append(Mean_Matrix)
        # Cov_Matrix1.append(Cov_Matrix)

        # Mean_Matrix,Cov_Matrix = learn_mean_cov(sample(100,i))
        # Mean_Matrix2.append(Mean_Matrix)
        # Cov_Matrix2.append(Cov_Matrix)

#        Mean_Matrix,Cov_Matrix = learn_mean_cov(sample(1000,i))
#        Mean_Matrix3.append(Mean_Matrix)
#        Cov_Matrix3.append(Cov_Matrix)

#    print(Mean_Matrix1)

#    print(cov_mean_bios(Cov_Matrix1))
#    print(mean_mean_bios(Mean_Matrix1))


print('p(x) =', parzen_window_est(sample(100,2), h=1, center=[1,5]))
print ("P2(x) =",knn(sample(100,2), 3, center=[1,5]))
    #print(Mean_Matrix2,"\n",Cov_Matrix2)

        
  



if __name__ == "__main__":
    main()



