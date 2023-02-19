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


def learn_mean_cov(pts):
    learned_mean = np.matrix([[0.0], [0.0]])
    learned_cov  = np.zeros( (2, 2) )
    count = len(pts)
    for pt in pts:
        learned_mean += pt
        learned_cov  += pt*pt.transpose()

    learned_mean /= count
    learned_cov /= count
    learned_cov -= learned_mean * learned_mean.transpose()
    return learned_mean,learned_cov

    
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
    target_mean = np.matrix([ [0.0], [5.0]])
    target_cov  = np.matrix([[  1.0, 1.0], 
                             [  1.0, 2.0]])
                         
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
    if c==0:

         mu = np.array([1.0, 5.0])
         X, Y = np.meshgrid(x1_tweaked, x2_tweaked)
         pos = np.dstack((X, Y))
         rv = multivariate_normal(mu, target_cov)
         Z = rv.pdf(pos)
         fig = plt.figure()
         ax = fig.add_subplot(111, projection='3d')
         ax.plot_surface(X, Y, Z)
     #    fig.show()
    return tweaked_all
#optional 2
#################################
def bio_mean(cov,meanb,covb,data):
    plus=[0,0]
    for i in data:
        plus+=i
    map_mu=np.linalg.inv(len(data)*np.linalg.inv(cov)+np.linalg.inv(covb))*(np.linalg.inv(covb)*meanb+np.linalg.inv(cov)*plus)

    new_array = [tuple(row) for row in map_mu]
    uniques = np.unique(new_array)
    return uniques
#################################

    return tweaked_all
def mean_mean_bios(mean):
    true_mean = np.matrix([ [1.0], [5.0]])
    mean_matrix = np.zeros((2,1))
    var_matrix = np.zeros((2,1))
    for m in mean:
        for i in range(2):
            mean_matrix[i,0] += m[i,0]
    mean_matrix /= len(mean)

    for m in mean:
        for i in range(2):
            var_matrix[i] += (m[i,0]-mean_matrix[i,0])**2
    var_matrix /= len(mean)
    bios_matrix = true_mean - mean_matrix
    return var_matrix, bios_matrix

            
def cov_mean_bios(cov):
    true_cov = np.matrix([[  1.0, 0.7], 
                         [0.7, 0.6]])
    mean_matrix = np.zeros((2,2))
    var_matrix = np.zeros((2,2))
    for c in cov:
        for i in range (2):
            for j in range (2):
                mean_matrix[i,j] += c[i,j]
    mean_matrix /= len(cov)
    for c in cov:
        for j in range(2):
            for k in range(2):
                var_matrix[j,k] += (c[j,k]-mean_matrix[j,k])**2
    var_matrix /= len(cov)
    bios_matrix = true_cov - mean_matrix
    return var_matrix,bios_matrix
    

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
    for i in range(1):
        print(type(sample(100,1)))
        
        Mean_Matrix,Cov_Matrix = learn_mean_cov(sample(10,i))
        Mean_Matrix1.append(Mean_Matrix)
        Cov_Matrix1.append(Cov_Matrix)
        
        Mean_Matrix,Cov_Matrix = learn_mean_cov(sample(100,i))
        Mean_Matrix2.append(Mean_Matrix)
        Cov_Matrix2.append(Cov_Matrix)

        Mean_Matrix,Cov_Matrix = learn_mean_cov(sample(1000,i))
        Mean_Matrix3.append(Mean_Matrix)
        Cov_Matrix3.append(Cov_Matrix)
    co=np.array([[1,1],
                 [1,2]])
    cob=np.array([[10,0],
                 [0,10]])
    meanb=np.array([[0],[0]])
    
        
    print ( "map_mean1",np.unique(bio_mean(co,meanb,cob,sample(10,i))))
    print ( "map_mean2",bio_mean(co,meanb,cob,sample(100,i)))
    print ( "map_mean3",bio_mean(co,meanb,cob,sample(1000,i)))

    
    print("\nMean Matrix1:\n",Mean_Matrix1)
    print("\nCov_Matrix1:\n",Cov_Matrix1)
    
    print("\nMean Matrix2:\n",Mean_Matrix2)
    print("\nCov_Matrix2:\n",Cov_Matrix2)
    
    print("\nMean Matrix3:\n",Mean_Matrix3)
    print("\nCov_Matrix3:\n",Cov_Matrix3)

    print('Variance and Bios For Covariance1: ', cov_mean_bios(Cov_Matrix1))
    print('Variance and Bios For MU1: ', mean_mean_bios(Mean_Matrix1))

    print('Variance and Bios For Covariance2: ', cov_mean_bios(Cov_Matrix2))
    print('Variance and Bios For MU2: ', mean_mean_bios(Mean_Matrix2))

    print('Variance and Bios For Covariance3: ', cov_mean_bios(Cov_Matrix3))
    print('Variance and Bios For Mu3: ', mean_mean_bios(Mean_Matrix3))
    xc = input()

if __name__ == "__main__":
    main()
