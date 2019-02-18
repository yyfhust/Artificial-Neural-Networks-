
"""

Author: Yifan Ye
Email: 645577765yyf@gmail.com

"""

import numpy as np 
class RBF_with_competitivelearning ():

    def __init__(self, sigma, hidden_layer_nodes, CL_epoch,eta_CL):
        self.hidden_layer_nodes = hidden_layer_nodes
        self.sigma=sigma
        self.CL_epoch=CL_epoch
        self.eta_CL=eta_CL

    def winner_function(self,x,weights_list):
          winner=weights_list[0]
          indice=0
          sim_list=[]
          for weight in weights_list:
            sim_list.append(self.similarity(x,weight))
            indice=np.argmin(sim_list)
            winner=weights_list[indice]
            return winner,indice


    def similarity(self,x,weight):
        return np.linalg.norm(x-weight)

    def weight_modification(self,x,weight,eta):
        weight=weight+eta*(x-weight)
        
        return weight

    def choice_training_vector(self,patterns):
          sample=np.random.choice(patterns)
          return sample

    def centers_initiate(self,x,hidden_shape):
        random_args = np.random.choice(len(x), hidden_shape)
        centers = x[random_args]

        for i in range(self.CL_epoch):
            sample = self.choice_training_vector(x.reshape((-1)))
            winner,indice=self.winner_function(sample,centers)
            winner=self.weight_modification(sample,winner,self.eta_CL)
            centers[indice]=winner
        return centers

    def kernel(self,center, sigma, x ):
        return np.exp(-self.sigma*np.linalg.norm(center-x))

    def compute_transfer(self,x,centers):

        if len(x.shape) == 1:    ## get single_x
            G= np.zeros([x.shape[0], centers.shape[0]])
            for j, single_center in enumerate(centers):
                G[0][j] = self.kernel(single_center,self.sigma,x)
            return G            
        
        else:
            G= np.zeros([x.shape[0], centers.shape[0]])
            for i,single_x in enumerate(x): 
                for j, single_center in enumerate(centers):
                    G[i][j] = self.kernel(single_center,self.sigma,single_x)
            return G

    def fit_least_squares(self, x_train , y_train ):

        self.centers= self.centers_initiate(x_train, self.hidden_layer_nodes)   ##initiate centers based on x_train
        G = self.compute_transfer(x_train,self.centers)   ## G: N*n      N: number of train set; n : number of hidden nodes
        
        self.W=   np.dot  (   np.dot(np.linalg.pinv(np.dot(np.transpose(G),G)), np.transpose(G) ), y_train)

    def fit_delta(self, x_train , y_train, lr, epoch ):   ## lr: learning_rate
        self.centers= self.centers_initiate(x_train, self.hidden_layer_nodes)   ##initiate centers based on x_train
        self.W = np.random.randn(self.hidden_layer_nodes,x_train.shape[1])
        for e in range(epoch):
            for i, signle_x in enumerate(x_train):
                G =self.compute_transfer( signle_x , self.centers)
                self.W  += lr* (y_train[i]- np.dot( G,self.W) )*np.transpose(G)


    def predict(self, x):

        G=self.compute_transfer(x,self.centers)

        return np.dot(G,self.W)
