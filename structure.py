import numpy as np
import matplotlib.pyplot as plt 

class structure:
    def __init__(self, shape, eps, mu):
        self.shape = shape
        self.eps = eps
        self.mu = mu
        self.epsr = np.ones(shape, dtype='float64')
        self.mur = np.ones(shape, dtype='float64')

    def change_space(self, space):
        space.set_epsr(space.epsr + self.epsr - 1)
        space.set_mur(space.epsr + self.mur - 1)

class Rectangle(structure):
    def __init__(self, shape, center, width, depth, height, eps, mu):
        super(Rectangle, self).__init__(shape, eps, mu)
        self.center = center
        self.width = width
        self.depth = depth
        self.height = height
        min_x, max_x =  center[0] - width //2 , center[0] + width //2
        min_y, max_y =  center[1] - depth //2 , center[1] + depth //2
        min_z, max_z =  center[1] - height //2 , center[1] + height //2
        
        self.epsr[min_x:max_x, min_y:max_y, min_z:max_z] = self.eps
        self.mur[min_x:max_x, min_y:max_y, min_z:max_z] = self.mu
        
    def __str__(self):
        return 'Rectangle({},{},{},{}'.format(self.center, self.width, self.depth, self.height)


class Sphere(structure):
    def __init__(self, shape, center, R, eps, mu, smoothing = False):
        super(Sphere, self).__init__(shape, eps, mu)
        self.center = center
        self.R = R
        self.smoothing = smoothing
        self.get_epsr()

    def get_epsr(self):
        x = np.arange(self.shape[0])
        y = np.arange(self.shape[1])
        z = np.arange(self.shape[2])
        X,Y,Z = np.meshgrid(x,y,z)
        self.r = np.sqrt((self.center[1]-X)**2 + (self.center[0]-Y)**2 + (self.center[2]-Z)**2)
        self.region = np.where(self.r >= self.R, 0, 1)
        self.epsr += self.region * (self.eps - 1)
        if self.smoothing == True:
            smoothing_region = 1 - (np.where((0 < (self.r - self.R)) & ((self.r - self.R) < 1), 1/3*(self.r+np.sqrt(self.r**2-1)-2*self.R)**2, 1))
            self.epsr += smoothing_region * (self.eps - 1)
        del self.r
        

        
