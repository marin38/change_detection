import numpy as np
import cv2

class BatchGeneratorStatic(object):
    
    def __init__(self, batch_size = 32, dim_x = 240, dim_y = 192, dim_z = 6, shuffle = True):
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z
        self.batch_size = batch_size
        self.shuffle = shuffle

    def generate(self, dir_name, file_names):
        # Infinite loop
        while 1:
        # Generate order of exploration of dataset
            indexes = self.__get_exploration_order(file_names)

            # Generate batches
            imax = int(len(indexes)/self.batch_size)
            for i in range(imax):
                # Find list of IDs
                list_files_temp = [file_names[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]
                # Generate data
                X_out, y_out = self.data_generation(dir_name, list_files_temp)

                yield X_out, y_out

    def __get_exploration_order(self, X):
        
        # Find exploration order
        indexes = np.arange(len(X))
        if self.shuffle == True:
            np.random.shuffle(indexes)

        return indexes

    def data_generation(self, dir_name, list_files_temp):
        # X : (n_samples, v_size, v_size, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim_y, self.dim_x, self.dim_z))
        y = np.empty((self.batch_size, self.dim_y, self.dim_x))

        # Generate data
        for i in range(len(list_files_temp)):
            # Store volume
            img_1 = cv2.imread(dir_name+'1/'+list_files_temp[i])
            img_2 = cv2.imread(dir_name+'2/'+list_files_temp[i])
            img_gt = cv2.imread(dir_name+'gt/'+list_files_temp[i],0)
            
            #print(img.shape)
            X[i, :, :, :3] = img_1/255
            X[i, :, :, 3:] = img_2/255
            
            y[i, :, :]=img_gt/255

            
            #y = y/255
            #y=y.flatten()
        return X, y
    def debug(self, X, y):
        indexes = self.__get_exploration_order(X)
        imax = int(len(indexes)/self.batch_size)
        for i in range(1):
            # Find list of IDs
            list_files_temp = [file_names[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]
            # Generate data
            X_out, y_out = self.data_generation(dir_name, list_files_temp)
            return  X_out, y_out