
import torch
import torch.utils.data

import cv2 as cv
import numpy as np
import pickle as pkl



class DatasetForSingle:
    def __init__(self, f_name, does_augment, size=None):
        self.dataset = None
        self.aug = does_augment
        with open(f_name, 'rb') as f: self.dataset = pkl.load(f)
        
        self.param = None
        if size == '96x96':
            self.param = {'ro_center': 75,
                          'crop': [100, 120],
                          'resize': [112, 96],
                          'scale': 1,
                          }
        elif size == '224x224':
            self.param = {'ro_center': 150,
                          'crop': [200, 240],
                          'resize': [224, 112],
                          'scale': 2,
                          }
        else:
            raise NotImplementedError()
            
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        tmp_data = self.dataset[idx][0]
        label = int(self.dataset[idx][2])
        
        if self.aug: 
            # make a clip(64~80)
            Fr = np.random.randint(64, tmp_data.shape[1] + 1)    # num of frames
            Re = tmp_data.shape[1] - Fr    # rest of all frames
            A = 0 if Re == 0 else np.random.randint(0, Re)
            tmp_data = tmp_data[:, A:A + Fr, :, :]
            
            # resample an index list including 32 frames
            tmp_idx = []
            for bb in range(32):
                jit = np.random.uniform(-1,1)
                idx = (Fr / 32) * (bb + jit / 2)
                tmp_idx.append(int(idx))
            # for bb
            
            # rotation param
            Ro = np.random.randint(-25,26)   # rotation degree  # -15(20),16(21)        
            (h, w) = tmp_data.shape[2:]
            Cx = (w // 2) + np.random.randint(-self.param['ro_center'], self.param['ro_center'])
            Cy = (h // 2) + np.random.randint(-self.param['ro_center'], self.param['ro_center'])
            Ma = cv.getRotationMatrix2D((Cx, Cy), Ro, 1.0)  # rotate matrix
            
            # resize param
            Le = np.random.randint(self.param['crop'][0], self.param['crop'][1]) 
            Re_x = tmp_data.shape[2] - Le
            Re_y = tmp_data.shape[3] - Le
            B = 0 if Re_x == 0 else np.random.randint(0, Re_x)
            C = 0 if Re_y == 0 else np.random.randint(0, Re_y)

            # execute rotation, resize and resample
            tmp_img_lst = []
            for cc in tmp_idx:
                # rotation
                tmp_img = cv.warpAffine(tmp_data[0][cc], Ma, (w, h))

                # resize
                tmp_img = tmp_img[B:B + Le, C:C + Le]
                if self.size == '96x96': tmp_img = cv.resize(tmp_img, (self.param['resize'][1], self.param['resize'][1]))
                if self.size == '224x224': tmp_img = cv.resize(tmp_img, (self.param['resize'][0], self.param['resize'][0]))
                
                tmp_img_lst.append(tmp_img)
            # for cc
            
            tmp_img_lst = [tmp_img_lst] * 3
            data = torch.from_numpy(np.array(tmp_img_lst)).float()
                
        else:
            st1 = self.param['scale'] * 4
            ed1 = st1 + self.param['scale'] + self.param['resize'][0]
            
            st2 = self.param['scale'] * 24
            ed2 = st2 + self.param['scale'] + self.param['resize'][0]
            
            tmp_data = tmp_data[:, :, st1:ed1, st2:ed2]
            
            tmp_img_lst = []
            for dd in range(32):
                jit = np.random.uniform(-1,1)
                idx = (80 / 32) * (dd + jit / 2)
                idx = int(idx)
                
                tmp_img = tmp_data[0][idx]
                if self.size == '96x96': tmp_img = cv.resize(tmp_img, (self.param['resize'][1], self.param['resize'][1]))
                tmp_img_lst.append(tmp_img)
            # for dd   
            
            tmp_img_lst = [tmp_img_lst] * 3
            data = torch.from_numpy(np.array(tmp_img_lst)).float()

        # if self.aug
        
        return {'data':data, 'label':label}




class DatasetForFlow:
    def __init__(self, f_name, does_augment, size=None):
        self.dataset = None
        self.aug = does_augment
        with open(f_name, 'rb') as f: self.dataset = pkl.load(f)
        
        self.param = None
        if size == '96x96':
            self.param = {'ro_center': 75,
                          'crop': [100, 120],
                          'resize': [112, 96],
                          'scale': 1,
                          }
        elif size == '224x224':
            self.param = {'ro_center': 150,
                          'crop': [200, 240],
                          'resize': [224, 112],
                          'scale': 2,
                          }
        else:
            raise NotImplementedError()
            
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        tmp_data = self.dataset[idx][0]
        tmp_mask = self.dataset[idx][1].astype(np.uint8)
        label = int(self.dataset[idx][2])
        
        if self.aug: 
            # make a clip(64~80)
            Fr = np.random.randint(64, tmp_data.shape[1] + 1)    # num of frames
            Re = tmp_data.shape[1] - Fr    # rest of all frames
            A = 0 if Re == 0 else np.random.randint(0, Re)
            
            tmp_data = tmp_data[:, A:A + Fr, :, :]
            tmp_mask = tmp_mask[:, A:A + Fr, :, :]
            
            # resample an index list including 32 frames
            tmp_idx = []
            for bb in range(32):
                jit = np.random.uniform(-1,1)
                idx = (Fr / 32) * (bb + jit / 2)
                tmp_idx.append(int(idx))
            # for bb
            
            # rotation param
            Ro = np.random.randint(-25,26)  
            (h, w) = tmp_data.shape[2:]
            Cx = (w // 2) + np.random.randint(-self.param['ro_center'], self.param['ro_center'] + 1)
            Cy = (h // 2) + np.random.randint(-self.param['ro_center'], self.param['ro_center'] + 1)
            Ma = cv.getRotationMatrix2D((Cx, Cy), Ro, 1.0)  # rotate matrix
            
            # resize param
            Le = np.random.randint(self.param['crop'][0], self.param['crop'][1])    
            Re_x = tmp_data.shape[2] - Le
            Re_y = tmp_data.shape[3] - Le
            B = 0 if Re_x == 0 else np.random.randint(0, Re_x)
            C = 0 if Re_y == 0 else np.random.randint(0, Re_y)


            # execute rotation, resize and resample
            tmp_img_lst = []
            tmp_msk_lst = []
            tmp_msk_x_lst = []
            tmp_msk_y_lst = []
            for cc in tmp_idx:
                # rotation
                tmp_msk_x = cv.warpAffine(tmp_mask[0][cc], Ma, (w, h)) 
                tmp_msk_y = cv.warpAffine(tmp_mask[1][cc], Ma, (w, h)) 

                # resize
                tmp_msk_x = tmp_msk_x[B:B + Le, C:C + Le]
                tmp_msk_y = tmp_msk_y[B:B + Le, C:C + Le]
                tmp_msk_x = cv.resize(tmp_msk_x, (self.param['resize'][1], self.param['resize'][1]))
                tmp_msk_y = cv.resize(tmp_msk_y, (self.param['resize'][1], self.param['resize'][1]))
                tmp_msk_x_lst.append(tmp_msk_x)
                tmp_msk_y_lst.append(tmp_msk_y)
                
                # rotation
                tmp_img = cv.warpAffine(tmp_data[0][cc], Ma, (w, h))

                # resize
                tmp_img = tmp_img[B:B + Le, C:C + Le]
                if self.size == '96x96': tmp_img = cv.resize(tmp_img, (self.param['resize'][1], self.param['resize'][1]))
                if self.size == '224x224': tmp_img = cv.resize(tmp_img, (self.param['resize'][0], self.param['resize'][0]))
                tmp_img_lst.append(tmp_img)
     
            # for cc
            tmp_img_lst = [tmp_img_lst] * 3
            data = torch.from_numpy(np.array(tmp_img_lst)).float()
            
            tmp_msk_lst = [tmp_msk_x_lst, tmp_msk_y_lst]
            flow = torch.from_numpy(np.array(tmp_msk_lst)).int().float()
            flow /= 255.0
                
        else:
            st1 = self.param['scale'] * 4
            ed1 = st1 + self.param['scale'] + self.param['resize'][0]
            
            st2 = self.param['scale'] * 24
            ed2 = st2 + self.param['scale'] + self.param['resize'][0]
            
            tmp_data = tmp_data[:, :, st1:ed1, st2:ed2]
            tmp_mask = tmp_mask[:, :, st1:ed1, st2:ed2]
            
            tmp_img_lst = []
            tmp_msk_lst = []
            tmp_msk_x_lst = []
            tmp_msk_y_lst = []
            for dd in range(32):
                jit = np.random.uniform(-1,1)
                idx = (80 / 32) * (dd + jit / 2)
                idx = int(idx)
                
                tmp_img = tmp_data[0][idx]
                if self.size == '96x96': tmp_img = cv.resize(tmp_img, (self.param['resize'][1], self.param['resize'][1]))
                tmp_img_lst.append(tmp_img)

                tmp_msk_x = cv.resize(tmp_mask[0][idx], (self.param['resize'][1], self.param['resize'][1]))
                tmp_msk_y = cv.resize(tmp_mask[1][idx], (self.param['resize'][1], self.param['resize'][1]))
                tmp_msk_x_lst.append(tmp_msk_x)
                tmp_msk_y_lst.append(tmp_msk_y)
            # for dd   
            
            tmp_img_lst = [tmp_img_lst] * 3
            data = torch.from_numpy(np.array(tmp_img_lst)).float()
            
            tmp_msk_lst = [tmp_msk_x_lst, tmp_msk_y_lst]
            flow = torch.from_numpy(np.array(tmp_msk_lst)).int().float()
            flow /= 255.0
        # if self.aug
        
        
        return {'data':data, 'flow':flow, 'label':label}





class DatasetForMask:
    def __init__(self, f_name, does_augment, size=None):
        self.dataset = None
        self.aug = does_augment
        with open(f_name, 'rb') as f: self.dataset = pkl.load(f)
        
        self.param = None
        self.size = size
        if size == '96x96':
            self.param = {'ro_center': 75,
                          'crop': [100, 120],
                          'resize': [112, 96],
                          'scale': 1,
                          }
        elif size == '224x224':
            self.param = {'ro_center': 150,
                          'crop': [200, 240],
                          'resize': [224, 112],
                          'scale': 2,
                          }
        else:
            raise NotImplementedError()
            
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        tmp_data = self.dataset[idx][0]
        tmp_mask = self.dataset[idx][1].astype(np.uint8)
        label = int(self.dataset[idx][2])
        
        if self.aug: 
            # make a clip(64~80)
            Fr = np.random.randint(64, tmp_data.shape[1] + 1)    # num of frames
            Re = tmp_data.shape[1] - Fr    # rest of all frames
            A = 0 if Re == 0 else np.random.randint(0, Re)
            
            tmp_data = tmp_data[:, A:A + Fr, :, :]
            tmp_mask = tmp_mask[:, A:A + Fr, :, :]
            
            # resample an index list including 32 frames
            tmp_idx = []
            for bb in range(32):
                jit = np.random.uniform(-1,1)
                idx = (Fr / 32) * (bb + jit / 2)
                tmp_idx.append(int(idx))
            # for bb
            
            # rotation param
            Ro = np.random.randint(-25,26)  
            (h, w) = tmp_data.shape[2:]
            Cx = (w // 2) + np.random.randint(-self.param['ro_center'], self.param['ro_center'] + 1)
            Cy = (h // 2) + np.random.randint(-self.param['ro_center'], self.param['ro_center'] + 1)
            Ma = cv.getRotationMatrix2D((Cx, Cy), Ro, 1.0)  # rotate matrix
            
            # resize param
            Le = np.random.randint(self.param['crop'][0], self.param['crop'][1])    
            Re_x = tmp_data.shape[2] - Le
            Re_y = tmp_data.shape[3] - Le
            B = 0 if Re_x == 0 else np.random.randint(0, Re_x)
            C = 0 if Re_y == 0 else np.random.randint(0, Re_y)


            # execute rotation, resize and resample
            tmp_img_lst = []
            tmp_msk_lst = []
            for cc in tmp_idx:
                # rotation
                tmp_msk = cv.warpAffine(tmp_mask[0][cc], Ma, (w, h)) 

                # resize
                tmp_msk = tmp_msk[B:B + Le, C:C + Le]
                tmp_msk = cv.resize(tmp_msk, (self.param['resize'][1], self.param['resize'][1]))
                tmp_msk_lst.append(tmp_msk)

                
                # rotation
                tmp_img = cv.warpAffine(tmp_data[0][cc], Ma, (w, h))

                # resize
                tmp_img = tmp_img[B:B + Le, C:C + Le]
                if self.size == '96x96': tmp_img = cv.resize(tmp_img, (self.param['resize'][1], self.param['resize'][1]))
                if self.size == '224x224': tmp_img = cv.resize(tmp_img, (self.param['resize'][0], self.param['resize'][0]))
                tmp_img_lst.append(tmp_img)
            # for cc
            
            tmp_img_lst = [tmp_img_lst] * 3
            data = torch.from_numpy(np.array(tmp_img_lst)).float()
            
            tmp_msk_lst = [tmp_msk_lst] * 2
            mask = torch.from_numpy(np.array(tmp_msk_lst)).int().float()
            mask /= 255.0
                
        else:
            st1 = self.param['scale'] * 4
            ed1 = st1 + self.param['scale'] + self.param['resize'][0]
            
            st2 = self.param['scale'] * 24
            ed2 = st2 + self.param['scale'] + self.param['resize'][0]
            
            tmp_data = tmp_data[:, :, st1:ed1, st2:ed2]
            tmp_mask = tmp_mask[:, :, st1:ed1, st2:ed2]
            
            tmp_img_lst = []
            tmp_msk_lst = []
            for dd in range(32):
                jit = np.random.uniform(-1,1)
                idx = (80 / 32) * (dd + jit / 2)
                idx = int(idx)
                
                tmp_img = tmp_data[0][idx]
                if self.size == '96x96': tmp_img = cv.resize(tmp_img, (self.param['resize'][1], self.param['resize'][1]))
                tmp_img_lst.append(tmp_img)

                tmp_msk = cv.resize(tmp_mask[0][idx], (self.param['resize'][1], self.param['resize'][1]))
                tmp_msk_lst.append(tmp_msk)
            # for dd   
            
            tmp_img_lst = [tmp_img_lst] * 3
            data = torch.from_numpy(np.array(tmp_img_lst)).float()
            
            tmp_msk_lst = [tmp_msk_lst]
            mask = torch.from_numpy(np.array(tmp_msk_lst)).int().float()
            mask /= 255.0
        # if self.aug
        
        
        return {'data':data, 'mask':mask, 'label':label}







