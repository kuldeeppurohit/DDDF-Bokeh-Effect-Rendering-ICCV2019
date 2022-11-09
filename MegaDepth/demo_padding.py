import torch
import sys
from torch.autograd import Variable
import numpy as np
from options.train_options import TrainOptions
opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch
from data.data_loader import CreateDataLoader
from models.models import create_model
from skimage import io
from skimage.transform import resize


#img_path = '24.png'

model = create_model(opt)

input_height = 384
input_width  = 512


def test_simple(model):
    total_loss =0 
    toal_count = 0
    print("============================= TEST ============================")
    model.switch_to_eval()


#    for ii in range(4693,4694):
#        img_path = '/media/dataset/finalresults/EDSR/RCAN-master/AIM2019/Bokeh/dvd/LR/'+str(ii)+'.jpg'  
        
    for ii in range(200):
        img_path = '../Set14/LR/'+str(ii)+'.png'          
        img = np.float32(io.imread(img_path))/255.0
        h, w, c = img.shape
        #print(h,w,c,h//32,w//32)
#        img = resize(img, (h//2, w//2), order = 1)   
        h1=h
        w1=w
        H=((h1//32)+1)*32
        W=((w1//32)+1)*32
        blurPad = np.pad(img, ((0, H - h1), (0, W - w1), (0, 0)), 'edge')
    #    blurPad = np.expand_dims(blurPad, 0)    
        input_img =  torch.from_numpy( np.transpose(blurPad, (2,0,1)) ).contiguous().float()
        input_img = input_img.unsqueeze(0)

        input_images = Variable(input_img.cuda() )
        with torch.no_grad():        
          pred_log_depth = model.netG.forward(input_images) 
        pred_log_depth = torch.squeeze(pred_log_depth)
        pred_depth = torch.exp(pred_log_depth)


        pred_inv_depth = 1/pred_depth
        pred_inv_depth = pred_inv_depth.data.cpu().numpy()
        pred_inv_depth = pred_inv_depth/np.amax(pred_inv_depth)

        pred_inv_depth = pred_inv_depth[:h1, :w1]
        pred_inv_depth = resize(pred_inv_depth, (h, w), order = 1)    
        pred_inv_depth= np.stack((pred_inv_depth,pred_inv_depth,pred_inv_depth), axis=2)    
        io.imsave('../Set14/LR3/'+str(ii)+'.png', pred_inv_depth)
    
    
    # print(pred_inv_depth.shape)
    sys.exit()



test_simple(model)
print("We are done")
