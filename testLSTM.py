import os
import numpy as np
import pickle as pkl
import torch
import random
from model_file import device, model, loss_fn, optimizer, data_folder


def load_checkpoint(fpath, model, optimizer):
    print('load_checkpoint')
    checkpoint = torch.load(fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer


if __name__ == "__main__":
    data_split = None
    with open(os.path.join(data_folder, 'data_split.pkl'), 'rb') as fileObject2:
        data_split = pkl.load(fileObject2)
    _, _, data_test = data_split

    model_name = str(model.__class__.__name__)
    outdir = './' + model_name + '_checkpoints'
    outname = 'checkpoint_19998.pt'
    file_path = os.path.join(outdir, outname)
    model, optimizer = load_checkpoint(file_path, model, optimizer)
        
    model.eval()
        
    with torch.no_grad():
        test_dataset_X, _ = data_test
        # print(test_dataset_X.shape, type(test_dataset_X))
        # print(test_dataset_Y.shape, type(test_dataset_Y))
        rand_num = random.randint(0, test_dataset_X.shape[0]-1)
        dv_x = test_dataset_X[rand_num,:,:]
        # dv_y = test_dataset_Y[rand_num]
        # print(dv_x.shape, type(dv_x))
        # print(dv_y.shape, type(dv_y))
        dv_x = np.expand_dims(dv_x, axis=0)
        # dv_y = np.expand_dims(dv_y, axis=0)
        # print(dv.shape, type(dv))
        print('Input:', dv_x)
        
        X_test = torch.tensor(dv_x, dtype=torch.float).to(device=device)
        # Y_test = torch.tensor(dv_y, dtype=torch.float).to(device=device)

        model.init_hidden(X_test.size(0))
        # lstm_out, _ = mv_net.l_lstm(x_batch,nnet.hidden)    
        # lstm_out.contiguous().view(x_batch.size(0),-1)

        y_pred = model(X_test)
        print('Predicted output:', y_pred.reshape(y_pred.shape[0]).data.cpu().numpy().tolist())
        # print('Actual output:', Y_test.data.cpu().numpy().tolist())