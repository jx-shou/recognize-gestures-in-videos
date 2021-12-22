
import torch
import torch.nn as nn
import torch.utils.data

from time import time


from optimizer import AdamW, CyclicLRWithRestarts
from data_io import DatasetForMask
from network.i3d_branch import BranchI3d




B_SIZE = 3
IMG_SIZE = '96x96'   # '96x96' '224x224'
train_set = DatasetForMask('./dataset_dir/train_dataset_name', True, IMG_SIZE)
train_iter = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=B_SIZE)

val_set = DatasetForMask('./dataset_dir/val_dataset_name', False, IMG_SIZE)
val_iter = torch.utils.data.DataLoader(val_set, shuffle=False, batch_size=B_SIZE)


net = BranchI3d(out_channels=1).cuda()
print('i3d-branch')

criterion_ce = nn.CrossEntropyLoss(reduction='mean')
criterion_mse = nn.MSELoss(reduction='sum')

optimizer = AdamW(net.parameters(), lr=0.0003, weight_decay=0.004)
scheduler = CyclicLRWithRestarts(optimizer, B_SIZE, 1050, restart_period=10, t_mult=2, policy="cosine")



epoch = 310
for ep in range(epoch):
    tt = time()
    
    train_loss = 0.0
    train_right_cnt = 0
    train_mask_loss = 0.0
    train_acc = 0.0
    
    net.train()
    scheduler.step()
    for i, batch in enumerate(train_iter):
        data = batch['data'].cuda()
        mask = batch['mask'].cuda()
        label = batch['label'].cuda().long()

        out_res, out_msk = net(data)

        loss1 = criterion_ce(out_res, label)
        train_loss += loss1.item()
        
        loss2 = criterion_mse(out_msk, mask)
        train_mask_loss += loss2.item()    
    
        loss = 10 * loss1 + loss2
        optimizer.zero_grad()        
        loss.backward()
        optimizer.step()
        scheduler.batch_step()
        
        _, predict = torch.max(out_res, 1)
        train_right_cnt += (predict == label).sum().item()
    
        
    # for i, batch
    

    train_loss /= train_set.__len__()
    train_mask_loss /= train_set.__len__()
    train_acc = train_right_cnt / train_set.__len__()


    val_loss = 0.0
    val_right_cnt = 0
    val_acc = 0.0
    
    net.eval()
    with torch.no_grad():
        for val_batch in val_iter:
            val_data = val_batch['data'].cuda()
            val_mask = val_batch['mask'].cuda()
            val_label = val_batch['label'].cuda().long()
            
            val_out_res, val_out_msk = net(val_data)
            
            val_loss_mask = criterion_mse(val_out_msk, val_mask)
            val_loss += val_loss_mask.item()
            
            _, val_predict = torch.max(val_out_res, 1)
            val_right_cnt += (val_predict == val_label).sum().item()  
            
     # with       
                  
    val_acc = val_right_cnt / len(val_set)  
    val_loss /= val_set.__len__()

    tim = int(time() - tt)
    
    print("[%2d/%2d] L:%.8f mask_L:%.8f Acc:%.4f valMask_L:%.8f valAcc:%.4f T:%d" 
          %(ep, epoch, train_loss, train_mask_loss, train_acc, val_loss, val_acc, tim))   

    
































