import math
import torch
import torch.nn as nn
import os
import datetime
from networkTool import *
from torch.utils.tensorboard import SummaryWriter
from attentionModel import TransformerLayer,TransformerModule

ntokens = 255 # the size of vocabulary
ninp = 4*(54+4+6) # embedding dimension
nlayers = 3 # the number of layers in the MSA
nhead = 4 # the number of heads in the MSA
dropout = 0 # the dropout value
batchSize = 32

class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.ninp = ninp

        # self.pos_encoder = APE(ninp, dropout)
        self.pos_encoder = LeCE(ninp)

        encoder_layers = TransformerLayer(ninp, nhead, dropout)
        self.transformer_encoder = TransformerModule(encoder_layers, nlayers)
        self.encoder0 = nn.Embedding(ntoken, 54)
        self.encoder1 = nn.Embedding(MAX_OCTREE_LEVEL+1, 6)
        self.encoder2 = nn.Embedding(9, 4)
        self.decoder0 = nn.Linear(ninp, ninp)
        self.act = nn.SiLU()
        self.decoder2 = nn.Linear(ninp, ntoken)
        self.init_weights()

        self.decoder1 = Star(ninp)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder0.weight.data = nn.init.xavier_normal_(self.encoder0.weight.data )
        self.decoder0.bias.data.zero_()
        self.decoder0.weight.data= nn.init.xavier_normal_(self.decoder0.weight.data )
        self.decoder2.bias.data.zero_()
        self.decoder2.weight.data = nn.init.xavier_normal_(self.decoder2.weight.data )

    def forward(self, src, src_mask, dataFeat):
        bptt = src.shape[0]
        batch = src.shape[1]

        oct = src[:,:,:,0]
        level = src[:,:,:,1]
        octant = src[:,:,:,2]

        level -= torch.clip(level[:,:,-1:] - 10,0,None)
        torch.clip_(level,0,MAX_OCTREE_LEVEL)
        aOct = self.encoder0(oct.long())
        aLevel = self.encoder1(level.long())
        aOctant = self.encoder2(octant.long())

        a = torch.cat((aOct,aLevel,aOctant),3)
        a = a.reshape((bptt,batch,-1))

        src = a.reshape((bptt,a.shape[1],self.ninp))* math.sqrt(self.ninp)

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder0(output)
        output = self.decoder1(output)
        output = self.act(output)
        output = self.decoder2(output)
        return output

######################################################################
# ``APE`` module
#

# class APE(nn.Module):
#
#     def __init__(self, d_model, dropout=0.1, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
#
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         x = x + self.pe[:x.size(0), :]
#         return self.dropout(x)

class LeCE(torch.nn.Module):
    def __init__(self, ninp):
        super().__init__()
        self.proj = nn.Conv1d(ninp, ninp, 3, 1, 0, 0, groups=ninp)
        self.conv0_0 = nn.Conv1d(in_channels=ninp,out_channels=ninp // 4,kernel_size=1,stride=1,padding=0,dilation=1,groups=1,bias=False)
        self.conv0_1 = nn.Conv1d( in_channels=ninp // 4, out_channels=ninp // 2, kernel_size=1,  stride=1, padding=0, dilation=1,  groups=1,  bias=False)

        self.conv1_0 = nn.Conv1d( in_channels=ninp, out_channels=ninp // 4, kernel_size=1, stride=1,  padding=0,  dilation=1, groups=1, bias=False)
        self.conv1_1 = nn.Conv1d( in_channels=ninp // 4, out_channels=ninp // 4, kernel_size=3,stride=1, padding=0, dilation=0, groups=ninp // 4,  bias=False)
        self.conv1_2 = nn.Conv1d( in_channels=ninp // 4, out_channels=ninp // 2, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)

        self.act = nn.SiLU()

    def forward(self, x):
        x = x.permute(1, 2, 0)
        x = self.proj(x) + x
        out0 = self.conv0_1(self.act(self.conv0_0(x)))
        out1 = self.conv1_2(self.act(self.conv1_1(self.act(self.conv1_0(x)))))
        out = torch.cat((out0, out1),dim=1) + x
        out = out.permute(2, 0, 1)

        return out

class Star(nn.Module):
    def __init__(self, ninp=ninp, drop=0.):
        super().__init__()
        self.fc1 = nn.Conv1d(in_channels=ninp,out_channels=ninp,kernel_size=1,stride=1,padding=0,dilation=1,groups=1,bias=True)
        self.dwconv1 = nn.Conv1d(in_channels=ninp, out_channels=ninp, kernel_size=3, stride=1, padding=0, dilation=0,
                                groups=ninp, bias=True, padding_mode='zeros')
        self.dwconv2 = nn.Conv1d(in_channels=ninp, out_channels=ninp, kernel_size=3, stride=1, padding=0, dilation=0,
                                 groups=ninp, bias=True, padding_mode='zeros')

        self.fc2 = nn.Conv1d(in_channels=ninp,out_channels=ninp,kernel_size=1,stride=1,padding=0,dilation=1,groups=1,bias=True)
        self.g = nn.Conv1d(in_channels=ninp, out_channels=ninp, kernel_size=1, stride=1, padding=0, dilation=1,
                             groups=1, bias=True)
        self.drop = nn.Dropout(drop)
        self.act = nn.SiLU()

    def forward(self, x):
        input = x
        x = x.permute(1, 2, 0)
        x = self.dwconv1(x)
        x1, x2 = self.fc1(x),self.fc2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = x.permute(2, 0, 1)
        x = input + x

        return x

######################################################################
# Functions to generate input and target sequence
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len].clone()
    target = source[i+1:i+1+seq_len,:,-1,0].reshape(-1)
    data[:,:,0:-1,:] = source[i+1:i+seq_len+1,:,0:-1,:]
    data[:,:,-1,1:3] = source[i+1:i+seq_len+1,:,-1,1:3]
    return data[:,:,-levelNumK:,:], (target).long(),[]

######################################################################
# Run the model
# -------------
#
model = TransformerModel(ntokens, ninp, nhead, nlayers, dropout).to(device)
if __name__=="__main__":
    import dataset
    import torch.utils.data as data
    import time
    import os

    epochs = 8
    best_model = None
    batch_size = 128
    TreePoint = bptt*16
    train_set = dataset.DataFolder(root=trainDataRoot, TreePoint=TreePoint,transform=None,dataLenPerFile= 391563.61670395226)
    train_loader = data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=False, num_workers=4,drop_last=True)

    if not os.path.exists(checkpointPath):
        os.makedirs(checkpointPath)
    printl = CPrintl(expName+'/loss.log')
    writer = SummaryWriter('./log/'+expName)
    printl(datetime.datetime.now().strftime('\r\n%Y-%m-%d:%H:%M:%S'))
    model_structure(model,printl)
    printl(expComment+' Pid: '+str(os.getpid()))
    log_interval = int(batch_size*TreePoint/batchSize/bptt)

    criterion = nn.CrossEntropyLoss()
    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    best_val_loss = float("inf")
    idloss = 0

    # reload
    saveDic = None
    # saveDic = reload(100030,checkpointPath)
    if saveDic:
        scheduler.last_epoch = saveDic['epoch'] - 1
        idloss = saveDic['idloss']
        best_val_loss = saveDic['best_val_loss']
        model.load_state_dict(saveDic['encoder'])

    def train(epoch):
        global idloss,best_val_loss
        model.train()
        total_loss = 0.
        start_time = time.time()
        total_loss_list = torch.zeros((1,7))

        for Batch, d in enumerate(train_loader):
            batch = 0

            train_data = d[0].reshape((batchSize,-1,4,6)).to(device).permute(1,0,2,3)
            src_mask = model.generate_square_subsequent_mask(bptt).to(device)
            for index, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
                data, targets,dataFeat = get_batch(train_data, i)
                optimizer.zero_grad()
                if data.size(0) != bptt:
                    src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
                output = model(data, src_mask,dataFeat)
                loss = criterion(output.view(-1, ntokens), targets)/math.log(2)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                total_loss += loss.item()
                batch = batch+1

                if batch % log_interval == 0:
                    cur_loss = total_loss / log_interval
                    elapsed = time.time() - start_time

                    total_loss_list = " - "
                    printl('| epoch {:3d} | Batch {:3d} | {:4d}/{:4d} batches | '
                        'lr {:02.2f} | ms/batch {:5.2f} | '
                        'loss {:5.2f} | losslist  {} | ppl {:8.2f}'.format(
                            epoch, Batch, batch, len(train_data) // bptt, scheduler.get_last_lr()[0],
                            elapsed * 1000 / log_interval,
                            cur_loss,total_loss_list, math.exp(cur_loss)))
                    total_loss = 0

                    start_time = time.time()

                    writer.add_scalar('train_loss', cur_loss,idloss)
                    idloss+=1

            if Batch%1==0:
                save(epoch*100000+Batch,saveDict={'encoder':model.state_dict(),'idloss':idloss,'epoch':epoch,'best_val_loss':best_val_loss},modelDir=checkpointPath)

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(epoch)
        printl('-' * 89)
        scheduler.step()
        printl('-' * 89)
