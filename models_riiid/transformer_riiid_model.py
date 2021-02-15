import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding, MultiheadAttention, LayerNorm, Dropout
from torch.nn import Linear
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
import warnings
import gc
warnings.filterwarnings('ignore')
import random

#SETTINGS 
MAXLENGTH = 40
NDAY = 1
NDAY_LENGTH = MAXLENGTH * NDAY
EMBEDDING_DIM = 128
CONTENT_ID_VOCAB_SIZE = 13524
PART_VOCAB_SIZE = 8
CONTAINER_VOCAB_SIZE = 10002
USER_VOCAB_SIZE = 40001
RESPONSE_VOCAB_SIZE = 3
DAY_VOCAB_SIZE = 400
DAYS_VOCAB_SIZE = 1200
NUM_HEADS = 8
NUM_ENCODERS = 4
DROPOUT = 0.2

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = 'cpu'

class Encoder(nn.Module):
    
    def __init__(self, questions_size=CONTENT_ID_VOCAB_SIZE, responses_size=RESPONSE_VOCAB_SIZE, 
                 part_size=PART_VOCAB_SIZE, task_container_id_size = CONTAINER_VOCAB_SIZE, 
                 user_id_size = USER_VOCAB_SIZE,  day_size = DAYS_VOCAB_SIZE,
                 maxlength=NDAY_LENGTH, num_heads=NUM_HEADS, embedding_size=EMBEDDING_DIM,  
                 dropout = DROPOUT
                ):
        
        super(Encoder, self).__init__()
        self.input_length = maxlength
        #embedding layers for question, response
        #user, part, task_container_id and position
        self.embedding_ques = Embedding(num_embeddings = questions_size, 
                                        embedding_dim = embedding_size
                                       )
        
        self.embedding_response = Embedding(num_embeddings = responses_size,
                                            embedding_dim = embedding_size
                                           )
        self.embedding_user = Embedding( num_embeddings = user_id_size,
                                         embedding_dim = embedding_size
                                       )
        
        self.embedding_part = Embedding(    num_embeddings = part_size,
                                            embedding_dim = embedding_size
                                           )        
        
        self.embedding_task = Embedding(    num_embeddings = task_container_id_size,
                                            embedding_dim = embedding_size
                                           )        
        
        self.embedding_pos =  Embedding(    num_embeddings = maxlength + day_size + DAY_VOCAB_SIZE,
                                            embedding_dim = embedding_size
                                           )  
        #linear layers for day and days
        self.linear_day = Linear(maxlength, embedding_size)
        self.linear_days = Linear(maxlength, embedding_size)
        

        #multihead attention
        self.attention  = MultiheadAttention(embed_dim = embedding_size,
                                             num_heads = num_heads,
                                             dropout = dropout
                                            )
        self.dropout1 = Dropout(dropout)
        
    def __call__(self, questions, part, responses, task_container_ids, user, 
                 day, days, batch = 128, block=False):
        
        flag = len(questions.shape) != 1
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = 'cpu'

        if block:

            exercise_ids = self.embedding_ques(questions)
            task_container_ids = self.embedding_task(task_container_ids)
            parts = self.embedding_part(part)
            user = self.embedding_user(user)
            #use tanh activation for day/days
            day = F.tanh(self.linear_day(day)) 
            days = F.tanh(self.linear_days(days))
            #if we have batches
            if flag:
                user = user.unsqueeze(1)
                days = days.unsqueeze(1)
                day = day.unsqueeze(1) 
                    
            pos = torch.arange(self.input_length, device=device).unsqueeze(0)
            responses = self.embedding_response(responses)
            #add all features
            x = exercise_ids + task_container_ids + parts + responses + user + day + days
        else:
            x = questions
            pos = torch.arange(self.input_length, device=device).unsqueeze(0) 
            
        pos = self.embedding_pos(pos)
        #add positional embedding to
        #the input features        
        x = x + pos
        x = x.permute(1, 0, 2)
        size = x.shape[0]
        x_1 = x
        #I'm using upper triangular mask (np.triu)
        x, _ = self.attention(x, x, x, 
                           attn_mask = torch.from_numpy( np.triu(np.ones((size, size)), k=1).astype('bool')).to(device))
        #skip connection
        x += x_1
        x =  x.permute(1, 0, 2) 
        x =  self.dropout1(x)
        return x

encoder = Encoder().to(device)

class SPACE(nn.Module):
    def __init__(self, num_encoders=NUM_ENCODERS, embedding_size=EMBEDDING_DIM, 
                 maxlength=NDAY_LENGTH, dropout = DROPOUT):
        super(SPACE, self).__init__()
        self.maxlength = maxlength
        self.embedding_size = embedding_size
        self.encoders = nn.ModuleList([Encoder(embedding_size=embedding_size) for _ in range(num_encoders) ])
        self.linear1 = Linear(maxlength*embedding_size, 1)
        self.dropout1 = Dropout(dropout)
        
    def __call__(self, questions, parts, responses, task_container_ids, user, day, days,  batch=128):
        for ix, encoder in enumerate(self.encoders):
            if ix == 0:
                x = encoder(questions, parts, responses, task_container_ids, user, 
                            day, days, batch=batch, block=True)
            else:
                x = encoder(x, _, _, _, _, _,_, batch=batch)
        if batch>1:
            x = x.reshape(batch, self.embedding_size*self.maxlength )
            
        else:
            x = x.view(-1)
        x = F.relu(x)
        x = self.dropout1(x)
        x =  self.linear1(x)
        return x

space = SPACE()
space.to(device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
space = SPACE()
epochs = 20
#use mse as loss function
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(space.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-8)
space.to(device)
criterion.to(device)

def train_epoch(model=space, train_iterator=train_dataloader, optim=optimizer, criterion=criterion, device=device):
    model.train()

    train_loss = []
    num_corrects = 0
    num_total = 0
    labels = []
    outs = []
    tbar = tqdm(train_iterator)
    for item in tbar:
        exercise_ids = item[0].to(device).long()
        parts = item[1].to(device).long()
        responses = item[2].to(device).long()
        task_container_ids = item[3].to(device).long()
        user = item[4].to(device).long()
        day = item[5].to(device).float()
        days = item[6].to(device).float()
        label = item[7].to(device).float()
        optim.zero_grad()
        output = model(exercise_ids, parts, responses, task_container_ids, user, day,
                       days, batch = exercise_ids.shape[0])
        output = torch.reshape(output, label.shape)    
        loss = criterion(output, label)
        outs.extend(output.view(-1).data.cpu().numpy())
        labels.extend(label.view(-1).data.cpu().numpy())
        loss.backward()
        optim.step()
        train_loss.append(loss.item())
        
        tbar.set_description('train loss - {:.4f}'.format(loss))
        
    loss = np.mean(train_loss)
    mae = mean_absolute_error(labels, outs)
    return loss, mae, labels, outs

def val_epoch(model=space, val_iterator=valid_dataloader,
              criterion=criterion, device=device):
    model.eval()

    val_loss = []
    labels = []
    outs = []
    tbar = tqdm(val_iterator)
    #going through new_users_data
    for item in tbar:
        exercise_ids = item[0].to(device).long()
        parts = item[1].to(device).long()
        responses = item[2].to(device).long()
        task_container_ids = item[3].to(device).long()
        user = item[4].to(device).long()
        day = item[5].to(device).float()
        days = item[6].to(device).float()
        label = item[7].to(device).float()
        with torch.no_grad():
            output = model(exercise_ids, parts, responses, task_container_ids,
                           user, day, days, batch = exercise_ids.shape[0])
        
        output = torch.reshape(output, label.shape)
        loss = criterion(output, label)
        val_loss.append(loss.item())
        outs.extend(output.view(-1).data.cpu().numpy())
        labels.extend(label.view(-1).data.cpu().numpy())
        tbar.set_description('valid loss - {:.4f}'.format(loss))
    
    mae = mean_absolute_error(labels, outs)                     
    return np.average(val_loss), mae, labels, outs

#train model for 20 epochs
#and save checkpoints
val_losses = list()
train_losses = list()
model_dir = './model/'
for epoch in range(epochs):
    train_loss, mae , train_labels, train_outputs = train_epoch(model = space, device=device)
    print("epoch - {} train_loss - {:.2f} mae - {:.2f}".format(epoch, train_loss, mae))
    val_loss, mae, valid_labels, valid_outputs = val_epoch(model = space, device=device)
    print("epoch - {} val_loss - {:.2f} mae - {:.2f}".format(epoch, val_loss, mae))
    torch.save(space.state_dict(), os.path.join(model_dir, 'ndays-{}-epoch-{}.pt'.format(NDAY, epoch)))
    train_losses.append(train_loss)
    val_losses.append(val_loss)