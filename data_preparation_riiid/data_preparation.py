#Prepare Dataset

#imports
import gc
import pandas as pd
import numpy as np
import torch
import numpy as np
import pandas as pd
import warnings
import pickle
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import defaultdict
import random
import warnings
warnings.filterwarnings('ignore')

#Generate Number of Sessions Column

#SETTINGS
MAXLENGTH = 40
NDAY = 10
NDAY_LENGTH = MAXLENGTH * NDAY
data = pd.read_feather('../input/riiidtraindatamultipleformats/riiid_train.feather/riiid_train.feather')
s = (data['user_id'].value_counts() >= 100)
s = s[s.values].index
s = np.array(s)
users = s[:40000]
user_replace_dict = {}
for ix, user in enumerate(s):
    user_replace_dict[user] = ix

data = data[data['user_id'].isin(users)]
data.head()
data = data[['timestamp', 'user_id', 'content_id', 'content_type_id', 'task_container_id', 'answered_correctly']]
data = data[data['content_type_id']==False]
data.drop(columns=['content_type_id'], inplace=True)
data['timestamp'] = data['timestamp'].apply(lambda a: a/60000)
data = data.sort_values(['timestamp',], ascending=True).reset_index(drop = True)
data['sessions_continue'] = None
data['days'] = None

for user in tqdm(data.user_id.unique()):
    #43200 -> number of minutes in 1 month
    day = 1440 #number of minutes per day
    data_user = data[data.user_id == user]
    data_user_timestamp = data_user['timestamp']
    #initial time for the very first interaction
    initial = data_user_timestamp.iloc[0]
    #checks whether the interaction belongs 
    #to the first day interactions for the user
    flag = False
    #index for the current timestamp
    ix = 1
    #index for the previous timestamp
    p_ix = 0
    while ix < len(data_user):
        #variable to store number of sessions
        #for the next 30 days (not including the current day)
        session_count = 0
        if flag:
            #initial time(for the first interaction) for the current day
            initial = data_user_timestamp.iloc[ix]
            p_ix = ix
            ix += 1
            if ix >= len(data_user):
                break
        prev = initial
        time = data_user_timestamp.iloc[ix]
        #whether user had only one 
        #interaction per day or not
        switch = True
        #find the last interaction for current day
        while time - initial < 1440:
            ix += 1
            keep_stored = ix
            if ix >= len(data_user):
                break
            time = data_user_timestamp.iloc[ix]
            switch = False
        if switch:
            keep_stored = p_ix + 1
        if ix >= len(data_user):
            break
        #initial time updated
        #here by initial is meant
        #the first interaction for the next day
        #which is stored in time variable
        initial = time
        time = data_user_timestamp.iloc[ix]
        #calculate total number of sessions
        #for the next 30 days
        while time - initial <= 43200:
            if time - prev > 60.:
                session_count += 1
            prev = time
            ix += 1
            if ix >= len(data_user):
                break
            time = data_user_timestamp.iloc[ix]
        #get indices for the timestamps for the current day
        indices = data_user_timestamp.iloc[p_ix:keep_stored].index
        flag = True
        data['sessions_continue'].loc[indices] = session_count
        #set ix to be equal to keep_stored
        #in keep_stored is stored the index for
        #initial time for the next day
        ix = keep_stored

#Generate Days Column

#calculate number of days
#user used the tool so far
from tqdm import tqdm
for user in tqdm(data.user_id.unique()):
    day = 1440 #number of minutes per day
    data_user = data[data.user_id == user]
    data_user_timestamp = data_user['timestamp']
    initial = data_user_timestamp.iloc[0]
    indices = data_user_timestamp.index
    counter = 1
    ix = 1
    time = data_user_timestamp.iloc[ix]
    day = 0
    days = list()
    while ix < len(data_user):
        if time - initial < 1440:
            counter += 1
        else:
            days.extend([day]*counter)
            day += 1
            counter = 1
            initial = data_user_timestamp.iloc[ix]
        ix += 1
        if ix >= len(data_user):
            break
        time = data_user_timestamp.iloc[ix]
    days.extend([day]*counter)
    data['days'].loc[indices] = days

#Generate Day Column
data['day'] = data['timestamp']//1440

#Generate Data in Day Units


question_data = pd.read_csv('../input/riiid-test-answer-prediction/questions.csv')
question_data['content_id'] = question_data['question_id']
question_data.drop(columns=['question_id', 'bundle_id', 'correct_answer', 'tags'], inplace=True)
data = pd.merge(data, question_data, on='content_id')
grouped_data = data[['user_id', 'content_id', 'part', 'answered_correctly', 
                                                'task_container_id', 'sessions_continue', 'day', 'days'
                                    ]].groupby(['user_id'])
del data
gc.collect()
data = pd.DataFrame(grouped_data)

class DataPreparator(object):
    def __init__(self, data, maxlength=MAXLENGTH):
        super(DataPreparator, self).__init__()
        self.maxlength = maxlength
        self.data = data
        self.users = list()
        self.new_data = defaultdict()
        for user in self.data[0].values:
            self.users.append(user)

        
    def __len__(self):
        return len(self.data)
    
    def __call__(self,):
        """
        generates new data per user
        in day units and returns them
        as a dictionary
        """
        from tqdm import tqdm
        for row in tqdm(range(len(self.data))):
            #going through each row of the dataframe
            #each row in grouped dataframe is for single user
            user_data = self.data[1].iloc[row]
            user = self.data[0].iloc[row]
            user_id = user_replace_dict[user]
            group_user = user_data[['content_id', 'part', 'answered_correctly', 
                                                'task_container_id', 'sessions_continue', 'day', 'days'
                                    ]].groupby(['day'])
            group_user = pd.DataFrame(group_user)
            del user_data
            gc.collect()
            #create key for the user and stack day units
            #each day unit will contain these features in this order
            #content_ids, parts, answers, task_container_ids, user, 
            #day, days, labels (in label is meant sessions_continue)
            #the days are sorted in ascending order
            days_list = list()
            for ix in range(len(group_user)):
                interactions_per_day = group_user[1].iloc[ix]
                content_ids = interactions_per_day['content_id'].values
                parts = interactions_per_day['part'].values
                answers = interactions_per_day['answered_correctly'].values
                task_container_ids = interactions_per_day['task_container_id'].values
                labels = interactions_per_day['sessions_continue'].values[0]
                days = interactions_per_day['days'].values[0]
                day = interactions_per_day['day'].values[0]
            
                #0s should be used as padding values
                content_ids = content_ids + 1
                #part containes 1-7 values
                parts = parts
                answers = answers + 1
                task_container_ids = task_container_ids + 1
                labels = labels
                #only single label for each day
                labels = np.array(labels)
                days = np.array(days)
                user = np.array(user_id)
                day = np.array(day)

                n = len(content_ids)


                if n >= self.maxlength:
                    content_ids = content_ids[-self.maxlength :]
                    parts = parts[-self.maxlength: ]
                    task_container_ids = task_container_ids[-self.maxlength :]
                    answers = answers[-self.maxlength:]
                else:
                    content_ids = np.pad(content_ids, (self.maxlength - n, 0))
                    task_container_ids = np.pad(task_container_ids, (self.maxlength - n, 0))
                    answers = np.pad(answers, (self.maxlength - n, 0))
                    parts = np.pad(parts, (self.maxlength - n, 0))
                    
                days_list.append([content_ids, parts, answers, task_container_ids,
                                           user, day, days, labels])
            self.new_data[user_id] = days_list

        
        return self.new_data

dataprep = DataPreparator(data)
data = dataprep()

#shuffle data randomly
random.shuffle(data)
#put 99% in training data
train_size = (len(data)*98)//100
test_size = (len(data)*1)//100
keys = list(data.keys())
train_data = {k:data[k] for k in keys[:train_size]}
val_data = {k:data[k]  for k in keys[train_size:train_size+test_size] }
test_data = {k:data[k] for k in keys[-test_size:]}

class SPACE_DATASET(Dataset):
    def __init__(self, data, maxlength=MAXLENGTH, maxday=NDAY, total_max_length=NDAY_LENGTH):
        super(SPACE_DATASET, self).__init__()
        self.maxlength = maxlength
        self.total_max_length = NDAY_LENGTH
        self.maxday = NDAY
        self.data = data
        self.length = 0
        self.user_days = list()
        for user in self.data.keys():
            days_for_user = len(self.data[user])
            self.length += np.ceil(len(self.data[user])/self.maxday)
            for ix in range(0, days_for_user, 4):
                if ix + self.maxday > days_for_user:
                    self.user_days.append((user, ix, days_for_user))
                else:
                    self.user_days.append((user, ix, ix + self.maxday))

    def __len__(self):
        return len(self.user_days)
    
    def __getitem__(self, ix):
        #content_ids, parts, answers, task_container_ids, user, 
        #day, days, labels (in label is meant sessions_continue)
        user, start_ix, end_ix = self.user_days[ix]
        data_days = self.data[user][start_ix:end_ix]
        content_ids = np.array([])
        parts = np.array([])
        answers = np.array([])
        task_container_ids = np.array([])
        day = np.array([])
        days = np.array([])
        labels = 0
        for day_unit in data_days:
            content_ids_day, parts_day, answers_day, task_day, user, d1, d2, labels = day_unit
            content_ids = np.append(content_ids, content_ids_day)
            parts = np.append(parts, parts_day)
            answers = np.append(answers, answers_day)
            task_container_ids = np.append(task_container_ids, task_day)
            #padding will be needed for day/days too thus they will be incremented by one
            #in data_prep part/content_id/task_container_id are already incremented by one
            day = np.append(day, [d1 + 1]*self.maxlength)
            days = np.append(days, [d2 + 1]*self.maxlength)
        

        #get log1 for the label
        labels = np.log1p(labels)
        labels = np.array(labels)
        n = len(content_ids)

        #pad sequence if less than total_max_length
        if n < self.total_max_length:
            content_ids = np.pad(content_ids, (self.total_max_length - n, 0))
            task_container_ids = np.pad(task_container_ids, (self.total_max_length - n, 0))
            answers = np.pad(answers, (self.total_max_length - n, 0))
            parts = np.pad(parts, (self.total_max_length - n, 0))
            day = np.pad(day, (self.total_max_length - n, 0))
            days = np.pad(days, (self.total_max_length - n, 0))
            
        content_ids = torch.from_numpy(content_ids).long()
        parts = torch.from_numpy(parts).long()
        task_container_ids = torch.from_numpy(task_container_ids).long()
        answers = torch.from_numpy(answers).long()
        user = torch.from_numpy(user).long()
        day = torch.from_numpy(day).float()
        days = torch.from_numpy(days).float()
        labels = torch.from_numpy(labels).float()

        return content_ids, parts, answers, task_container_ids, user, day, days, labels

train_data = SPACE_DATASET(train_data)
val_data = SPACE_DATASET(val_data)
test_data = SPACE_DATASET(test_data)

train_dataloader = DataLoader(train_data, batch_size=256, shuffle=True, num_workers=8)
valid_dataloader = DataLoader(val_data, batch_size=256, shuffle=True, num_workers=8)
test_dataloader = DataLoader(test_data, batch_size=256, shuffle=True, num_workers=8)
item = train_data.__getitem__(0)
del train_data, val_data, test_data, data
gc.collect()