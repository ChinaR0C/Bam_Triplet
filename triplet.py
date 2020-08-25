from PIL import Image
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch.autograd import Variable


trans = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
#     transforms.Resize((224, 224)), #왜 224?
    transforms.ToTensor(), #nparray => tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #0~1값을 -0.5~0.5로 변경
])

#dataset

n_media = 7
n_content = 7

class TripletDataset(Dataset):
#     Train: For each sample (anchor) randomly chooses a positive and negative samples

    def __init__(self, dataset, kind):
        self.dataset = dataset
        self.kind = kind
        self.data = []
        self.labels = []
        self.label_to_indices = {}
        
        for m in range(n_media):
            for c in range(n_content):
                self.label_to_indices[(m,c)] = []
                
        for i in range(len(dataset)):
            e = self.dataset.__getitem__(i)
            self.data.append(e[0])
            m = int(e[1]/n_media)
            c = e[1] % n_media
            self.labels.append((m,c))
            self.label_to_indices[(m,c)].append(i)
            
    
        if self.kind == 'test':
            random_state = np.random.RandomState(29)
            triplets = []
            keys = list(self.label_to_indices.keys())
            anchor_key_index = -1
            
            for k in range(len(keys)):
                if keys[k] == self.labels[i]:
                    anchor_key_index = k
                    
            
            for i in range(len(self.data)):
                pos_index = random_state.choice(self.label_to_indices[self.labels[i]])
                
                r = list(range(len(keys)))
                r.remove(anchor_key_index)
                neg_label = keys[random_state.choice(r)]
                neg_index = random_state.choice(self.label_to_indices[neg_label])
                
                triplets.append([i, pos_index, neg_index])
            self.test_triplets = triplets
                
            
            
    def __getitem__(self, index):
        if self.kind == 'train':
            img1, label1 = self.data[index], self.labels[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])

            negative_m_range = list(range(n_media))
            negative_m_range.remove(label1[0])
            negative_m = np.random.choice(negative_m_range)

            negative_c_range = list(range(n_content))
            negative_c_range.remove(label1[1])
            negative_c = np.random.choice(negative_c_range)

            negative_label = (negative_m, negative_c)
            negative_index = np.random.choice(self.label_to_indices[negative_label])

            img2 = self.data[positive_index]
            img3 = self.data[negative_index]
            
        elif self.kind == 'test':
            img1 = self.data[self.test_triplets[index][0]]
            img2 = self.data[self.test_triplets[index][1]]
            img3 = self.data[self.test_triplets[index][2]]                
            
        else:
            print('TripletDataset Error')
            return
        
        return (img1, img2, img3), []
    
    def __len__(self):
        return len(self.dataset)
        
#ImageFoler를 통해 불러온 trainset을 DataLoader를 사용해
#Batch 형식으로 네트워크에 올리기
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

train_dataset = torchvision.datasets.ImageFolder(root = './BAM', transform = trans)
test_dataset = torchvision.datasets.ImageFolder(root = './test', transform = trans)
triplet_train_dataset = TripletDataset(train_dataset, kind = 'train')
triplet_test_dataset = TripletDataset(test_dataset, kind = 'test')

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_loader = DataLoader(triplet_train_dataset, batch_size = 1, shuffle = True, **kwargs)
test_loader = DataLoader(triplet_test_dataset, batch_size = 1, shuffle = True, **kwargs)


import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16

class StyleNet(nn.Module):
    def __init__(self):
        super(StyleNet, self).__init__()
        self.covnet = nn.Sequential(*list(vgg16(pretrained = True).features)[:29])  #vgg block5-1 conv 까지
        
        self.fc = nn.Linear(512 * 512, 2048)
        
    def gram_matrix(self, x):
        b, c, h, w = x.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = x.view(b * c, h * w)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product
        
        return G #512 * 512

#         # we 'normalize' the values of the gram matrix
#         # by dividing by the number of element in each feature maps.
#         return G.div(a * b * c * d)
    
    def forward(self, x):
        
        output = self.covnet(x)      
        output = self.gram_matrix(output)
        output = torch.flatten(output)
        output = self.fc(output)
        return output
    
class ContentNet(nn.Module):
    def __init__(self):
        super(ContentNet, self).__init__()
        self.convnet = nn.Sequential(*list(vgg16(pretrained = True).features)) 
        self.avg_pool = nn.AvgPool2d(7)
        self.fc1 = nn.Linear(512, 2048)
        
    def forward(self, x):
        output = self.convnet(x)
        output = self.avg_pool(output)
        output = torch.flatten(output)
        output = self.fc1(output)
        return output
    
class TripletNet(nn.Module):
    def __init__(self, style, content):
        super(TripletNet, self).__init__()
        self.style = style
        self.content = content
        
        
        
    def forward(self, x1, x2, x3):
        output1 = torch.cat([self.style(x1), self.content(x1)], dim = 0)
        output2 = torch.cat([self.style(x1), self.content(x2)], dim = 0)
        output3 = torch.cat([self.style(x1), self.content(x3)], dim = 0)
        
        return output1, output2, output3
    
    def get_style(self, x):
        return self.style(x)
    def get_content(self, x):
        return self.content(x)

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """
    
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative, size_average = True):
        distance_positive = (anchor - positive).pow(2).sum()  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum()  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()

margin = 1
style_net = StyleNet()
content_net = ContentNet()
model = TripletNet(style_net, content_net)

if cuda:
    model.cuda()
loss_fn = TripletLoss(margin)
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
n_epochs = 20


from tqdm import tqdm

def train_epoch(train_loader, model, loss_fn, optimizer, cuda):
    model.train()
    losses = []
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()


        optimizer.zero_grad()
        outputs = model(*data)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs
        if target is not None:
            target = (target,)
            loss_inputs += target

        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    
    total_loss /= (batch_idx + 1)
    return total_loss
    
#     for batch_idx, data in enumerate(train_loader):
#         optimizer.zero_grad()
#         if cuda:
#             data = tuple(d.cuda() for d in data)
        
#         outputs = model(*data)
#         loss = loss_fn(outputs[0], outputs[1], outputs[2])
#         total_loss += loss
#         loss.backward()
#         optimizer.step()
        
#     return total_loss/(len(train_loader)/batch_size)
        
        
# def test_epoch(test_loader, model, loss_fn, optimizer, cuda):
#     model.eval()
#     total_loss = 0
    
#     for batch_idx, data in enumerate(test_loader):
#         if cuda:
#             data = tuple(d.cuda() for d in data)

#         outputs = model(*data)
#         loss = loss_fn(outputs[0], outputs[1], outputs[2])
#         total_loss += loss

    
def test_epoch(val_loader, model, loss_fn, cuda):
    with torch.no_grad():
        model.eval()
        val_loss = 0
        for batch_idx, (data, target) in enumerate(tqdm(val_loader)):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

    return val_loss       
        
        
            
        
def fit(train_loader, val_loader, model, loss_fn, optimizer,  n_epochs, cuda):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """

    # Train stage
    for epoch in range(n_epochs):       
        train_loss = train_epoch(train_loader, model, loss_fn, optimizer, cuda)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)

        val_loss = test_epoch(val_loader, model, loss_fn, cuda)
        val_loss /= len(val_loader)

        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,
                                                                                 val_loss)

        print(message)

fit(train_loader, test_loader, model, loss_fn, optimizer, n_epochs, cuda)
    
        
        
        
            
        





