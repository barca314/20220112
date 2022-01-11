
import torch as pt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

from model import Model

from torch.utils.data import DataLoader, TensorDataset

VERBOSE = False

epoch_size = 10
bag_size= 100
num_feature = 128
learning_rate = 1e-4
momentum = 0.2


# load dataset
data = pd.read_csv('./mnist/mnist_train.csv').values
data_test = pd.read_csv('./mnist/mnist_test.csv').values


# filter the zeros

def extract_data(d):
    return d[d[:, 0] == 0], d[d[:, 0] == 7]

zeros, sevens = extract_data(data)
zeros_test, sevens_test = extract_data(data_test)

if VERBOSE:
    print(zeros.shape, sevens.shape)
    print(zeros_test.shape, sevens_test.shape)



if VERBOSE:
    # plot the images to jusitify correctness
    size = 4

    plt.figure(figsize=(size, size))
    for i in range(size*size):
        plt.subplot(size, size, i+1)
        plt.imshow(zeros[i, 1:].reshape(28, 28), cmap='gray')

    plt.show()

    plt.figure(figsize=(size, size))
    for i in range(size*size):
        plt.subplot(size, size, i+1)
        plt.imshow(sevens[i, 1:].reshape(28, 28), cmap='gray')

    plt.show()


zeros_image = zeros[:, 1:].reshape(-1, 28, 28)
sevens_image = sevens[:, 1:].reshape(-1, 28, 28)

def generate_data(ratio, isTest=False):

    ret = np.zeros((bag_size, 28, 28))

    zeros_num = math.ceil(ratio * bag_size)
    sevens_num = bag_size - zeros_num


    if not isTest:

        zeros_ind = np.random.randint(0, zeros.shape[0], zeros_num)
        sevens_ind = np.random.randint(0, sevens.shape[0], sevens_num)
        ret[:zeros_num, :] = zeros_image[zeros_ind]
        ret[zeros_num:, :] = sevens_image[sevens_ind]
    else:

        zeros_ind = np.random.randint(0, zeros_test.shape[0], zeros_num)
        sevens_ind = np.random.randint(0, sevens_test.shape[0], sevens_num)
        ret[:zeros_num, :] = zeros_test[zeros_ind, 1:].reshape(-1, 28, 28)
        ret[zeros_num:, :] = sevens_test[zeros_ind, 1:].reshape(-1, 28, 28)

    return ret

# test function


train_size = 10000
batch_size = 10

test_size = 100


train_img = np.zeros((train_size, bag_size, 28, 28))
train_lab = np.random.rand(train_size)

for i in range(train_size):
    train_img[i] = generate_data(train_lab[i])


test_img = np.zeros((test_size, bag_size, 28, 28))
test_lab = np.random.rand(test_size)

for i in range(test_size):
    test_img[i] = generate_data(test_lab[i])

train_img = pt.from_numpy(train_img).float()
train_lab = pt.from_numpy(train_lab).float()

test_img = pt.from_numpy(test_img).float()
test_lab = pt.from_numpy(test_lab).float()


trian_loader = DataLoader(TensorDataset(train_img, train_lab), batch_size, True)
test_loader = DataLoader(TensorDataset(test_img, test_lab), batch_size, True)
# test_loader = DataLoader(test_img, batch)

device=pt.device("cuda" if pt.cuda.is_available() else "cpu")

model = Model(1, bag_size, num_feature, 21, 0.1).to(device)

criterion = pt.nn.L1Loss()
optim = pt.optim.SGD(model.parameters(), lr=learning_rate)


for epoch in range(epoch_size):

    loss_list = []

    model.train()
    for x, y in trian_loader:
        x = x.view(-1, 1, 28, 28).to(device)
        y = y.to(device)
        
        logits = model(x).squeeze()
        loss = criterion(logits, y)
        optim.zero_grad()
        loss.backward()
        optim.step()

    model.eval()

    for x, y in test_loader:
        x = x.view(-1, 1, 28, 28).to(device)
        y = y.to(device)

        logits = model(x).squeeze()
        loss = criterion(logits, y)

        loss_list.append(loss.item())

    print(f'{epoch + 1} epoch: ===== loss:', np.array(loss_list).mean())
    
    if epoch % 10 == 0:
        print(logits)
        print(y)



