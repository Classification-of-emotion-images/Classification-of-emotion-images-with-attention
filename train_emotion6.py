import numpy as np
import matplotlib.pyplot as plt
import random
import time
import math

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms
from torch.autograd import Variable
from sklearn import metrics
from collections import Counter
from sklearn import cluster

from normalization import Normalize


class train_emotion6:
    def __init__(self, mean, std, attention=False, attention_list=None, random_seed=21):
        """
        Args:
            mean (float): ===========================================================
            std (float): ============================================================
            attention_list (list): Receive a list of colors you want to give attention
            random_seed (int): Seed to fix the result
        """
        
        self.attention = attention
        self.attention_list = attention_list

        red = [255, 0, 0]
        orange = [255, 165, 0]
        yellow = [255, 255, 0]
        green = [0, 128, 0]
        blue = [0, 0, 255]
        indigo = [75, 0, 130]
        purple = [128, 0, 128]
        turquoise = [64, 224, 208]
        pink = [255, 192, 203]
        magenta = [255, 0, 255]
        brown = [165, 42, 42]
        gray = [128, 128, 128]
        silver = [192, 192, 192]
        gold = [255, 215, 0]
        white = [255, 255, 255]
        black = [0, 0, 0]
        self.colors = [red, orange, yellow, green, blue, indigo, purple, turquoise,
                pink, magenta, brown, gray, silver, gold, white, black]

        # emotion6, two_color
        self.anger_dict = {}
        self.sadness_dict = {}
        self.disgust_dict = {}
        self.fear_dict = {}
        self.joy_dict = {}
        self.surprise_dict = {}

        self.inv_normalize = transforms.Normalize(
            mean=-1*np.divide(mean, std),
            std=1/std)

        if random_seed is not None:
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed(random_seed)
            #torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
            #torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            np.random.seed(random_seed)
            random.seed(random_seed)

        self.model = None
        self.lr = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.earlystop = None
        self.color_attention_dict = {}

    # train
    def train(self, dataloaders, criterion, num_epochs, batch_size, patience):
        self.model.to(self.device)
        best_acc = 0.0
        phases = dataloaders.keys()

        train_losses = list()
        train_acc = list()
        valid_losses = list()
        valid_acc = list()

        # EarlyStopping
        if (patience != None):
            self.earlystop = EarlyStopping(patience=patience, verbose=True)

        for epoch in range(1, num_epochs + 1):
            print('Epoch {}/{}'.format(epoch, num_epochs))
            print('----------')
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

            # if (epoch % 10 == 0):
            #     self.lr *= 0.9

            for phase in phases:
                # Train
                if phase == 'train':

                    self.model.train()
                    running_loss = 0.0
                    running_corrects = 0
                    total = 0
                    j = 0

                    for batch_idx, (data, target) in enumerate(dataloaders[phase]):
                        data, target = Variable(data), Variable(target)
                        data = data.type(torch.cuda.FloatTensor)
                        target = target.type(torch.cuda.LongTensor)
                        optimizer.zero_grad()
                        output = self.model(data)
                        loss = criterion(output, target)
                        _, preds = torch.max(output, 1)
                        running_corrects = running_corrects + torch.sum(preds == target.data)
                        running_loss += loss.item() * data.size(0)
                        j = j + 1

                        loss.backward()
                        optimizer.step()
                                                                                                 
                    epoch_acc = running_corrects.double() / (len(dataloaders[phase])*batch_size)
                    epoch_loss = running_loss / (len(dataloaders[phase])*batch_size)

                    train_losses.append(epoch_loss)
                    train_acc.append(epoch_acc)
                
                # Valid
                else:
                    with torch.no_grad():
                        self.model.eval()
                        running_loss = 0.0
                        running_corrects = 0
                        total = 0
                        j = 0

                        for batch_idx, (data, target) in enumerate(dataloaders[phase]):
                            if self.attention:
                                rgb_data = self.inv_normalize(data)
                                rgb_data = rgb_data * 255

                            data, target = Variable(data), Variable(target)
                            data = data.type(torch.cuda.FloatTensor)
                            target = target.type(torch.cuda.LongTensor)
                            optimizer.zero_grad()
                            output = self.model(data)

                            if self.attention:
                                trans_output = []
                                for output, col_data in zip(output, rgb_data):
                                    trans_data = np.transpose(col_data, (2, 1, 0))
                                    trans_data = trans_data.numpy()

                                    if str(list(trans_data)) in self.color_attention_dict:
                                        color = self.color_attention_dict[str(list(trans_data))]
                                    else:
                                        color = self.img_cluster(trans_data)
                                        c1 = self.rgb2color(color[0])
                                        c2 = self.rgb2color(color[1])
                                        if c2 == None:
                                            c2 = c1
                                        if c1 == None:
                                            c1 = c2
                                        color = (c1, c2)
                                        self.color_attention_dict[str(list(trans_data))] = color

                                    # attention with math.exp
                                    if self.attention == 'log':
                                        if self.attention_list is None:
                                            output[0] = output[0] * 0.9 ** (self.anger_dict.get(color, 0) / sum(self.anger_dict.values())) # anger
                                            output[1] = output[1] * 0.9 ** (self.disgust_dict.get(color, 0) / sum(self.disgust_dict.values())) # disgust
                                            output[2] = output[2] * 0.9 ** (self.fear_dict.get(color, 0) / sum(self.fear_dict.values())) # fear
                                            output[3] = output[3] * 0.9 ** (self.joy_dict.get(color, 0) / sum(self.joy_dict.values())) # joy
                                            output[4] = output[4] * 0.9 ** (self.sadness_dict.get(color, 0) / sum(self.sadness_dict.values())) # sadness
                                            output[5] = output[5] * 0.9 ** (self.surprise_dict.get(color, 0) / sum(self.surprise_dict.values())) # surprise
                                        else:
                                            if color in self.attention_list:
                                                output[0] = output[0] * 0.9 ** (self.anger_dict.get(color, 0) / sum(self.anger_dict.values())) # anger
                                                output[1] = output[1] * 0.9 ** (self.disgust_dict.get(color, 0) / sum(self.disgust_dict.values())) # disgust
                                                output[2] = output[2] * 0.9 ** (self.fear_dict.get(color, 0) / sum(self.fear_dict.values())) # fear
                                                output[3] = output[3] * 0.9 ** (self.joy_dict.get(color, 0) / sum(self.joy_dict.values())) # joy
                                                output[4] = output[4] * 0.9 ** (self.sadness_dict.get(color, 0) / sum(self.sadness_dict.values())) # sadness
                                                output[5] = output[5] * 0.9 ** (self.surprise_dict.get(color, 0) / sum(self.surprise_dict.values()))  # surprise

                                    # # attention with math.log
                                    else:
                                        if self.attention_list is None:
                                            output[0] = output[0] * math.log((sum(self.anger_dict.values()) / self.anger_dict.get(color, 1e-7))) # anger
                                            output[1] = output[1] * math.log((sum(self.disgust_dict.values()) / self.disgust_dict.get(color, 1e-7))) # disgust
                                            output[2] = output[2] * math.log((sum(self.fear_dict.values()) / self.fear_dict.get(color, 1e-7))) # fear
                                            output[3] = output[3] * math.log((sum(self.joy_dict.values()) / self.joy_dict.get(color, 1e-7))) # joy
                                            output[4] = output[4] * math.log((sum(self.sadness_dict.values()) / self.sadness_dict.get(color, 1e-7))) # sadness
                                            output[5] = output[5] * math.log((sum(self.surprise_dict.values()) / self.surprise_dict.get(color, 1e-7))) # surprise
                                        else:
                                            if color in self.attention_list:
                                                output[0] = output[0] * math.log((sum(self.anger_dict.values()) / self.anger_dict.get(color, 1e-7))) # anger
                                                output[1] = output[1] * math.log((sum(self.disgust_dict.values()) / self.disgust_dict.get(color, 1e-7))) # disgust
                                                output[2] = output[2] * math.log((sum(self.fear_dict.values()) / self.fear_dict.get(color, 1e-7))) # fear
                                                output[3] = output[3] * math.log((sum(self.joy_dict.values()) / self.joy_dict.get(color, 1e-7))) # joy
                                                output[4] = output[4] * math.log((sum(self.sadness_dict.values()) / self.sadness_dict.get(color, 1e-7))) # sadness
                                                output[5] = output[5] * math.log((sum(self.surprise_dict.values()) / self.surprise_dict.get(color, 1e-7)))  # surprise
                                    
                                    output = output.cpu()
                                    output = output.data.numpy()

                                    trans_output.append(output)

                            if self.attention:
                                trans_output = torch.as_tensor(trans_output)
                                trans_output = trans_output.cuda().requires_grad_(True)
                                output = trans_output
                        
                            loss = criterion(output, target)
                            _, preds = torch.max(output, 1)
                            running_corrects = running_corrects + torch.sum(preds == target.data)
                            running_loss += loss.item() * data.size(0)
                            j = j + 1

                        epoch_acc = running_corrects.double() / (len(dataloaders[phase])*batch_size)
                        epoch_loss = running_loss / (len(dataloaders[phase])*batch_size)

                        valid_losses.append(epoch_loss)
                        valid_acc.append(epoch_acc)

                print('{} Epoch: {}\tLoss: {:.6f} \tAcc: {:.6f}'.format(phase, epoch, running_loss / (j * batch_size), running_corrects.double() / (j * batch_size)))
                
                if phase == 'valid' and (patience != None):
                    self.earlystop(epoch_loss, self.model)  # early stop with valid loss 

            # print('EalryStop :', self.earlystop.early_stop)

            if (patience != None) and (self.earlystop.early_stop):
                print("Early stopping")
                self.model.load_state_dict(torch.load('./checkpoint.pt'))
                break

            # print('{} Accuracy: '.format(phase),epoch_acc.item())
            print()

        return train_losses, train_acc, valid_losses, valid_acc

    # test
    def test(self, dataloader, criterion, batch_size):
        with torch.no_grad():
            self.model.eval()
            running_corrects = 0
            running_loss = 0
            pred = []
            true = []
            pred_wrong = []
            true_wrong = []
            image = []
            # sm = nn.LogSoftmax(dim=1)

            for batch_idx, (data, target) in enumerate(dataloader):
                if self.attention:
                    rgb_data = self.inv_normalize(data)
                    rgb_data = rgb_data * 255
                    
                data, target = Variable(data), Variable(target)
                data = data.type(torch.cuda.FloatTensor)
                target = target.type(torch.cuda.LongTensor)
                output = self.model(data)
                
                if self.attention:
                    trans_output = []
                    for output, col_data in zip(output, rgb_data):
                        trans_data = np.transpose(col_data, (2, 1, 0))
                        trans_data = trans_data.numpy()

                        if str(list(trans_data)) in self.color_attention_dict:
                            color = self.color_attention_dict[str(list(trans_data))]
                        else:
                            color = self.img_cluster(trans_data)
                            c1 = self.rgb2color(color[0])
                            c2 = self.rgb2color(color[1])
                            if c2 == None:
                                c2 = c1
                            if c1 == None:
                                c1 = c2
                            color = (c1, c2)
                            self.color_attention_dict[str(list(trans_data))] = color

                        # attention with math.exp
                        if self.attention == 'exp':
                            if self.attention_list is None:
                                output[0] = output[0] * 0.9 ** (self.anger_dict.get(color, 0) / sum(self.anger_dict.values())) # anger
                                output[1] = output[1] * 0.9 ** (self.disgust_dict.get(color, 0) / sum(self.disgust_dict.values())) # disgust
                                output[2] = output[2] * 0.9 ** (self.fear_dict.get(color, 0) / sum(self.fear_dict.values())) # fear
                                output[3] = output[3] * 0.9 ** (self.joy_dict.get(color, 0) / sum(self.joy_dict.values())) # joy
                                output[4] = output[4] * 0.9 ** (self.sadness_dict.get(color, 0) / sum(self.sadness_dict.values())) # sadness
                                output[5] = output[5] * 0.9 ** (self.surprise_dict.get(color, 0) / sum(self.surprise_dict.values())) # surprise
                            else:
                                if color in self.attention_list:
                                    output[0] = output[0] * 0.9 ** (self.anger_dict.get(color, 0) / sum(self.anger_dict.values())) # anger
                                    output[1] = output[1] * 0.9 ** (self.disgust_dict.get(color, 0) / sum(self.disgust_dict.values())) # disgust
                                    output[2] = output[2] * 0.9 ** (self.fear_dict.get(color, 0) / sum(self.fear_dict.values())) # fear
                                    output[3] = output[3] * 0.9 ** (self.joy_dict.get(color, 0) / sum(self.joy_dict.values())) # joy
                                    output[4] = output[4] * 0.9 ** (self.sadness_dict.get(color, 0) / sum(self.sadness_dict.values())) # sadness
                                    output[5] = output[5] * 0.9 ** (self.surprise_dict.get(color, 0) / sum(self.surprise_dict.values()))  # surprise
                        
                        # attention with math.log
                        elif self.attention == 'log':
                            if self.attention_list is None:
                                output[0] = output[0] * math.log((sum(self.anger_dict.values()) / self.anger_dict.get(color, 1e-7))) # anger
                                output[1] = output[1] * math.log((sum(self.disgust_dict.values()) / self.disgust_dict.get(color, 1e-7))) # disgust
                                output[2] = output[2] * math.log((sum(self.fear_dict.values()) / self.fear_dict.get(color, 1e-7))) # fear
                                output[3] = output[3] * math.log((sum(self.joy_dict.values()) / self.joy_dict.get(color, 1e-7))) # joy
                                output[4] = output[4] * math.log((sum(self.sadness_dict.values()) / self.sadness_dict.get(color, 1e-7))) # sadness
                                output[5] = output[5] * math.log((sum(self.surprise_dict.values()) / self.surprise_dict.get(color, 1e-7))) # surprise
                            else:
                                if color in self.attention_list:
                                    output[0] = output[0] * math.log((sum(self.anger_dict.values()) / self.anger_dict.get(color, 1e-7))) # anger
                                    output[1] = output[1] * math.log((sum(self.disgust_dict.values()) / self.disgust_dict.get(color, 1e-7))) # disgust
                                    output[2] = output[2] * math.log((sum(self.fear_dict.values()) / self.fear_dict.get(color, 1e-7))) # fear
                                    output[3] = output[3] * math.log((sum(self.joy_dict.values()) / self.joy_dict.get(color, 1e-7))) # joy
                                    output[4] = output[4] * math.log((sum(self.sadness_dict.values()) / self.sadness_dict.get(color, 1e-7))) # sadness
                                    output[5] = output[5] * math.log((sum(self.surprise_dict.values()) / self.surprise_dict.get(color, 1e-7)))  # surprise
                        
                        output = output.cpu()
                        output = output.data.numpy()
                        trans_output.append(output)

                if self.attention:
                    trans_output = torch.as_tensor(trans_output)
                    trans_output = trans_output.cuda().requires_grad_(True)
                    output = trans_output

                loss = criterion(output, target)
                # output = sm(output)
                _, preds = torch.max(output, 1)
                running_corrects = running_corrects + torch.sum(preds == target.data)
                running_loss += loss.item() * data.size(0)
                preds = preds.cpu().numpy()
                target = target.cpu().numpy()
                preds = np.reshape(preds, (len(preds), 1))
                target = np.reshape(target, (len(preds), 1))
                data = data.cpu().numpy()

                for i in range(len(preds)):
                    pred.append(preds[i])
                    true.append(target[i])
                    if(preds[i] != target[i]):
                        pred_wrong.append(preds[i])
                        true_wrong.append(target[i])
                        image.append(data[i])

            epoch_acc = running_corrects.double()/(len(dataloader)*batch_size)
            epoch_loss = running_loss/(len(dataloader)*batch_size)

            print(epoch_acc, epoch_loss)

            return true, pred, image, true_wrong, pred_wrong, epoch_acc, epoch_loss

    def error_plot(self, loss):
        plt.figure(figsize=(10, 5))
        plt.plot(loss)
        plt.title("Valid loss plot")
        plt.xlabel("epochs")
        plt.ylabel("Loss")
        plt.show()


    def acc_plot(self, acc):
        plt.figure(figsize=(10, 5))
        plt.plot(acc)
        plt.title("Valid accuracy plot")
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.show()

    # To plot the wrong predictions given by model
    def wrong_plot(self, n_figures, true, ima, pred, encoder, inv_normalize):
        print('Classes in order Actual and Predicted')
        n_row = int(n_figures/3)
        fig, axes = plt.subplots(figsize=(14, 10), nrows=n_row, ncols=3)
        for ax in axes.flatten():
            a = random.randint(0, len(true)-1)

            image, correct, wrong = ima[a], true[a], pred[a]
            image = torch.from_numpy(image)
            correct = int(correct)
            c = encoder[correct]
            wrong = int(wrong)
            w = encoder[wrong]
            f = 'A:'+c + ',' + 'P:'+w
            if inv_normalize != None:
                image = inv_normalize(image)
            image = image.numpy().transpose(1, 2, 0)
            im = ax.imshow(image)
            ax.set_title(f)
            ax.axis('off')
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred, classes,
                            normalize=False,
                            title=None,
                            cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = metrics.confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            # ... and label them with the respective list entries
            xticklabels=classes, yticklabels=classes,
            title=title,
            ylabel='True label',
            xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        return ax

    def performance_matrix(self, true, pred):
        precision = metrics.precision_score(true, pred, average='macro')
        recall = metrics.recall_score(true, pred, average='macro')
        accuracy = metrics.accuracy_score(true, pred)
        f1_score = metrics.f1_score(true, pred, average='macro')
        print('Precision: {} Recall: {}, Accuracy: {}: ,f1_score: {}'.format(
            precision* 100, recall* 100, accuracy* 100, f1_score* 100))
            
    def most_common_color(self, candidates):
        assert isinstance(candidates, list), 'Must be a list type'
        if len(candidates) == 0:
            return None
        return Counter(candidates).most_common(n=1)[0][0]

    def img_cluster(self, img):
        img_2d = img.reshape((img.shape[0] * img.shape[1], img.shape[2]))

        kmeans = cluster.KMeans(n_clusters=7)
        kmeans.fit(img_2d)

        cluster_centers = kmeans.cluster_centers_
        centers = []
        for c in range(len(cluster_centers)):
            k = []
            for rgb in range(len(self.colors)):
                k.append(sum(abs(cluster_centers[c]-self.colors[rgb])))
            centers.append(self.colors[k.index(min(k))])
        
        for j in range(len(centers)):
            centers[j] = str(centers[j])
            
        first = self.most_common_color(centers)
        for _ in range(len(centers)):
            if first in centers:
                centers.remove(first)
        second = self.most_common_color(centers)
        if second == None:
            return [first, first]
        
        return [first, second]
    
    def rgb2color(self, x):
        if x == '[255, 0, 0]':
            return 'red'
        elif x == '[255, 165, 0]':
            return 'orange'
        elif x == '[255, 255, 0]':
            return 'yellow'
        elif x == '[0, 128, 0]':
            return 'green'
        elif x == '[0, 0, 255]':
            return 'blue'
        elif x == '[75, 0, 130]':
            return 'indigo'
        elif x == '[128, 0, 128]':
            return 'purple'
        elif x == '[64, 224, 208]':
            return 'turquoise'
        elif x == '[255, 192, 203]':
            return 'pink'
        elif x == '[255, 0, 255]':
            return 'magenta'
        elif x == '[165, 42, 42]':
            return 'brown'
        elif x == '[128, 128, 128]':
            return 'gray'
        elif x == '[192, 192, 192]':
            return 'silver'
        elif x == '[255, 215, 0]':
            return 'gold'
        elif x == '[255,  255, 255]':
            return 'white'
        elif x == '[0, 0, 0]':
            return 'black'


# EarlyStopping
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss
