import numpy as np

import imageio

from torchvision import transforms
from sklearn import cluster
from collections import Counter


class make_emotion6_color_dict:
    def __init__(self, encoder, mean, std):
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
        
        self.encoder = encoder

        self.inv_normalize = transforms.Normalize(
            mean=-1*np.divide(mean,std),
            std=1/std)

        self.anger_dict = {}
        self.disgust_dict = {}
        self.fear_dict = {}
        self.joy_dict = {}
        self.sadness_dict = {}
        self.surprise_dict = {}

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

    def make_dict(self, data_loader):
        for batch_idx, (data, targets) in enumerate(data_loader):
            data = self.inv_normalize(data)
            rgb_data = data
            rgb_data = rgb_data * 255

            targets = targets.cpu().numpy()

            for col_data, target in zip(rgb_data, targets):
                trans_data = np.transpose(col_data, (2, 1, 0))
                trans_data = trans_data.numpy()

                target = self.encoder[target]

                color = self.img_cluster(trans_data)
                c1 = self.rgb2color(color[0])
                c2 = self.rgb2color(color[1])

                if c2 == None:
                    c2 = c1
                if c1 == None:
                    c1 = c2

                color = (c1, c2)
                
                if target == 'anger':
                    if color in self.anger_dict:
                        self.anger_dict[color] += 1
                    else:
                        self.anger_dict[color] = 1
                    
                elif target == 'disgust':
                    if color in self.disgust_dict:
                        self.disgust_dict[color] += 1
                    else:
                        self.disgust_dict[color] = 1
                
                elif target == 'fear':
                    if color in self.fear_dict:
                        self.fear_dict[color] += 1
                    else:
                        self.fear_dict[color] = 1

                elif target == 'joy':
                    if color in self.joy_dict:
                        self.joy_dict[color] += 1
                    else:
                        self.joy_dict[color] = 1
                
                elif target == 'sadness':
                    if color in self.sadness_dict:
                        self.sadness_dict[color] += 1
                    else:
                        self.sadness_dict[color] = 1
                    
                elif target == 'surprise':
                    if color in self.surprise_dict:
                        self.surprise_dict[color] += 1
                    else:
                        self.surprise_dict[color] = 1
