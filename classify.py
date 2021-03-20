import torch
import numpy as np
from PIL import Image
import os
import random
from IPython import display
from IPython.core.interactiveshell import InteractiveShell
import subprocess
InteractiveShell.ast_node_interactivity = "all"
import glob
import clip
perceptor, preprocess = clip.load('ViT-B/32')
import sys
c_encs=[]
categories = []
def load(categorylist):
    global c_encs
    global categories
    load_categories = categorylist #@param ["imagenet", "dog vs cat", "pokemon", "words in the communist manifesto", "other (open this cell and write them into a list of strings)"]
    # filename=sys.argv[2]
    if(load_categories not in ["emojis", "imagenet", "dog vs cat", "pokemon", "words in the communist manifesto", "other (open this cell and write them into a list of strings)"]):
        # print("The only supported categories currently are: ")
        # print(["imagenet", "dog vs cat", "pokemon", "words in the communist manifesto"])
        categories = categorylist
    elif(load_categories=="imagenet"):
        # !wget https://gist.githubusercontent.com/aaronpolhamus/964a4411c0906315deb9f4a3723aac57/raw/aa66dd9dbf6b56649fa3fab83659b2acbf3cbfd1/map_clsloc.txt -q
        import pandas as pd
        categories = pd.read_csv("categories/map_clsloc.txt", sep=" ", header = None)[2]
        for category in range(len(categories)):
            categories[category] = categories[category].replace("_", " ")
    elif(load_categories=="dog vs cat"):
        categories = ["dog", "cat"]
    elif(load_categories=="pokemon"):
        # !wget https://gist.githubusercontent.com/kingchloexx/339d43fe9ce0c77634fedacd2a8c1e14/raw/9fc5c32ef1619df4a32ff79b1d986c3e7d559634/pokemon.txt -q
        import pandas as pd
        categories = pd.read_csv("categories/pokemon.txt", sep=".", header=None)[1]
    elif(load_categories=="words in the communist manifesto"):
        # !wget http://www.gutenberg.org/cache/epub/61/pg61.txt -O communism.txt -q
        ccc = open("categories/communism.txt", "r").read().split()
        categories = []
        for i in ccc:
            if i not in categories:
                categories.append(i)
    elif(load_categories=="emojis"):
        categories = open("categories/emojis.txt", "r").readlines()
    c_encs = [perceptor.encode_text(clip.tokenize(category).cuda()).detach().clone() for category in categories]


import PIL
def classify(filename, return_raw=False):
    im_enc = perceptor.encode_image(preprocess(Image.open(filename)).unsqueeze(0).to("cpu"))
    distances = [torch.cosine_similarity(e, im_enc).item() for e in c_encs]

    if(return_raw==False):
        return categories[int(distances.index(max(distances)))]
    else:
        return distances
def encode(object):
    o = object.lower()
    if("jpg" in o[-5:]) or ("png" in o[-5:]) or ("jpeg" in o[-5:]):
        return perceptor.encode_image(preprocess(Image.open(object)).unsqueeze(0).to("cpu"))
    else:
        return perceptor.encode_text(clip.tokenize(object).cuda()).detach().clone()
