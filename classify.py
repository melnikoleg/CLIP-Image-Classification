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
# create categories

# load imagenet categories
c_encs=[]
categories = []
def load(categorylist):
    global c_encs
    global categories
    load_categories = categorylist #@param ["imagenet", "dog vs cat", "pokemon", "words in the communist manifesto", "other (open this cell and write them into a list of strings)"]
    # filename=sys.argv[2]
    if(load_categories not in ["imagenet", "dog vs cat", "pokemon", "words in the communist manifesto", "other (open this cell and write them into a list of strings)"]):
        print("The only supported categories currently are: ")
        print(["imagenet", "dog vs cat", "pokemon", "words in the communist manifesto"])
    if(load_categories=="imagenet"):
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
    # encode categories with clip
    c_encs = [perceptor.encode_text(clip.tokenize(category).cuda()).detach().clone() for category in categories]


#@title classify
import PIL
# encode image and classify
# file_url = "https://blobcdn.same.energy/b/f3/f1/f3f13a482a241d15b0571b9b4281f20c5d5f3755"#@param {type:"string"}
# !wget "$file_url" -O /content/input.jpg -q
# filename = "/content/input.jpg"
def classify(filename):
    im_enc = perceptor.encode_image(preprocess(Image.open(filename)).unsqueeze(0).to("cpu"))
    distances = [torch.cosine_similarity(e, im_enc).item() for e in c_encs]
    # print("#"*30)
    # print("Category: ",categories[int(distances.index(max(distances)))])
    # print("#"*30)
    # display.display(display.Image(filename))

    # resize image to fit in console better
    # base_width = 360
    # image = Image.open(filename)
    # width_percent = (base_width / float(image.size[0]))
    # hsize = int((float(image.size[1]) * float(width_percent)))
    # image = image.resize((base_width, hsize), PIL.Image.ANTIALIAS)
    # display.display(display.Image(image))
    return categories[int(distances.index(max(distances)))]