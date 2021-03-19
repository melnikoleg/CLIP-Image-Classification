# CLIP-Image-Classification

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kingchloexx/CLIP-Image-Classification/blob/main/Multi_Domain_Pretrained_Classifier_with_CLIP.ipynb)

### Use

```python
from classify import load, classify

filename = "/content/input.jpg"

load_categories = "imagenet" #valid categories include["imagenet", "dog vs cat", "pokemon", "words in the communist manifesto"] or you can use a list of your own categories

print("loading categories")
load(load_categories)

print("classifying")
print(classify(filename))
```



#### load_categories

```python
load("imagenet") #imagenet categories
load("dog vs cat") #dog and cat as categories
load("words in the communist manifesto") #speaks for itself
load(["banana", "elephant", "monkey"]) #any custom words in a list will do as well
```



### Dependencies

I made it as simple as I could lmao, first you've gotta install the dependencies which can be done with this block of text if you're using Google Colab:

```python
import subprocess

CUDA_version = [s for s in subprocess.check_output(["nvcc", "--version"]).decode("UTF-8").split(", ") if s.startswith("release")][0].split(" ")[-1]
print("CUDA version:", CUDA_version)

if CUDA_version == "10.0":
    torch_version_suffix = "+cu100"
elif CUDA_version == "10.1":
    torch_version_suffix = "+cu101"
elif CUDA_version == "10.2":
    torch_version_suffix = ""
else:
    torch_version_suffix = "+cu110"
!git clone https://github.com/kingchloexx/CLIP-Image-Classification
! pip install torch==1.7.1{torch_version_suffix} torchvision==0.8.2{torch_version_suffix} -f https://download.pytorch.org/whl/torch_stable.html ftfy regex
!pip install ftfy

```

If you're using a conda environment, be sure to use `conda install cudatoolkit` and at the time of writing this, the command for installing the correct torch version would be

```
pip install torch==1.7.1 torchvision==0.8.2 -f https://download.pytorch.org/whl/torch_stable.html ftfy regex
```

as (at the current time) conda installs cuda 10.2.8 when using `conda install cudatoolkit`

👍
