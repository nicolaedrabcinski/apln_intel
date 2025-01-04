from pathlib import Path

from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from torchvision import datasets, transforms, models

from tqdm.auto import tqdm

from timeit import default_timer as timer

import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import random
import streamlit as st

# Set title
st.title("Intel Dataset Image Classification")



