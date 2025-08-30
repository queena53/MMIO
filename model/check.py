import numpy as np
file_path="pretrained_resnet.npz"
poem=np.load(file_path,allow_pickle=True)
print(poem.files)