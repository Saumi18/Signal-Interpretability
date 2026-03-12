import os
import numpy as np
import torch
from torch.utils.data import Dataset

modulation_map = {
"BPSK":0,
"QPSK":1,
"8PSK":2,
"16QAM":3,
"64QAM":4,
"2FSK":5,
"4FSK":6,
"GMSK":7,
"AM":8,
"FM":9
}


class RFdataset(Dataset):

    def __init__(self,root,task):

        self.samples = []
        self.task = task

        if task == "jam":

            self.load_jam_dataset(root)

        elif task == "mix":

            self.load_mix_dataset(root)

        elif task == "mod":

            self.load_mod_dataset(root)

        elif task == "dual":

            self.load_dual_dataset(root)


    def load_jam_dataset(self,root):

        clean_path = os.path.join(root,"clean")
        jam_path = os.path.join(root,"jammed")

        for mod in os.listdir(clean_path):

            folder = os.path.join(clean_path,mod)

            for file in os.listdir(folder):

                self.samples.append((os.path.join(folder,file),0))


        for mod in os.listdir(jam_path):

            folder = os.path.join(jam_path,mod)

            for file in os.listdir(folder):

                self.samples.append((os.path.join(folder,file),1))


    def load_mix_dataset(self,root):

        clean_path = os.path.join(root,"clean")
        mixed_path = os.path.join(root,"mixed")

        for mod in os.listdir(clean_path):

            folder = os.path.join(clean_path,mod)

            for file in os.listdir(folder):

                self.samples.append((os.path.join(folder,file),0))


        for pair in os.listdir(mixed_path):

            folder = os.path.join(mixed_path,pair)

            for file in os.listdir(folder):

                self.samples.append((os.path.join(folder,file),1))


    def load_mod_dataset(self,root):

        clean_path = os.path.join(root,"clean")

        for mod in os.listdir(clean_path):

            folder = os.path.join(clean_path,mod)

            label = modulation_map[mod]

            for file in os.listdir(folder):

                self.samples.append((os.path.join(folder,file),label))


    def load_dual_dataset(self,root):

        mixed_path = os.path.join(root,"mixed")

        for pair in os.listdir(mixed_path):

            mod1,mod2 = pair.split("_")

            label = np.zeros(10)

            label[modulation_map[mod1]] = 1
            label[modulation_map[mod2]] = 1

            folder = os.path.join(mixed_path,pair)

            for file in os.listdir(folder):

                self.samples.append((os.path.join(folder,file),label))


    def __len__(self):

        return len(self.samples)


    def __getitem__(self,idx):

        path,label = self.samples[idx]

        spec = np.load(path)

        spec = torch.tensor(spec).float().unsqueeze(0)

        label = torch.tensor(label)

        return spec,label