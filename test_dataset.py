from torch.utils.data import Dataset, DataLoader
import h5py
from dataset import EEGdata2


data_set = EEGdata2(r"water2/motor_imagery109_2s/motor_imagery109_2s.h5")

data_loader = DataLoader(data_set, batch_size=32, shuffle = False, )

for i, ele in enumerate(data_loader):
    data, label, person = ele
    print( person)
    if i>1000:
        break
    