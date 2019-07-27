from charades_dataset_full import load_rgb_frames, load_flow_frames, image_to_tensor
from pytorch_i3d import InceptionI3d
import numpy as np
import torch.utils.data as data_utl
import videotransforms
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch
import os

def build_model(load_model='', mode="rgb"):
    if mode == 'flows':
        i3d = InceptionI3d(400, in_channels=2)
    else:
        i3d = InceptionI3d(400, in_channels=3)
    i3d.replace_logits(157)
    i3d.load_state_dict(torch.load(load_model))
    i3d.train(False)  # Set model to evaluate mode
    return i3d


def eval_model(model, data, save_dir):
    inputs, labels, name = data
    b, c, t, h, w = inputs.shape
    if t > 100:
        #features = []
        for start in range(1, t-56, 100):
            end = min(t-1, start+100+26)
            start = max(1, start-24)
            print('now',name, start)
            ip = Variable(torch.from_numpy(inputs.numpy()[:, :, start:end]), volatile=True)
            feature = model.extract_features(ip).squeeze(
                0).permute(1, 2, 3, 0).data.cpu().numpy()
            np.save(os.path.join(save_dir, name[0] + "__" + str(start)), feature), 
        #np.save(os.path.join(save_dir, name[0]),
        #        np.concatenate(features, axis=0))
    else:
        # wrap them in Variable
        inputs = Variable(inputs, volatile=True)
        features = model.extract_features(inputs)
        np.save(os.path.join(save_dir, name[0]), features.squeeze(
            0).permute(1, 2, 3, 0).data.cpu().numpy())


class DemoSet(data_utl.Dataset):
    def __init__(self, root, transforms=None, mode="rgb"):
        self.root = root
        self.transforms = transforms
        self.mode = mode
        self.data = [
            ['changing_tire_0002_rgb', None, 154, 3696],  # 3696
            ['changing_tire_0003_rgb', None, 93, 2233]  # 2233
        ]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        vid, label, dur, nf = self.data[index]
        # if os.path.exists(os.path.join(self.save_dir, vid+'.npy')):
        #     return 0, 0, vid

        if self.mode == 'rgb':
            imgs = load_rgb_frames(self.root, vid, 1, nf)
        elif self.mode == 'flows':
            imgs = load_flow_frames(self.root, vid, 1, nf)
        else:
            imgs1 = load_rgb_frames(self.root, vid, 1, nf)
            imgs2 = load_flow_frames(self.root, vid, 1, nf)
            imgs = imgs1 + imgs2
        imgs = self.transforms(imgs)
        label = np.zeros((157, nf), np.float32)

        return image_to_tensor(imgs), torch.from_numpy(label), vid

    def __len__(self):
        return 2


def build_loader(data_dir):
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])
    dataset = DemoSet(data_dir, test_transforms)
    return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=False)


def main():
    i3d = build_model('models/rgb_charades.pt', )
    loader = build_loader('./data/flows')
    for data in loader:
        eval_model(i3d, data, 'output/two_stream_features')


if __name__ == '__main__':
    main()

