import random
import pathlib
from PIL import Image
#from torch.utils import data
# from torch.utils.data import Dataset
#from torchvision import transforms
import jittor as jt
from jittor import dataset
from jittor import transform


IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}
class ImagePathDataset(dataset.Dataset):
    def __init__(self,drop_last, batch_size, path, image_mode='L', transform=None, max_images=None):
        super.__init__(batch_size=batch_size,shuffle=True,drop_last=drop_last)
        path = pathlib.Path(path)
        files = sorted([file for ext in IMAGE_EXTENSIONS
                       for file in path.glob('*.{}'.format(ext))])
        if max_images is None:
            self.files = files
        elif max_images < len(files):
            self.files = random.sample(files, max_images)
        else:
            print(f"max_images larger or equal to total number of files, use {len(files)} images instead.")
            self.files = files
        self.transform = transform
        self.image_mode = image_mode

    def __getitem__(self, index):
        image_path = self.files[index]
        image = Image.open(image_path).convert(self.image_mode)
        if self.transform is not None:
            image = self.transform(image)
        return index, image

    def __len__(self):
        return len(self.files)


def data_sampler(dataset, shuffle):
    if shuffle:
        return jt.dataset.RandomSampler(dataset)
    else:
        return jt.dataset.SequentialSampler(dataset)


def create_dataloader(data_dir, size, batch, img_channel=3):
    mean, std = [0.5 for _ in range(img_channel)], [0.5 for _ in range(img_channel)]
    transform = jt.transform.Compose([
            jt.transform.Resize(size),
            jt.transform.ToTensor(),
            jt.transform.ImageNormalize(mean, std),
        ])

    if img_channel == 1:
        image_mode = 'L'
    elif img_channel == 3:
        image_mode = 'RGB'
    else:
        raise ValueError("image channel should be 1 or 3, but got ", img_channel)

    dataset = ImagePathDataset(True, batch, data_dir, image_mode, transform)

    sampler = data_sampler(dataset, shuffle=True)
    #loader = jt.dataset.DataLoader(dataset, batch_size=batch, sampler=sampler, drop_last=True)
    loader = dataset
    return loader, sampler


def yield_data(loader, sampler, distributed=False):
    epoch = 0
    while True:
        # if distributed:
        #     sampler.set_epoch(epoch)
        for i, batch in loader:
            yield batch
        epoch += 1
