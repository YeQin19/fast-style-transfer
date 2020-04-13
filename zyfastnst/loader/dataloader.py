from torchvision import transforms
from torch.utils.data import DataLoader
from zyfastnst.utils import tensor_normalizer
from torchvision import datasets


def nstloader(image_size, dataset, bacth_size, num_workers):
    transform = transforms.Compose([transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size),
                                    transforms.ToTensor(),
                                    tensor_normalizer()])
    # http://pytorch.org/docs/master/torchvision/datasets.html#imagefolder
    dataset = datasets.ImageFolder(dataset, transform)
    # http://pytorch.org/docs/master/data.html#torch.utils.data.DataLoader
    loader = DataLoader(dataset, batch_size=bacth_size, shuffle=True, num_workers=num_workers)

    return dataset, loader