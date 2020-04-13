from collections import namedtuple
import time
import os

import numpy as np
import torch
import torchvision.models.vgg as vgg
from torchvision import transforms
from torch.optim import Adam
from torch.autograd import Variable
from PIL import Image
from tqdm import tqdm

from zyfastnst.model.transformer_net import TransformerNet
from zyfastnst.loader import nstloader
from zyfastnst.loss import FastNSTLoss
from zyfastnst.utils import gram_matrix, tensor_normalizer, save_debug_image

from PIL import ImageFile

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
ImageFile.LOAD_TRUNCATED_IMAGES = True


def train(style_img, train_dataset, train_loader, save_model_path, device):
    lossOutput = namedtuple("LossOutput", ["relu1_2", "relu2_2", "relu3_3", "relu4_3"])
    vgg_model = vgg.vgg16(pretrained=True)
    vgg_model = vgg_model.to(device)
    loss_network = FastNSTLoss(vgg_model, lossOutput)
    loss_network.eval()
    del vgg_model

    style_img = Image.open(style_img).convert('RGB')
    style_img_tensor = transforms.Compose([
        transforms.ToTensor(),
        tensor_normalizer()]
    )(style_img).unsqueeze(0)
    style_img_tensor = style_img_tensor.to(device)

    # http://pytorch.org/docs/master/notes/autograd.html#volatile
    style_loss_features = loss_network(Variable(style_img_tensor))
    gram_style = [Variable(gram_matrix(y).data, requires_grad=False) for y in style_loss_features]

    transformer = TransformerNet().to(device)
    mse_loss = torch.nn.MSELoss()

    CONTENT_WEIGHT = 1
    STYLE_WEIGHT = 1e5
    LOG_INTERVAL = 200
    REGULARIZATION = 1e-7

    LR = 1e-3
    optimizer = Adam(transformer.parameters(), LR)
    transformer.train()
    for epoch in range(2):
        agg_content_loss = 0.
        agg_style_loss = 0.
        agg_reg_loss = 0.
        count = 0
        for batch_id, (x, _) in tqdm(enumerate(train_loader), total=len(train_loader)):
            # for batch_id, (x, _) in train_loader:
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()
            x = Variable(x)
            if torch.cuda.is_available():
                x = x.cuda()

            y = transformer(x)
            xc = Variable(x.data)

            features_y = loss_network(y)
            features_xc = loss_network(xc)

            f_xc_c = Variable(features_xc[1].data, requires_grad=False)

            content_loss = CONTENT_WEIGHT * mse_loss(features_y[1], f_xc_c)

            reg_loss = REGULARIZATION * (
                    torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) +
                    torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])))

            style_loss = 0.
            for m in range(len(features_y)):
                gram_s = gram_style[m]
                gram_y = gram_matrix(features_y[m])
                style_loss += STYLE_WEIGHT * mse_loss(gram_y, gram_s.expand_as(gram_y))

            total_loss = content_loss + style_loss + reg_loss
            total_loss.backward()
            optimizer.step()

            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()
            agg_reg_loss += reg_loss.item()

            if (batch_id + 1) % LOG_INTERVAL == 0:
                mesg = "{} [{}/{}] content: {:.6f}  style: {:.6f}  reg: {:.6f}  total: {:.6f}".format(
                    time.ctime(), count, len(train_dataset),
                    agg_content_loss / LOG_INTERVAL,
                    agg_style_loss / LOG_INTERVAL,
                    agg_reg_loss / LOG_INTERVAL,
                    (agg_content_loss + agg_style_loss + agg_reg_loss) / LOG_INTERVAL
                )
                print(mesg)
                agg_content_loss = 0
                agg_style_loss = 0
                agg_reg_loss = 0
                transformer.eval()
                y = transformer(x)
                save_debug_image(x.data, y.data, "img/debug/{}_{}.png".format(epoch, count))
                transformer.train()


    torch.save(transformer.state_dict(), save_model_path)


if __name__ == '__main__':
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    SEED = 1080
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    IMAGE_SIZE = 256
    BATCH_SIZE = 4
    DATASET = "E:/Data/coco-test/"
    train_dataset, train_loader = nstloader(IMAGE_SIZE, DATASET, BATCH_SIZE, 4)

    STYLE_IMAGE = "img/style_images/starry-night-cropped.jpg"
    save_model_path = "model/model_udnie_starry2.pth"

    train(STYLE_IMAGE, train_dataset, train_loader, save_model_path, device)


