import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

from zyfastnst.model.transformer_net import TransformerNet
from zyfastnst.utils import recover_image, tensor_normalizer

import torch
import torchvision


if __name__ == '__main__':
    # 加载预训练的模型
    save_model_path = 'model/model_udnie_picasso.pth'

    transformer = TransformerNet()
    transformer = transformer.eval()
    transformer.load_state_dict(torch.load(save_model_path))

    # 加载输入图像
    img = Image.open("img/content_images/00285.jpg").convert('RGB')
    transform = transforms.Compose([transforms.ToTensor(),
                                    tensor_normalizer()])
    img_tensor = transform(img).unsqueeze(0)

    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda()
        transformer = transformer.cuda()

    # 保存图片的路径
    save_path = 'save.png'
    try:
        img_output = transformer(img_tensor)
        torchvision.utils.save_image(img_output, save_path)
        plt.figure()
        plt.imshow(recover_image(img_output.detach().cpu().numpy())[0])
        plt.pause(0.001)
    except:
        print("CUDA out of memory! Please change the image!")