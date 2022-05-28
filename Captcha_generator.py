import torch
import torch.nn as nn
from torchvision.utils import save_image, make_grid
import torchvision.transforms as transforms
import random
from PIL import Image
import streamlit as st


class Generator(nn.Module):
    def __init__(self, z_dim=64, noise_dim=100, channels_img=3):
        super(Generator, self).__init__()
        self.network = nn.Sequential(
            nn.ConvTranspose2d(in_channels=noise_dim, out_channels=z_dim * 16, kernel_size=(4, 4), stride=(1, 1),
                               padding=(0, 0)),
            nn.BatchNorm2d(z_dim * 16),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=z_dim * 16, out_channels=z_dim * 8, kernel_size=(4, 4), stride=(2, 2),
                               padding=(1, 1)),
            nn.BatchNorm2d(z_dim * 8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=z_dim * 8, out_channels=z_dim * 4, kernel_size=(4, 4), stride=(2, 2),
                               padding=(1, 1)),
            nn.BatchNorm2d(z_dim * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=z_dim * 4, out_channels=z_dim * 2, kernel_size=(4, 4), stride=(2, 2),
                               padding=(1, 1)),
            nn.BatchNorm2d(z_dim * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=z_dim * 2, out_channels=channels_img, kernel_size=(4, 4), stride=(2, 2),
                               padding=(1, 1)),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.network(x)
        # print("Gen: ", x.shape)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Generator()
batch_size = 64
noise_dim = 100
checkpoint = torch.load("C:/Users/harsh/PycharmProjects/Harshvir_S/HackGDSC/Generator4.pth.tar",
                        map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
transform1 = transforms.Compose([transforms.Resize([72, 216]), transforms.ToPILImage()])
transform2 = transforms.Compose([transforms.Resize([640, 800]), transforms.ToPILImage()])

make_single = st.button("Generate Captcha")
make_multiple = st.checkbox("Make Multiple Captcha")
# C:/Users/harsh/PycharmProjects/Harshvir_S/HackGDSC/Captcha_generator.py
if make_single:
    random_img = torch.randn((batch_size, noise_dim, 1, 1)).to(device)
    num = random.randint(0, 64)
    img = model(random_img)
    # print(img.shape)
    grid = make_grid(img)
    # print(grid.shape)
    # save_image(img[num], "Captcha.jpg")
    # print(img[num].shape)
    img = transform1(img[num])
    grid = transform2(grid)
    if make_multiple:
        st.image(grid)
    else:
        st.image(img)

# print(img.shape)
# cv2.imshow("captcha", img)
# cv2.waitKey(0)
