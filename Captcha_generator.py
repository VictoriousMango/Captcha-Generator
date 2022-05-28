import torch
import torch.nn as nn
from torchvision.utils import save_image, make_grid
import torchvision.transforms as transforms
import random
from PIL import Image
import streamlit as st
from io import BytesIO
buf = BytesIO()


st.title("Captcha Generator")
st.header("Captcha Generation Using DCGAN")
st.write("Dataset Available At: https://www.kaggle.com/datasets/fanbyprinciple/captcha-images")
st.write("In GANs, there is a generator and a discriminator. The Generator generates fake samples of data\n"
         "and tries to fool the Discriminator. The Discriminator, on the other hand, tries to distinguish between the\n"
         "real and fake samples. The Generator and the Discriminator are both Neural Networks and they both run in\n"
         "competition with each other in the training phase. The steps are repeated several times and in this,\n"
         "the Generator and Discriminator get better and better in their respective jobs after each repetition.")


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

st.markdown("All Captcha are **_4_ Characters Long.**")
make_single = st.button("Generate Captcha")
make_multiple = st.checkbox("Make Multiple Captcha")

if make_single:
    random_img = torch.randn((batch_size, noise_dim, 1, 1)).to(device)
    print(random_img.shape)
    num = random.randint(0, 63)
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
        grid.save(buf, format="JPEG")
        byte_im = buf.getvalue()
    else:
        st.image(img)
        img.save(buf, format="JPEG")
    btn = st.download_button(
        label="Download Image",
        data=byte_im,
        file_name="Generated Captcha.png",
        mime="image/jpeg",
    )
