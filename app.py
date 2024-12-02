import streamlit as st
import numpy as np
import torch
from torchvision import transforms as T
from scipy import integrate
from io import BytesIO

from model import Unet


page_title = "😺고양이 이미지 생성기"
st.set_page_config(page_title, layout='centered')
st.title(page_title)
st.info("버튼을 클릭하면 고양이 이미지를 생성합니다. 이상한 고양이가 만들어 질 수도 있어요😂")

eps = 1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dim = 128
dim_mults = [1, 2, 4, 8]
channels = 3
img_size = 128


# Rectified Flow sampling
@torch.no_grad()
def ode_sampling(model, image_size, batch_size=16, channels=1):
    shape = (batch_size, channels, image_size, image_size)
    device = next(model.parameters()).device

    b = shape[0]
    x = torch.randn(shape, device=device)
    
    def ode_func(t, x):
        x = torch.tensor(x, device=device, dtype=torch.float).reshape(shape)
        t = torch.full(size=(b,), fill_value=t, device=device, dtype=torch.float).reshape((b,))
        v = model(x, t)
        return v.cpu().numpy().reshape((-1,)).astype(np.float64)
    
    res = integrate.solve_ivp(ode_func, (eps, 1.), x.reshape((-1,)).cpu().numpy(), method='RK45')
    x = torch.tensor(res.y[:, -1], device=device).reshape(shape)
    return x.clamp(-1, 1)


@st.cache_resource()
def load_model(dim, dim_mults, channels):
    saved_model_path = "./results/huggan/AFHQv2/ckpt/size_128_500ep.pth"
    model = Unet(dim=dim, dim_mults=dim_mults, channels=channels).to(device)
    loaded_state_dict = torch.load(saved_model_path, weights_only=True)
    model.load_state_dict(loaded_state_dict['model'])
    model.eval()
    return model


def norm_range(t):
    low = float(t.min())
    high = float(t.max())
    t.clamp_(min=low, max=high)
    t.sub_(low).div_(max(high - low, 1e-5))
    t = t.squeeze(0)
    return t


def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im


model = load_model(dim, dim_mults, channels)

if st.button("야옹~"):
    with st.spinner("이미지 생성 중..."):
        generated_tensor = ode_sampling(model, img_size, batch_size=1, channels=channels)
        norm_tensor = norm_range(generated_tensor)
        img = T.ToPILImage()(norm_tensor)
        st.success("이미지 생성 완료!")
    st.image(img)
    downloadable_image = convert_image(img)
    st.download_button("다운로드", downloadable_image, "generated_image.png", "image.png")

