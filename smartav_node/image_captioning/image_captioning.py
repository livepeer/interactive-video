import base64
import io
import torch
from datetime import datetime
from PIL import Image
from image_captioning.models.blip import blip_decoder
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def transform_raw_image(raw_image, image_size=384):
    encoded_data = raw_image.split(',')[1]
    decoded_string = io.BytesIO(base64.b64decode(encoded_data))
    img = Image.open(decoded_string).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    image = transform(img).unsqueeze(0).to(device)
    return image


def load_model():
    image_size = 384
    model = blip_decoder(pretrained='image_captioning/checkpoints/model_base.pth', image_size=image_size, vit='base')
    model.eval()
    model = model.to(device)

    return model


def process_image(model, raw_image):
    # Transform the raw_image
    image = transform_raw_image(raw_image)

    with torch.no_grad():
        # beam search
        caption = model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5)
        print('caption: ' + caption[0])

    return caption[0]
