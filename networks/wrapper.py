from networks.FFNet import FFNet
from typing import List
import torch
from PIL import Image
from torchvision import transforms

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class FFNetWrapper:
    def __init__(self, checkpoint):
        self.model = FFNet()
        self.model.load_state_dict(torch.load(checkpoint, map_location=device))
        self.model.eval()

    def predict(self, images: List[str]):
        with torch.no_grad():
            transform = transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
            tensors = []
            for image_path in images:
                img = Image.open(image_path)
                tensor = transform(img).to(device)
                tensors.append(tensor)
            batch = torch.stack(tensors)
            return torch.sum(self.model(batch)[0])
