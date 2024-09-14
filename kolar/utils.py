import numpy as np
import torch
import torchvision.transforms as T

from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import requests
from io import BytesIO


class VisionaryUtils:

    def __init__(self, input_size=448, max_num=12) -> None:
        self.input_size = input_size
        self.max_num = max_num
        self.transform = self.__build_transform(self.input_size)

    def __build_transform(self, input_size):
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
        return transform
    
    def __dynamic_preprocess(self, image, min_num=1, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, self.max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= self.max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.__find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, self.input_size)

        # calculate the target width and height
        target_width = self.input_size * target_aspect_ratio[0]
        target_height = self.input_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // self.input_size)) * self.input_size,
                (i // (target_width // self.input_size)) * self.input_size,
                ((i % (target_width // self.input_size)) + 1) * self.input_size,
                ((i // (target_width // self.input_size)) + 1) * self.input_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((self.input_size, self.input_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def __find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio
   
    def __download_image(self, image_link: str) -> Image:
        try:
            # Send a GET request to the URL
            response = requests.get(image_link)
            response.raise_for_status()  # Check if the request was successful
            
            # Open the image from the response content
            image = Image.open(BytesIO(response.content)).convert('RGB')
            return image
        
        except requests.RequestException as e:
            print(f"Error fetching the image: {e}")
        except IOError as e:
            print(f"Error processing the image: {e}")

        return Image.new('RGB', (100, 100), color='black')

    def load_image(self, image_link: str) -> torch.Tensor:
        image = self.__download_image(image_link)
        images = self.__dynamic_preprocess(image, use_thumbnail=True)
        pixel_values = [self.transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values