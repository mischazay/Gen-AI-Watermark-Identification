import os
from PIL import Image, ImageFilter, ImageEnhance
import io
import numpy as np

def attack_jpeg(image, quality):
    """
    Applies JPEG compression to a PIL image.
    
    Args:
        image (PIL.Image.Image): The input image.
        quality (int): The JPEG quality factor (1-100).
        
    Returns:
        PIL.Image.Image: The attacked image.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")

def attack_gaussian_noise(image, std):
    """
    Adds Gaussian noise to a PIL image.
    
    Args:
        image (PIL.Image.Image): The input image.
        std (float): The standard deviation of the Gaussian noise.
        
    Returns:
        PIL.Image.Image: The attacked image.
    """
    
    img_array = np.array(image).astype(np.float32) / 255.0
    noise = np.random.normal(0, std, img_array.shape).astype(np.float32)
    noisy_img_array = np.clip(img_array + noise, 0, 1) * 255
    return Image.fromarray(noisy_img_array.astype(np.uint8))

def attack_blur(image, sigma):
    """
    Applies Gaussian blur to a PIL image.
    
    Args:
        image (PIL.Image.Image): The input image.
        sigma (float): The sigma for the Gaussian blur.
        
    Returns:
        PIL.Image.Image: The attacked image.
    """
    return image.filter(ImageFilter.GaussianBlur(sigma))

def attack_brightness(image, factor):
    """
    Adjusts the brightness of a PIL image.
    
    Args:
        image (PIL.Image.Image): The input image.
        factor (float): The brightness enhancement factor. 
                        1.0 gives the original image.
                        
    Returns:
        PIL.Image.Image: The attacked image.
    """
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def attack_crop(image, ratio):
    """
    Performs a center crop on a PIL image.
    
    Args:
        image (PIL.Image.Image): The input image.
        ratio (float): The ratio of the original size to keep.
        
    Returns:
        PIL.Image.Image: The attacked and resized image.
    """
    w, h = image.size
    new_w, new_h = int(w * ratio), int(h * ratio)
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    right = (w + new_w) // 2
    bottom = (h + new_h) // 2
    
    cropped_image = image.crop((left, top, right, bottom))
    return cropped_image.resize((w, h))

def get_attack_func(attack_name):
    """
    Returns the corresponding attack function for a given attack name.
    """
    attack_map = {
        'jpeg': attack_jpeg,
        'gaussian_noise': attack_gaussian_noise,
        'gaussian_blur': attack_blur,
        'brightness': attack_brightness,
        'crop': attack_crop
    }
    return attack_map.get(attack_name)