import torch
import torchvision.transforms as transforms
from PIL import Image
from typing import Dict, Union, List, Tuple, Optional
import json
import os

class CustomImageProcessor:
    """
    Custom Image Processor that replaces AutoImageProcessor with configurable parameters
    for different models while maintaining the same interface and functionality.
    """
    
    # Default configurations for common models
    MODEL_CONFIGS = {
        'resnet': {
            'image_mean': [0.485, 0.456, 0.406],
            'image_std': [0.229, 0.224, 0.225],
            'size': {'height': 224, 'width': 224},
            'crop_pct': 0.875,
            'interpolation': 'bilinear'
        },
        'vit': {
            'image_mean': [0.5, 0.5, 0.5],
            'image_std': [0.5, 0.5, 0.5],
            'size': {'height': 224, 'width': 224},
            'crop_pct': 0.9,
            'interpolation': 'bicubic'
        },
        'convnext': {
            'image_mean': [0.485, 0.456, 0.406],
            'image_std': [0.229, 0.224, 0.225],
            'size': {'shortest_edge': 224},
            'crop_pct': 0.875,
            'interpolation': 'bicubic'
        },
        'efficientnet': {
            'image_mean': [0.485, 0.456, 0.406],
            'image_std': [0.229, 0.224, 0.225],
            'size': {'shortest_edge': 224},
            'crop_pct': 1.0,
            'interpolation': 'bicubic'
        },
        'swin': {
            'image_mean': [0.485, 0.456, 0.406],
            'image_std': [0.229, 0.224, 0.225],
            'size': {'height': 224, 'width': 224},
            'crop_pct': 0.9,
            'interpolation': 'bicubic'
        }
    }
    
    def __init__(self, 
                 model_name: str = 'resnet',
                 image_mean: Optional[List[float]] = None,
                 image_std: Optional[List[float]] = None,
                 size: Optional[Dict[str, int]] = None,
                 crop_pct: float = 0.875,
                 interpolation: str = 'bilinear',
                 custom_config: Optional[Dict] = None):
        """
        Initialize the Custom Image Processor.
        
        Args:
            model_name: Name of the model type ('resnet', 'vit', 'convnext', etc.)
            image_mean: Custom mean values for normalization
            image_std: Custom std values for normalization
            size: Custom size configuration
            crop_pct: Crop percentage for center crop
            interpolation: Interpolation method ('bilinear', 'bicubic')
            custom_config: Complete custom configuration dictionary
        """
        # Use custom config if provided, otherwise use model-specific config
        if custom_config:
            config = custom_config
        else:
            # Extract model type from model name for common architectures
            model_type = self._extract_model_type(model_name)
            config = self.MODEL_CONFIGS.get(model_type, self.MODEL_CONFIGS['resnet'])
        
        # Override with provided parameters
        self.image_mean = image_mean if image_mean is not None else config['image_mean']
        self.image_std = image_std if image_std is not None else config['image_std']
        self.size = size if size is not None else config['size']
        self.crop_pct = crop_pct if crop_pct != 0.875 else config.get('crop_pct', 0.875)
        self.interpolation = interpolation if interpolation != 'bilinear' else config.get('interpolation', 'bilinear')
        
        # Set interpolation mode
        if self.interpolation == 'bicubic':
            self.interp_mode = transforms.InterpolationMode.BICUBIC
        else:
            self.interp_mode = transforms.InterpolationMode.BILINEAR
        
        # Calculate crop size
        self.crop_size = self._calculate_crop_size()
        
        # Create basic transforms
        self._create_transforms()
    
    def _extract_model_type(self, model_name: str) -> str:
        """Extract model type from model name string."""
        model_name_lower = model_name.lower()
        
        if 'resnet' in model_name_lower:
            return 'resnet'
        elif 'vit' in model_name_lower or 'vision' in model_name_lower:
            return 'vit'
        elif 'convnext' in model_name_lower:
            return 'convnext'
        elif 'efficientnet' in model_name_lower:
            return 'efficientnet'
        elif 'swin' in model_name_lower:
            return 'swin'
        else:
            # Default to resnet config for unknown models
            return 'resnet'
    
    def _calculate_crop_size(self) -> Union[int, Tuple[int, int]]:
        """Calculate crop size based on size configuration."""
        if "height" in self.size and "width" in self.size:
            return (self.size["height"], self.size["width"])
        elif "shortest_edge" in self.size:
            return self.size["shortest_edge"]
        else:
            # Default fallback
            return 224
    
    def _create_transforms(self):
        """Create the transformation pipelines."""
        # Normalization transform
        self.normalize = transforms.Normalize(mean=self.image_mean, std=self.image_std)
        
        # Calculate resize size based on crop percentage
        if isinstance(self.crop_size, tuple):
            resize_size = (
                int(self.crop_size[0] / self.crop_pct),
                int(self.crop_size[1] / self.crop_pct)
            )
        else:
            resize_size = int(self.crop_size / self.crop_pct)
        
        # Basic inference transform (equivalent to AutoImageProcessor)
        self.inference_transform = transforms.Compose([
            transforms.Resize(resize_size, interpolation=self.interp_mode),
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),
            self.normalize
        ])
    
    def __call__(self, images: Union[Image.Image, List[Image.Image]], 
                 return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        """
        Process images and return in the same format as AutoImageProcessor.
        
        Args:
            images: Single PIL Image or list of PIL Images
            return_tensors: Format to return tensors ("pt" for PyTorch)
            
        Returns:
            Dictionary with 'pixel_values' key containing processed tensors
        """
        if isinstance(images, Image.Image):
            images = [images]
        
        # Convert images to RGB and process
        processed_images = []
        for img in images:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            processed_img = self.inference_transform(img)
            processed_images.append(processed_img)
        
        # Stack tensors
        pixel_values = torch.stack(processed_images)
        
        return {"pixel_values": pixel_values}
    
    def preprocess(self, images: Union[Image.Image, List[Image.Image]], 
                   return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        """Alias for __call__ to match AutoImageProcessor interface."""
        return self.__call__(images, return_tensors)
    
    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs):
        """
        Create ImageProcessor from model name or path (similar to AutoImageProcessor.from_pretrained).
        
        Args:
            model_name_or_path: Name of the model or path to saved configuration
            **kwargs: Additional arguments to pass to the constructor
            
        Returns:
            CustomImageProcessor instance configured for the model
        """
        # Remove 'use_fast' if present (compatibility with AutoImageProcessor calls)
        kwargs.pop('use_fast', None)
        
        # Check if it's a path to a saved configuration
        if os.path.isdir(model_name_or_path):
            config_file = os.path.join(model_name_or_path, "preprocessor_config.json")
            if os.path.exists(config_file):
                return cls.from_config(config_file)
        
        # Otherwise, treat as model name
        return cls(model_name=model_name_or_path, **kwargs)
    
    def save_config(self, file_path: str):
        """Save current configuration to a JSON file."""
        config = {
            'image_mean': self.image_mean,
            'image_std': self.image_std,
            'size': self.size,
            'crop_pct': self.crop_pct,
            'interpolation': self.interpolation
        }
        
        with open(file_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def from_config(cls, file_path: str):
        """Load configuration from a JSON file."""
        with open(file_path, 'r') as f:
            config = json.load(f)
        
        return cls(custom_config=config)
    
    def save_pretrained(self, save_directory: str):
        """
        Save the processor configuration to a directory (compatible with Transformers).
        
        Args:
            save_directory: Directory to save the configuration
        """
        os.makedirs(save_directory, exist_ok=True)
        
        config_file = os.path.join(save_directory, "preprocessor_config.json")
        self.save_config(config_file)
        
        print(f"CustomImageProcessor configuration saved to {config_file}")
    
    @property
    def model_input_names(self):
        """Return the expected input names (compatibility with Transformers)."""
        return ["pixel_values"]
    
    def __repr__(self):
        """String representation of the processor."""
        return (f"CustomImageProcessor("
                f"image_mean={self.image_mean}, "
                f"image_std={self.image_std}, "
                f"size={self.size}, "
                f"crop_pct={self.crop_pct}, "
                f"interpolation='{self.interpolation}')")
    
    def to_dict(self):
        """Convert processor configuration to dictionary."""
        return {
            'image_mean': self.image_mean,
            'image_std': self.image_std,
            'size': self.size,
            'crop_pct': self.crop_pct,
            'interpolation': self.interpolation
        }
    
    def get_transform_for_training(self, 
                                   random_resize_crop: bool = True,
                                   horizontal_flip: bool = True,
                                   color_jitter: bool = False) -> transforms.Compose:
        """
        Get training transform with data augmentation.
        
        Args:
            random_resize_crop: Whether to use random resize crop
            horizontal_flip: Whether to use random horizontal flip
            color_jitter: Whether to use color jitter
            
        Returns:
            Training transform pipeline
        """
        transform_list = []
        
        if random_resize_crop:
            if isinstance(self.crop_size, tuple):
                transform_list.append(transforms.RandomResizedCrop(
                    self.crop_size, interpolation=self.interp_mode))
            else:
                transform_list.append(transforms.RandomResizedCrop(
                    (self.crop_size, self.crop_size), interpolation=self.interp_mode))
        else:
            # Use resize + center crop for training if random crop is disabled
            resize_size = int(self.crop_size / self.crop_pct) if isinstance(self.crop_size, int) else (
                int(self.crop_size[0] / self.crop_pct), int(self.crop_size[1] / self.crop_pct))
            transform_list.extend([
                transforms.Resize(resize_size, interpolation=self.interp_mode),
                transforms.CenterCrop(self.crop_size)
            ])
        
        if horizontal_flip:
            transform_list.append(transforms.RandomHorizontalFlip())
        
        if color_jitter:
            transform_list.append(transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05))
        
        transform_list.extend([
            transforms.ToTensor(),
            self.normalize
        ])
        
        return transforms.Compose(transform_list)
    
    def get_transform_for_validation(self) -> transforms.Compose:
        """Get validation transform (same as inference)."""
        return self.inference_transform
    
    def get_transform_for_test(self) -> transforms.Compose:
        """Get test transform (same as inference)."""
        return self.inference_transform

# Convenience function to create processor
def create_image_processor(model_name: str, **kwargs) -> CustomImageProcessor:
    """
    Convenience function to create a CustomImageProcessor.
    
    Args:
        model_name: Name of the model
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured CustomImageProcessor instance
    """
    return CustomImageProcessor.from_pretrained(model_name, **kwargs) 