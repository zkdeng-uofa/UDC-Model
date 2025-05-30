#!/usr/bin/env python3
"""
Example script demonstrating the usage of CustomImageProcessor.

This script shows how to:
1. Use the CustomImageProcessor with default model configurations
2. Create custom configurations for specific needs
3. Save and load configurations
4. Use the processor for different tasks (training, validation, testing)
"""

import sys
import os
from PIL import Image
import torch

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.image_processor import CustomImageProcessor, create_image_processor

def example_basic_usage():
    """Demonstrate basic usage with different model types."""
    print("=== Basic Usage Examples ===\n")
    
    # Example 1: ResNet configuration
    print("1. ResNet Image Processor:")
    resnet_processor = CustomImageProcessor.from_pretrained("resnet50")
    print(f"   Image mean: {resnet_processor.image_mean}")
    print(f"   Image std: {resnet_processor.image_std}")
    print(f"   Size: {resnet_processor.size}")
    print(f"   Crop percentage: {resnet_processor.crop_pct}")
    print(f"   Interpolation: {resnet_processor.interpolation}\n")
    
    # Example 2: Vision Transformer configuration
    print("2. Vision Transformer Image Processor:")
    vit_processor = CustomImageProcessor.from_pretrained("vit-base-patch16-224")
    print(f"   Image mean: {vit_processor.image_mean}")
    print(f"   Image std: {vit_processor.image_std}")
    print(f"   Size: {vit_processor.size}")
    print(f"   Crop percentage: {vit_processor.crop_pct}")
    print(f"   Interpolation: {vit_processor.interpolation}\n")
    
    # Example 3: ConvNeXt configuration
    print("3. ConvNeXt Image Processor:")
    convnext_processor = CustomImageProcessor.from_pretrained("convnext-base")
    print(f"   Image mean: {convnext_processor.image_mean}")
    print(f"   Image std: {convnext_processor.image_std}")
    print(f"   Size: {convnext_processor.size}")
    print(f"   Crop percentage: {convnext_processor.crop_pct}")
    print(f"   Interpolation: {convnext_processor.interpolation}\n")

def example_custom_configuration():
    """Demonstrate custom configuration."""
    print("=== Custom Configuration Example ===\n")
    
    # Create a custom configuration for high-resolution images
    custom_config = {
        'image_mean': [0.485, 0.456, 0.406],
        'image_std': [0.229, 0.224, 0.225],
        'size': {'height': 384, 'width': 384},
        'crop_pct': 0.95,
        'interpolation': 'bicubic'
    }
    
    custom_processor = CustomImageProcessor(custom_config=custom_config)
    print("Custom High-Resolution Image Processor:")
    print(f"   Image mean: {custom_processor.image_mean}")
    print(f"   Image std: {custom_processor.image_std}")
    print(f"   Size: {custom_processor.size}")
    print(f"   Crop percentage: {custom_processor.crop_pct}")
    print(f"   Interpolation: {custom_processor.interpolation}\n")

def example_save_load_config():
    """Demonstrate saving and loading configurations."""
    print("=== Save/Load Configuration Example ===\n")
    
    # Create a processor with custom settings
    processor = CustomImageProcessor(
        model_name="resnet",
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.25, 0.25, 0.25],
        size={'height': 256, 'width': 256},
        crop_pct=0.9,
        interpolation='bicubic'
    )
    
    # Save configuration
    config_path = "custom_processor_config.json"
    processor.save_config(config_path)
    print(f"Configuration saved to: {config_path}")
    
    # Load configuration
    loaded_processor = CustomImageProcessor.from_config(config_path)
    print("Loaded processor configuration:")
    print(f"   Image mean: {loaded_processor.image_mean}")
    print(f"   Image std: {loaded_processor.image_std}")
    print(f"   Size: {loaded_processor.size}")
    print(f"   Crop percentage: {loaded_processor.crop_pct}")
    print(f"   Interpolation: {loaded_processor.interpolation}\n")
    
    # Clean up
    if os.path.exists(config_path):
        os.remove(config_path)

def example_image_processing():
    """Demonstrate actual image processing."""
    print("=== Image Processing Example ===\n")
    
    # Create a dummy image for demonstration
    dummy_image = Image.new('RGB', (300, 300), color='red')
    
    # Create processor
    processor = CustomImageProcessor.from_pretrained("resnet50")
    
    # Process single image
    result = processor(dummy_image)
    print(f"Single image processing result shape: {result['pixel_values'].shape}")
    
    # Process multiple images
    images = [dummy_image, dummy_image.copy()]
    result = processor(images)
    print(f"Multiple images processing result shape: {result['pixel_values'].shape}")
    
    # Get different transforms
    train_transform = processor.get_transform_for_training(
        random_resize_crop=True,
        horizontal_flip=True,
        color_jitter=True
    )
    
    val_transform = processor.get_transform_for_validation()
    test_transform = processor.get_transform_for_test()
    
    print("Available transforms:")
    print(f"   Training transform: {len(train_transform.transforms)} steps")
    print(f"   Validation transform: {len(val_transform.transforms)} steps")
    print(f"   Test transform: {len(test_transform.transforms)} steps\n")

def example_convenience_function():
    """Demonstrate the convenience function."""
    print("=== Convenience Function Example ===\n")
    
    # Use convenience function
    processor = create_image_processor("efficientnet-b0")
    print("Created processor using convenience function:")
    print(f"   Model type detected: efficientnet")
    print(f"   Image mean: {processor.image_mean}")
    print(f"   Size: {processor.size}")
    print(f"   Interpolation: {processor.interpolation}\n")

def main():
    """Run all examples."""
    print("CustomImageProcessor Usage Examples")
    print("=" * 50)
    
    try:
        example_basic_usage()
        example_custom_configuration()
        example_save_load_config()
        example_image_processing()
        example_convenience_function()
        
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 