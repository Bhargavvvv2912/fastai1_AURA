import sys
import os

def test_fastai_v1_legacy():
    try:
        # THE TRIPWIRE
        # Modern torch (2.x) and scikit-image (0.2x+) 
        # often break legacy fastai1's data loaders and vision helpers.
        from fastai.vision import untar_data, URLs, ImageDataBunch, get_transforms, cnn_learner, models
        import torch
        import skimage

        # Download a tiny dataset (MNIST sample)
        path = untar_data(URLs.MNIST_SAMPLE)
        
        # This step often fails if torch/torchvision versions are mismatched 
        # or if internal PIL/skimage handling has shifted.
        data = ImageDataBunch.from_folder(path, ds_tfms=get_transforms(), size=24)
        
        # Initialize a basic learner
        learn = cnn_learner(data, models.resnet18)
        
        print(f"✅ Validation Passed: FastAI v1 functional with torch {torch.__version__} "
              f"and scikit-image {skimage.__version__}")
        return True

    except Exception as e:
        print(f"❌ Validation Failed: {type(e).__name__}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_fastai_v1_legacy()