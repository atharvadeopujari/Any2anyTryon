#!/bin/bash

# Create directories
mkdir -p weights
mkdir -p inputs
mkdir -p results

# Install compatible torchvision version
echo "Installing compatible torchvision version..."
pip install torchvision==0.12.0

# Download RealESRGAN repository
echo "Cloning RealESRGAN repository if not exists..."
if [ ! -d "Real-ESRGAN" ]; then
    git clone https://github.com/xinntao/Real-ESRGAN.git
    cd Real-ESRGAN
    pip install -r requirements.txt
    cd ..
    # Copy the inference script to the main directory
    cp Real-ESRGAN/inference_realesrgan.py .
else
    echo "RealESRGAN repository already exists."
fi

# Download pre-trained models
echo "Downloading pre-trained models..."
if [ ! -f "weights/RealESRGAN_x4plus.pth" ]; then
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P weights
else
    echo "RealESRGAN_x4plus.pth already exists."
fi

# Download GFPGAN for face enhancement
if [ ! -f "weights/GFPGANv1.3.pth" ]; then
    wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth -P weights
else
    echo "GFPGANv1.3.pth already exists."
fi

# Install required dependencies for RealESRGAN
pip install basicsr facexlib gfpgan

# Create a patch to fix the functional_tensor import issue
echo "Creating patch for torchvision compatibility..."
cat > patch_torchvision.py << EOL
import os
import re

# Path to the degradations.py file
file_path = os.path.join('myenv', 'lib', 'python3.10', 'site-packages', 'basicsr', 'data', 'degradations.py')
alt_path = os.path.join('Real-ESRGAN', 'basicsr', 'data', 'degradations.py')

# Check which path exists
if os.path.exists(file_path):
    path_to_patch = file_path
elif os.path.exists(alt_path):
    path_to_patch = alt_path
else:
    # Search for the file in site-packages
    import site
    for site_path in site.getsitepackages():
        potential_path = os.path.join(site_path, 'basicsr', 'data', 'degradations.py')
        if os.path.exists(potential_path):
            path_to_patch = potential_path
            break
    else:
        print("Could not find degradations.py")
        exit(1)

print(f"Patching file: {path_to_patch}")

# Read the file
with open(path_to_patch, 'r') as file:
    content = file.read()

# Replace the import
updated_content = re.sub(
    r'from torchvision.transforms.functional_tensor import rgb_to_grayscale',
    'from torchvision.transforms.functional import rgb_to_grayscale',
    content
)

# Write the updated content back to the file
with open(path_to_patch, 'w') as file:
    file.write(updated_content)

print("Patch applied successfully!")
EOL

# Run the patch script
python patch_torchvision.py

echo "Setup complete! The RealESRGAN model is ready to use."
echo "Now you can run the application with: python app.py" 