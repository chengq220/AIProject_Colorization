import numpy as np
from skimage import io, morphology
import matplotlib.pyplot as plt

def close_line_gaps_binary(image_path, output_path, kernel_size=3):
    # Read image (automatically converted to grayscale)
    image = io.imread(image_path, as_gray=True)
    
    # Apply binary closing
    binary = image < 0.5  # Adjust threshold if needed
    footprint = morphology.square(kernel_size)
    closed = morphology.binary_closing(binary, footprint=footprint)
    result = (~closed).astype(np.uint8) * 255
    
    io.imsave(output_path, result)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(binary, cmap='gray')
    plt.title('Original Binary')
    
    plt.subplot(1, 2, 2)
    plt.imshow(closed, cmap='gray')
    plt.title('After Closing')
    plt.show()

if __name__ == "__main__":
    close_line_gaps_binary("haruhi.png", "output/haruhi.png", thickness=5)