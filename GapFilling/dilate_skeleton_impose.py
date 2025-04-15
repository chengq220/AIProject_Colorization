import cv2
import numpy as np
from skimage.morphology import skeletonize

def dilate_skeleton_impose(image_path, thickness=9):
    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    invert = 255 - original
    
    # 2. Thicken the lines
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness, thickness))
    thickened = cv2.dilate(invert, kernel)
    binary = np.where(thickened < 210, 0, 255)

    #Skeletonize the image
    skeleton = skeletonize(binary // 255).astype(np.uint8) * 255
   
    # 4. Create overlay
    colored_skeleton = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
    colored_skeleton[np.where((colored_skeleton == [255, 255, 255]).all(axis=2))] = [0, 0, 255]  # Red
    
    original_color = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(original_color, 0.7, colored_skeleton, 0.3, 0)
    overlay = np.where(overlay < 150, 0, 255)
    cv2.imwrite('thickentest/overlay.png', overlay)  # Save overlay
    
    return skeleton, overlay

if __name__ == "__main__":
    skeleton, overlay = dilate_skeleton_impose("haruhi.png", thickness=5)
    print("Saved: skeleton.png, overlay.png")