import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from scipy.spatial.distance import directed_hausdorff, cdist
import matplotlib.pyplot as plt
import cv2
import warnings
from skimage.metrics import structural_similarity

"""
Performance metrics for our pipeline
"""
def load_image(img_path):
    """Load an image."""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")

    # Convert from BGR to RGB for better visualization
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def extract_lines_from_colored(img, line_thickness=1, adapt_method="color_gradient"):
    """
    Extract line art from colored images using various methods.

    Parameters:
    - img: Input colored image
    - line_thickness: Estimated line thickness to enhance
    - adapt_method: Method to use for line extraction
                    "color_gradient" - Uses color gradients
                    "canny" - Uses Canny edge detection with preprocessing
                    "adaptive" - Uses adaptive thresholding
    """
    # Create a copy to avoid modifying the original
    result = img.copy()

    if adapt_method == "color_gradient":
        # Convert to grayscale for initial processing
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Apply bilateral filter to smooth colors while preserving edges
        smooth = cv2.bilateralFilter(img, 9, 75, 75)

        # Compute gradients in color channels
        grad_channels = []
        for i in range(3):  # RGB channels
            sobelx = cv2.Sobel(smooth[:,:,i], cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(smooth[:,:,i], cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(sobelx**2 + sobely**2)
            grad_channels.append(magnitude)

        # Combine gradients from all channels
        gradient = np.maximum.reduce(grad_channels)

        # Normalize to 0-255 range
        gradient = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Threshold to get the lines
        _, binary = cv2.threshold(gradient, 40, 255, cv2.THRESH_BINARY)

        # Remove small noise with morphological operations
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Thin lines using morphological operations
        kernel = np.ones((line_thickness,line_thickness), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Dilate slightly to ensure line continuity
        kernel = np.ones((line_thickness,line_thickness), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=1)

        return binary

    elif adapt_method == "canny":
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Apply bilateral filter to smooth while preserving edges
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)

        # Apply Canny edge detection with automatic threshold
        median_val = np.median(blurred)
        lower = int(max(0, (1.0 - 0.33) * median_val))
        upper = int(min(255, (1.0 + 0.33) * median_val))
        edges = cv2.Canny(blurred, lower, upper)

        # Dilate to ensure line continuity
        kernel = np.ones((line_thickness,line_thickness), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        return edges

    elif adapt_method == "adaptive":
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Apply bilateral filter
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)

        # Invert for processing (assuming dark lines)
        inverted = 255 - blurred

        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(inverted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)

        # Remove small noise
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        return binary

    else:
        raise ValueError(f"Unknown adaptation method: {adapt_method}")

def skeletonize(img):
    """Custom skeletonization function using morphological operations."""
    # Ensure the image is binary
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    # Ensure binary image (0 and 255)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Create a skeleton using morphological operations
    skeleton = np.zeros(binary.shape, np.uint8)
    eroded = np.zeros(binary.shape, np.uint8)
    temp = np.zeros(binary.shape, np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        # Erode the image
        cv2.erode(binary, kernel, eroded)

        # Open the eroded image
        cv2.dilate(eroded, kernel, temp)

        # Get the difference between the eroded and opened image
        temp = cv2.subtract(binary, temp)

        # Add the difference to the skeleton
        skeleton = cv2.bitwise_or(skeleton, temp)

        # Set the eroded image as the new image
        binary = eroded.copy()

        # If the eroded image is zero, we're done
        if cv2.countNonZero(binary) == 0:
            break

    return skeleton

def compute_chamfer_distance(coords1, coords2, sample_size=5000):
    """Compute Chamfer distance between two point sets."""
    # Sample points if there are too many
    if len(coords1) > sample_size:
        idx1 = np.random.choice(len(coords1), sample_size, replace=False)
        coords1 = coords1[idx1]

    if len(coords2) > sample_size:
        idx2 = np.random.choice(len(coords2), sample_size, replace=False)
        coords2 = coords2[idx2]

    # Compute pairwise distances
    dist_matrix = cdist(coords1, coords2, 'euclidean')

    # Compute Chamfer distance
    chamfer1 = np.mean(np.min(dist_matrix, axis=1))
    chamfer2 = np.mean(np.min(dist_matrix, axis=0))

    return (chamfer1 + chamfer2) / 2

def compute_edge_metrics(edges1, edges2):
    """Compute metrics based on edges."""
    # Get edge coordinates
    coords1 = np.argwhere(edges1 > 0)
    coords2 = np.argwhere(edges2 > 0)

    metrics = {}

    # Basic edge metrics
    metrics['edge_pixels_gen'] = len(coords1)
    metrics['edge_pixels_gt'] = len(coords2)

    # Precision, Recall, F1 score
    if len(coords1) > 0 and len(coords2) > 0:
        # Create edge masks with dilation to account for slight misalignments
        kernel = np.ones((3, 3), np.uint8)
        dilated_edges2 = cv2.dilate(edges2, kernel, iterations=1)
        dilated_edges1 = cv2.dilate(edges1, kernel, iterations=1)

        # Calculate True Positives (edges in generated that are also in ground truth)
        true_positives = np.sum(np.logical_and(edges1 > 0, dilated_edges2 > 0))

        # Precision: How many detected edges are correct
        precision = true_positives / np.sum(edges1 > 0) if np.sum(edges1 > 0) > 0 else 0

        # Recall: How many actual edges were detected
        recall = true_positives / np.sum(edges2 > 0) if np.sum(edges2 > 0) > 0 else 0

        # F1 score
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1_score'] = f1

        # Compute Hausdorff distance
        try:
            forward = directed_hausdorff(coords1, coords2)[0]
            backward = directed_hausdorff(coords2, coords1)[0]
            metrics['hausdorff'] = max(forward, backward)
        except:
            metrics['hausdorff'] = float('inf')

        # Compute Chamfer distance
        try:
            metrics['chamfer'] = compute_chamfer_distance(coords1, coords2)
        except:
            metrics['chamfer'] = float('inf')
    else:
        metrics['precision'] = 0
        metrics['recall'] = 0
        metrics['f1_score'] = 0
        metrics['hausdorff'] = float('inf')
        metrics['chamfer'] = float('inf')

    return metrics

def compute_connectivity_metrics(bin_img1, bin_img2):
    """Compute metrics related to line connectivity and continuity."""
    # Create skeletons
    skeleton1 = skeletonize(bin_img1)
    skeleton2 = skeletonize(bin_img2)

    # Count endpoints (places where a line ends) - pixels with only one neighbor
    kernel = np.ones((3, 3), np.uint8)
    kernel[1, 1] = 0  # Remove center

    # Dilated skeletons minus original skeletons give a rough count of endpoints
    dilated1 = cv2.dilate(skeleton1, kernel, iterations=1)
    dilated2 = cv2.dilate(skeleton2, kernel, iterations=1)

    endpoints1 = cv2.bitwise_and(dilated1, cv2.bitwise_not(skeleton1))
    endpoints2 = cv2.bitwise_and(dilated2, cv2.bitwise_not(skeleton2))

    endpoint_count1 = np.sum(endpoints1 > 0)
    endpoint_count2 = np.sum(endpoints2 > 0)

    # Calculate endpoint difference ratio
    if max(endpoint_count1, endpoint_count2) > 0:
        endpoint_diff = abs(endpoint_count1 - endpoint_count2) / max(endpoint_count1, endpoint_count2)
    else:
        endpoint_diff = 0

    return {
        'endpoint_diff': endpoint_diff,
        'endpoints_gen': endpoint_count1,
        'endpoints_gt': endpoint_count2,
        'skeleton1': skeleton1,
        'skeleton2': skeleton2
    }

def compute_color_metrics(img1, img2):
    """Compute metrics related to color accuracy."""
    # Calculate color histogram similarity
    hist1 = cv2.calcHist([img1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([img2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

    # Normalize histograms
    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)

    # Calculate histogram correlation
    hist_corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    # Calculate color MSE per channel
    mse_r = np.mean((img1[:,:,0].astype(np.float64) - img2[:,:,0].astype(np.float64))**2)
    mse_g = np.mean((img1[:,:,1].astype(np.float64) - img2[:,:,1].astype(np.float64))**2)
    mse_b = np.mean((img1[:,:,2].astype(np.float64) - img2[:,:,2].astype(np.float64))**2)

    # Calculate Delta E (color difference) - simplified version
    img1_lab = cv2.cvtColor(img1, cv2.COLOR_RGB2Lab)
    img2_lab = cv2.cvtColor(img2, cv2.COLOR_RGB2Lab)

    delta_e = np.sqrt(np.sum((img1_lab.astype(np.float64) - img2_lab.astype(np.float64))**2, axis=2))
    mean_delta_e = np.mean(delta_e)

    return {
        'hist_correlation': hist_corr,
        'mse_r': mse_r,
        'mse_g': mse_g,
        'mse_b': mse_b,
        'mean_delta_e': mean_delta_e
    }

def evaluate_colored_lineart(gen_img, gt_img, line_extract_method="color_gradient"):
    """Evaluate colored line art quality."""
    # Ensure images are in RGB
    if len(gen_img.shape) != 3 or gen_img.shape[2] != 3:
        raise ValueError("Generated image must be a colored image (RGB)")
    if len(gt_img.shape) != 3 or gt_img.shape[2] != 3:
        raise ValueError("Ground truth image must be a colored image (RGB)")

    # Extract line art from colored images
    lines_gen = extract_lines_from_colored(gen_img, adapt_method=line_extract_method)
    lines_gt = extract_lines_from_colored(gt_img, adapt_method=line_extract_method)

    # Compute standard structural metrics
    # Convert to grayscale for SSIM
    gen_gray = cv2.cvtColor(gen_img, cv2.COLOR_RGB2GRAY)
    gt_gray = cv2.cvtColor(gt_img, cv2.COLOR_RGB2GRAY)

    standard_metrics = {
        'mse': mse(gen_gray, gt_gray),
        'psnr': psnr(gen_gray, gt_gray),
        'ssim': ssim(gen_gray, gt_gray)
    }

    # Compute edge metrics on the extracted lines
    edge_metrics = compute_edge_metrics(lines_gen, lines_gt)

    # Compute connectivity metrics on the extracted lines
    connectivity_metrics = compute_connectivity_metrics(lines_gen, lines_gt)

    # Compute color metrics on the original colored images
    color_metrics = compute_color_metrics(gen_img, gt_img)

    # Combine all metrics
    all_metrics = {
        'standard': standard_metrics,
        'edge': edge_metrics,
        'connectivity': connectivity_metrics,
        'color': color_metrics
    }

    # Compute an overall score (weighted combination of key metrics)
    # Normalize Hausdorff and Chamfer for the overall score
    max_distance = np.sqrt(gen_img.shape[0]**2 + gen_img.shape[1]**2)

    if edge_metrics['hausdorff'] != float('inf') and edge_metrics['chamfer'] != float('inf'):
        norm_hausdorff = 1 - min(edge_metrics['hausdorff'] / max_distance, 1)
        norm_chamfer = 1 - min(edge_metrics['chamfer'] / max_distance, 1)
    else:
        norm_hausdorff = 0
        norm_chamfer = 0

    # Colored LineArt Animation Score (CLAS) - includes color metrics with adjusted weights
    # 10% structure, 30% line position, 15% line quality, 45% color accuracy
    clas_score = (
        0.10 * standard_metrics['ssim'] +        # Structural similarity (10%)
        0.10 * edge_metrics['f1_score'] +        # Edge precision/recall balance (10%)
        0.15 * norm_hausdorff +                  # Edge position accuracy (15%)
        0.15 * norm_chamfer +                    # Average edge distance (15%)
        0.05 * (1 - connectivity_metrics['endpoint_diff']) +  # Line connectivity (5%)
        0.15 * color_metrics['hist_correlation'] +  # Color distribution similarity (15%)
        0.15 * (1 - min(color_metrics['mean_delta_e']/100, 1)) +  # Color perceptual accuracy (15%)
        0.15 * (1 - min((color_metrics['mse_r'] + color_metrics['mse_g'] + color_metrics['mse_b'])/(3*255*255), 1))  # Color channel accuracy (15%)
    )

    all_metrics['clas_score'] = clas_score

    return all_metrics, lines_gen, lines_gt, connectivity_metrics['skeleton1'], connectivity_metrics['skeleton2']

def visualize_colored_comparison(gen_img, gt_img, lines_gen, lines_gt, skeleton1, skeleton2, metrics):
    """Visualize the comparison results for colored line art."""
    plt.figure(figsize=(20, 15))

    # Original images
    plt.subplot(3, 3, 1)
    plt.title("Generated Image")
    plt.imshow(gen_img)
    plt.axis('off')

    plt.subplot(3, 3, 2)
    plt.title("Ground Truth")
    plt.imshow(gt_img)
    plt.axis('off')

    # Color difference
    plt.subplot(3, 3, 3)
    plt.title("Color Difference (Delta E)")
    img1_lab = cv2.cvtColor(gen_img, cv2.COLOR_RGB2Lab)
    img2_lab = cv2.cvtColor(gt_img, cv2.COLOR_RGB2Lab)
    delta_e = np.sqrt(np.sum((img1_lab.astype(np.float64) - img2_lab.astype(np.float64))**2, axis=2))
    # Normalize for visualization
    delta_e_norm = delta_e / np.max(delta_e) if np.max(delta_e) > 0 else delta_e
    plt.imshow(delta_e_norm, cmap='hot')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')

    # Extracted line art
    plt.subplot(3, 3, 4)
    plt.title("Generated Lines")
    plt.imshow(lines_gen, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 3, 5)
    plt.title("Ground Truth Lines")
    plt.imshow(lines_gt, cmap='gray')
    plt.axis('off')

    # Edge comparison with legend
    plt.subplot(3, 3, 6)
    plt.title("Line Comparison")
    line_overlay = np.zeros((gen_img.shape[0], gen_img.shape[1], 3), dtype=np.uint8)
    line_overlay[lines_gen > 0] = [255, 0, 0]  # Red for generated edges
    line_overlay[lines_gt > 0] = [0, 0, 255]   # Blue for ground truth edges
    # Purple for overlapping edges
    overlap = np.logical_and(lines_gen > 0, lines_gt > 0)
    line_overlay[overlap] = [255, 0, 255]
    plt.imshow(line_overlay)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', edgecolor='black', label='Generated Lines'),
        Patch(facecolor='blue', edgecolor='black', label='Ground Truth Lines'),
        Patch(facecolor='magenta', edgecolor='black', label='Overlap')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=8)
    plt.axis('off')

    # Skeletonized lines
    plt.subplot(3, 3, 7)
    plt.title("Generated Skeleton")
    plt.imshow(skeleton1, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 3, 8)
    plt.title("Ground Truth Skeleton")
    plt.imshow(skeleton2, cmap='gray')
    plt.axis('off')

    # Metrics summary
    plt.subplot(3, 3, 9)
    plt.title("Metrics Summary")
    plt.axis('off')

    summary_text = f"Colored LineArt Score: {metrics['clas_score']:.4f}\n\n"
    summary_text += f"Line F1 Score: {metrics['edge']['f1_score']:.4f}\n"
    summary_text += f"Line Precision: {metrics['edge']['precision']:.4f}\n"
    summary_text += f"Line Recall: {metrics['edge']['recall']:.4f}\n"

    if metrics['edge']['hausdorff'] != float('inf'):
        summary_text += f"Hausdorff: {metrics['edge']['hausdorff']:.2f} px\n"

    if metrics['edge']['chamfer'] != float('inf'):
        summary_text += f"Chamfer: {metrics['edge']['chamfer']:.2f} px\n"

    summary_text += f"Color Correlation: {metrics['color']['hist_correlation']:.4f}\n"
    summary_text += f"Color Diff (ΔE): {metrics['color']['mean_delta_e']:.2f}\n"

    plt.text(0.1, 0.5, summary_text, fontsize=12, va='center')

    plt.tight_layout()
    plt.show()

def main():
    print("Colored Line Art Animation Metrics")
    print("==================================\n")

    # Option to upload files or use specified paths
    try:
        print("Upload the GENERATED colored image:")
        uploaded_gen = files.upload()
        gen_filenames = list(uploaded_gen.keys())

        if len(gen_filenames) == 0:
            print("No generated image uploaded. Using default filename 'img1.png'.")
            gen_img_path = "img1.png"
        else:
            gen_img_path = gen_filenames[0]
            print(f"Using generated image: {gen_img_path}")

        print("\nNow upload the GROUND TRUTH colored image:")
        uploaded_gt = files.upload()
        gt_filenames = list(uploaded_gt.keys())

        if len(gt_filenames) == 0:
            print("No ground truth image uploaded. Using default filename '0076.png'.")
            gt_img_path = "0076.png"
        else:
            gt_img_path = gt_filenames[0]
            print(f"Using ground truth image: {gt_img_path}")
    except:
        # Use default filenames if upload fails
        print("Upload failed. Using default filenames.")
        gen_img_path = "img1.png"  # Path to generated image
        gt_img_path = "0076.png"   # Path to ground truth image

    try:
        # Load images
        gen_img = load_image(gen_img_path)
        gt_img = load_image(gt_img_path)

        # Resize if dimensions don't match
        if gen_img.shape != gt_img.shape:
            print(f"Resizing ground truth image to match generated image dimensions: {gen_img.shape[:2]}")
            gt_img = cv2.resize(gt_img, (gen_img.shape[1], gen_img.shape[0]))

        print(f"\nEvaluating colored line art quality...")
        print("This may take a moment for line extraction and analysis...")

        # Let user select the line extraction method
        print("\nSelect line extraction method:")
        print("1. Color gradient (best for clean, distinct lines)")
        print("2. Canny edge detection (best for sketchy or varied lines)")
        print("3. Adaptive thresholding (best for simple line art)")

        try:
            choice = int(input("Enter your choice (1-3), or press Enter for default (1): ") or "1")
            if choice == 1:
                method = "color_gradient"
            elif choice == 2:
                method = "canny"
            elif choice == 3:
                method = "adaptive"
            else:
                print("Invalid choice. Using default method (color_gradient).")
                method = "color_gradient"
        except:
            print("Invalid input. Using default method (color_gradient).")
            method = "color_gradient"

        # Evaluate the colored line art
        metrics, lines_gen, lines_gt, skeleton1, skeleton2 = evaluate_colored_lineart(
            gen_img, gt_img, line_extract_method=method
        )

        # Display results
        print("\nEvaluation Results:")
        print("\nLine Art Metrics:")
        print(f"- Edge Precision: {metrics['edge']['precision']:.4f} (Higher is better)")
        print(f"- Edge Recall: {metrics['edge']['recall']:.4f} (Higher is better)")
        print(f"- Edge F1 Score: {metrics['edge']['f1_score']:.4f} (Higher is better)")

        if metrics['edge']['hausdorff'] != float('inf'):
            print(f"- Hausdorff Distance: {metrics['edge']['hausdorff']:.4f} pixels (Lower is better)")
        else:
            print("- Hausdorff Distance: Not computable (insufficient edge points)")

        if metrics['edge']['chamfer'] != float('inf'):
            print(f"- Chamfer Distance: {metrics['edge']['chamfer']:.4f} pixels (Lower is better)")
        else:
            print("- Chamfer Distance: Not computable (insufficient edge points)")

        print("\nLine Connectivity Metrics:")
        print(f"- Endpoint Difference: {metrics['connectivity']['endpoint_diff']:.4f} (Lower is better)")

        print("\nColor Metrics:")
        print(f"- Color Histogram Correlation: {metrics['color']['hist_correlation']:.4f} (Higher is better)")
        print(f"- Mean Color Difference (Delta E): {metrics['color']['mean_delta_e']:.4f} (Lower is better)")
        print(f"- Red Channel MSE: {metrics['color']['mse_r']:.4f}")
        print(f"- Green Channel MSE: {metrics['color']['mse_g']:.4f}")
        print(f"- Blue Channel MSE: {metrics['color']['mse_b']:.4f}")

        print(f"\nColored LineArt Animation Score (CLAS): {metrics['clas_score']:.4f} (Higher is better, scale 0-1)")

        # Visualize the comparison
        visualize_colored_comparison(gen_img, gt_img, lines_gen, lines_gt, skeleton1, skeleton2, metrics)

    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()