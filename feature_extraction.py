import os
import cv2
import numpy as np
from skimage.feature import hog
import pickle
from skimage.feature import local_binary_pattern

COLOR_HIST_SIZE = 512
SIFT_FEATURE_SIZE = 2048

def histogram_equalization(image):
    equalized_image = cv2.equalizeHist(image)
    return equalized_image

def morphological_opening(image, kernel_size=(7, 7)):
    kernel = np.ones(kernel_size, np.uint8)
    opened_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return opened_image

def log_transform(image):
    c = 0.4
    log_image = c * (np.log(image + 1))
    return np.uint8(log_image)

def remove_background_with_enhancements(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 2.0)
    gray_log = log_transform(blurred)
    opened_log = morphological_opening(gray_log)
    gray_eq = histogram_equalization(opened_log)
    # gray_eq_median = cv2.medianBlur(gray_eq, 15)
    
    kernele = np.ones((13,13), np.uint8)
    gray_eq_median_erode = cv2.dilate(gray_eq, kernele, iterations=1)
    _, binary_image = cv2.threshold(gray_eq_median_erode, 137, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((5, 5), np.uint8)
    eroded_image = cv2.erode(binary_image, kernel, iterations=1)
    eroded_image_med = cv2.medianBlur(eroded_image, 15)
    
    edges = cv2.Canny(eroded_image_med, 70, 78)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask = np.zeros_like(binary_image)
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
    
    #use the mask to extract cloth
    result_image = cv2.bitwise_and(image, image, mask=mask)
    
    return result_image

def extract_color_histogram(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, [4, 4, 4], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def extract_sift_features(image, desired_size=SIFT_FEATURE_SIZE):
    resized_image = cv2.resize(image, (256, 256))
    
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)
    
    if descriptors is None:
        return np.zeros(desired_size)
    descriptors = descriptors.flatten()
    
    #handle cases where no of sift features is diff from the desired size
    if descriptors.size < desired_size:
        #pad with zeros if descriptors are smaller than the desired size
        padded_descriptors = np.zeros(desired_size)
        padded_descriptors[:descriptors.size] = descriptors
        return padded_descriptors
    else:
        #truncate if exceeds
        return descriptors[:desired_size]


def extract_lbp_features(image, num_points=24, radius=8):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray_image, num_points, radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, num_points + 3),
                             range=(0, num_points + 2))
    #normalize
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist


def extract_features(image_path, color_weight, sift_weight, lbp_weight):
    image = preprocess_image(image_path)
    
    color_histogram = extract_color_histogram(image)
    sift_features = extract_sift_features(image)
    lbp_features = extract_lbp_features(image)

    color_histogram *= color_weight
    sift_features *= sift_weight
    lbp_features *= lbp_weight
    
    combined_features = np.hstack([color_histogram, sift_features, lbp_features])
    
    return combined_features



def preprocess_image(image_path, resize_dim=(256, 256)):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found or cannot be opened: {image_path}")

    image = cv2.resize(image, resize_dim)
    preprocessed_image = remove_background_with_enhancements(image)
    return preprocessed_image

def save_features_to_disk(image_features, path="features.pkl"):
    """Save the extracted image features to disk."""
    with open(path, 'wb') as f:
        pickle.dump(image_features, f)

def load_features_from_disk(path="features.pkl"):
    """Load the extracted image features from disk."""
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        return None

def load_features():
    image_folder = 'C:\\Users\\shreya\\Desktop\\fcv\\dataset\\cloth'
    feature_file = 'features.pkl' 
    
    image_features = load_features_from_disk(feature_file)
    if image_features is not None:
        print("Loaded features from disk.")
        return image_features

    image_features = {}
    color_weight = 0.8
    sift_weight = 0.5
    lbp_weight = 0.6  
    total_images = len([name for name in os.listdir(image_folder) if name.endswith(('.png', '.jpg', '.jpeg', '.gif'))])
    processed_images = 0

    for filename in os.listdir(image_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.gif')):  
            image_path = os.path.join(image_folder, filename)
            try:
                image_features[filename] = extract_features(image_path, color_weight, sift_weight, lbp_weight)
                processed_images += 1

                if processed_images % 10 == 0:
                    print(f"Processed features for {processed_images} out of {total_images} images.")
            except ValueError as e:
                print(e)  

    save_features_to_disk(image_features, feature_file)
    print("Extracted features with LBP saved to disk.")
    
    return image_features
