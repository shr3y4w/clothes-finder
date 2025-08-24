import numpy as np
from feature_extraction import extract_features

def find_similar_images(query_image_path, image_features, color_weight, sift_weight, lbp_weight, top_k=50):
    """
    Find similar images using faster vectorized operations
    """
    query_features = extract_features(query_image_path, color_weight, sift_weight, lbp_weight)
    
    # Convert to numpy arrays for vectorized operations
    query_vector = np.array(query_features)
    image_names = list(image_features.keys())
    feature_vectors = np.array(list(image_features.values()))
    
    # Vectorized distance calculation (much faster than loops)
    distances = np.linalg.norm(feature_vectors - query_vector, axis=1)
    
    # Get indices of top k similar images
    top_indices = np.argsort(distances)[:top_k]
    
    # Return top k similar images
    similar_images = [image_names[i] for i in top_indices]
    
    return [f"/static/cloth/{image}" for image in similar_images]