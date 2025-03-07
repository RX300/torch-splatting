import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_tiny_nerf_data(file_path):
    """
    Visualize images and camera poses from tiny_nerf_data.npz
    """
    data = np.load(file_path)
    images = data['images']
    poses = data['poses']
    focal = float(data['focal'])
    
    print("Dataset information:")
    print(f"Number of images: {len(images)}")
    print(f"Image shape: {images[0].shape}")
    print(f"Focal length: {focal}")
    
    # Create output directory
    os.makedirs("tiny_nerf_visualization", exist_ok=True)
    
    # Visualize camera poses
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract camera positions and orientations from poses
    for i, pose in enumerate(poses):
        # The camera position is the translation part of the pose matrix
        cam_pos = pose[:3, 3]
        
        # The camera orientation is given by the rotation part
        cam_forward = pose[:3, 2]  # Z-axis
        cam_up = -pose[:3, 1]      # -Y-axis
        cam_right = pose[:3, 0]    # X-axis
        
        # Plot camera position
        ax.scatter(cam_pos[0], cam_pos[1], cam_pos[2], c='r', marker='o')
        
        # Plot camera orientation
        scale = 0.1
        ax.quiver(cam_pos[0], cam_pos[1], cam_pos[2], 
                  cam_forward[0], cam_forward[1], cam_forward[2], 
                  color='b', length=scale)
        ax.quiver(cam_pos[0], cam_pos[1], cam_pos[2], 
                  cam_up[0], cam_up[1], cam_up[2], 
                  color='g', length=scale)
        ax.quiver(cam_pos[0], cam_pos[1], cam_pos[2], 
                  cam_right[0], cam_right[1], cam_right[2], 
                  color='r', length=scale)
        
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Poses')
    plt.savefig("tiny_nerf_visualization/camera_poses.png")
    
    # Visualize some images
    num_to_visualize = min(5, len(images))
    fig, axes = plt.subplots(1, num_to_visualize, figsize=(15, 3))
    
    for i in range(num_to_visualize):
        axes[i].imshow(images[i])
        axes[i].set_title(f"Image {i}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig("tiny_nerf_visualization/sample_images.png")
    
    print("Visualization saved to 'tiny_nerf_visualization' folder")

if __name__ == "__main__":
    # Path to your tiny_nerf_data.npz file
    tiny_nerf_path = 'tiny_nerf_data.npz'
    visualize_tiny_nerf_data(tiny_nerf_path)