"""
demo.py - Quick demo of input layer
Run this to test everything
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from loader import MediaLoader
from sampler import FrameSampler
from grid import GridCreator

def demo_image(image_path):
    """Test with a single image"""
    print(f"\n=== Testing Image: {image_path} ===")
    
    loader = MediaLoader()
    result = loader.load_image(image_path)
    
    if result["success"]:
        print(f"✅ Image loaded: {result['shape']}")
        
        # Create grid (single image repeated)
        creator = GridCreator()
        frames = [result["data"]] * 12  # Repeat to fill grid
        grid = creator.create_grid(frames, location_text="Demo Image")
        
        if grid is not None:
            creator.save_grid(grid, "demo_image_grid.jpg")
            print("✅ Grid saved to demo_image_grid.jpg")
            return True
    
    print("❌ Failed to load image")
    return False

def demo_video(video_path):
    """Test with video file"""
    print(f"\n=== Testing Video: {video_path} ===")
    
    # Load video
    loader = MediaLoader()
    video = loader.load_video(video_path, max_frames=12)
    
    if not video["success"]:
        print("❌ Failed to load video")
        return False
    
    print(f"✅ Loaded {video['count']} frames from video")
    
    # Extract frames
    frames = [f["image"] for f in video["frames"]]
    
    # Create grid
    creator = GridCreator()
    grid = creator.create_grid(
        frames, 
        location_text=f"Demo Video: {Path(video_path).name}"
    )
    
    if grid is not None:
        creator.save_grid(grid, "demo_video_grid.jpg")
        print("✅ Grid saved to demo_video_grid.jpg")
        
        # Print info
        print(f"   Frames: {len(frames)}")
        print(f"   Grid size: {grid.shape}")
        return True
    
    return False

def main():
    """Run demo with available test files"""
    print("=" * 50)
    print("INPUT LAYER DEMO")
    print("=" * 50)
    
    # Check for test images
    image_folder = Path("test_images")
    if image_folder.exists():
        image_files = list(image_folder.glob("*.jpg")) + list(image_folder.glob("*.jpeg")) + list(image_folder.glob("*.png"))
        
        if image_files:
            print(f"\n✅ Found {len(image_files)} images in test_images/")
            for i, image in enumerate(image_files[:4]):  # Test first 4 images
                demo_image(image)
        else:
            print("\n⚠️  No images found in test_images/ folder")
    else:
        print("\n⚠️  test_images/ folder not found")
        # Fallback: look for test.jpg
        test_img = Path("test.jpg")
        if test_img.exists():
            demo_image(test_img)


    # Check for test videos
# Check for videos in demo_videos folder
video_folder = Path("demo_videos")
if video_folder.exists():
    mp4_files = list(video_folder.glob("*.mp4")) + list(video_folder.glob("*.avi")) + list(video_folder.glob("*.mov"))
    
    if mp4_files:
        print(f"\n✅ Found {len(mp4_files)} videos in demo_videos/")
        for i, video in enumerate(mp4_files[:3]):  # Test first 3 videos
            demo_video(video)
    else:
        print("\n⚠️  No videos found in demo_videos/ folder")
else:
    print("\n⚠️  demo_videos/ folder not found")
    
    # Fallback: look in current folder
    mp4_files = list(Path(".").glob("*.mp4"))
    if mp4_files:
        print(f"Found: {[f.name for f in mp4_files]}")
        demo_video(mp4_files[0])
    print("\n" + "=" * 50)
    print("Demo complete")
    print("=" * 50)

if __name__ == "__main__":
    main()