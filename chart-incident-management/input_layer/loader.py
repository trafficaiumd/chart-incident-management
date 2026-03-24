"""
loader.py - Load images or videos for processing
"""

import cv2
import numpy as np
from pathlib import Path

class MediaLoader:
    """Load images or videos from files"""
    
    def __init__(self):
        pass
    
    def load_image(self, path):
        """Load a single image file"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        # Read with OpenCV
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Could not read image: {path}")
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return {
            "success": True,
            "data": img_rgb,
            "source": str(path),
            "type": "image",
            "shape": img_rgb.shape
        }
    
    def load_video(self, path, max_frames=30, sample_evenly=True):
        """Load frames from video file"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        # Open video
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"Video: {total_frames} frames, {fps:.2f} fps, {duration:.1f}s")
        
        # Determine which frames to extract
        if sample_evenly:
            # Take evenly spaced frames
            indices = np.linspace(0, total_frames-1, min(max_frames, total_frames), dtype=int)
        else:
            # Take first N frames
            indices = range(min(max_frames, total_frames))
        
        # Extract frames
        frames = []
        for frame_num in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                frames.append({
                    "image": frame_rgb,
                    "frame_number": frame_num,
                    "timestamp": frame_num / fps if fps > 0 else frame_num,
                    "shape": frame_rgb.shape
                })
        
        cap.release()
        
        return {
            "success": True,
            "frames": frames,
            "count": len(frames),
            "source": str(path),
            "type": "video",
            "fps": fps,
            "duration": duration
        }


# Quick test
if __name__ == "__main__":
    loader = MediaLoader()
    
    # Test with an image (if exists)
    test_img = Path("test.jpg")
    if test_img.exists():
        result = loader.load_image(test_img)
        print(f"Image loaded: {result['shape']}")
    
    # Test with a video (if exists)
    test_vid = Path("test.mp4")
    if test_vid.exists():
        result = loader.load_video(test_vid, max_frames=12)
        print(f"Video loaded: {result['count']} frames")