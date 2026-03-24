"""
sampler.py - Extract meaningful frames from video
"""

import cv2
import numpy as np

class FrameSampler:
    """Extract key frames from video based on motion"""
    
    def __init__(self, motion_threshold=0.05, min_area=500):
        self.motion_threshold = motion_threshold
        self.min_area = min_area
        self.prev_frame = None
    
    def extract_key_frames(self, video_path, max_frames=12, use_motion=True):
        """
        Extract the most important frames from video
        If use_motion=True, tries to find frames with motion
        If use_motion=False, samples evenly
        """
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if not use_motion or total_frames < max_frames * 2:
            # Simple evenly spaced sampling
            return self._sample_evenly(cap, total_frames, max_frames)
        
        # Motion-based sampling
        return self._sample_by_motion(cap, total_frames, max_frames)
    
    def _sample_evenly(self, cap, total_frames, max_frames):
        """Take evenly spaced frames"""
        frames = []
        indices = np.linspace(0, total_frames-1, max_frames, dtype=int)
        
        for frame_num in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append({
                    "image": frame_rgb,
                    "frame_number": frame_num,
                    "method": "even"
                })
        
        cap.release()
        return frames
    
    def _sample_by_motion(self, cap, total_frames, max_frames):
        """Find frames with most motion/change"""
        motion_scores = []
        frames_data = []
        
        # Scan video for motion
        for frame_num in range(0, total_frames, max(1, total_frames // 100)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Calculate motion score
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            
            if self.prev_frame is not None:
                diff = cv2.absdiff(self.prev_frame, gray)
                thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
                motion_score = np.sum(thresh) / 255.0 / thresh.size
                motion_scores.append((frame_num, motion_score))
                frames_data.append((frame_num, frame))
            
            self.prev_frame = gray
        
        # Sort by motion score (highest first)
        motion_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Take top frames
        selected_frames = []
        for frame_num, _ in motion_scores[:max_frames]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                selected_frames.append({
                    "image": frame_rgb,
                    "frame_number": frame_num,
                    "method": "motion"
                })
        
        # Sort by frame number (chronological)
        selected_frames.sort(key=lambda x: x["frame_number"])
        
        cap.release()
        return selected_frames
    
    def extract_frames_at_indices(self, video_path, frame_indices):
        """Extract specific frames by index"""
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append({
                    "image": frame_rgb,
                    "frame_number": idx
                })
        
        cap.release()
        return frames


# Quick test
if __name__ == "__main__":
    sampler = FrameSampler()
    test_vid = Path("test.mp4")
    
    if test_vid.exists():
        frames = sampler.extract_key_frames(test_vid, max_frames=12)
        print(f"Extracted {len(frames)} frames")