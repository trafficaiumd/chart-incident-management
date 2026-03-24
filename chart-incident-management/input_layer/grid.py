"""
grid.py - Create 3x4 grid images for VLM input
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class GridCreator:
    """Create composite grid from multiple frames"""
    
    def __init__(self, rows=3, cols=4, cell_size=(300, 300)):
        self.rows = rows
        self.cols = cols
        self.cell_width = cell_size[1]
        self.cell_height = cell_size[0]
        self.grid_width = self.cols * self.cell_width
        self.grid_height = self.rows * self.cell_height
    
    def create_grid(self, frames, location_text="", timestamps=None):
        """
        Create grid from list of frames
        frames: list of images (numpy arrays)
        location_text: text to overlay (e.g., "I-95 NB @ MM 27")
        """
        if not frames:
            return None
        
        # Create blank grid (white background)
        grid = np.ones((self.grid_height, self.grid_width, 3), dtype=np.uint8) * 255
        
        # Determine how many frames to use
        num_frames = min(len(frames), self.rows * self.cols)
        
        # Sample frames evenly if we have more than needed
        if len(frames) > num_frames:
            indices = np.linspace(0, len(frames)-1, num_frames, dtype=int)
            selected = [frames[i] for i in indices]
        else:
            selected = frames
            # Pad with None if needed
            while len(selected) < num_frames:
                selected.append(None)
        
        # Place each frame in grid
        for idx, frame_data in enumerate(selected):
            if frame_data is None:
                continue
            
            row = idx // self.cols
            col = idx % self.cols
            
            # Get image
            if isinstance(frame_data, dict) and "image" in frame_data:
                img = frame_data["image"]
            else:
                img = frame_data  # Assume it's already an image array
            
            # Resize to fit cell
            img_resized = self._resize_to_fit(img, self.cell_width, self.cell_height)
            
            # Calculate position
            y_start = row * self.cell_height
            y_end = y_start + self.cell_height
            x_start = col * self.cell_width
            x_end = x_start + self.cell_width
            
            # Place in grid
            h, w = img_resized.shape[:2]
            y_offset = (self.cell_height - h) // 2
            x_offset = (self.cell_width - w) // 2
            
            grid[y_start+y_offset:y_start+y_offset+h, 
                 x_start+x_offset:x_start+x_offset+w] = img_resized
            
            # Add frame number
            self._add_number(grid, str(idx+1), (x_start+10, y_start+30))
        
        # Add location overlay
        if location_text:
            grid = self._add_text_overlay(grid, location_text, position="top")
        
        return grid
    
    def _resize_to_fit(self, img, target_w, target_h):
        """Resize image to fit within target dimensions (maintain aspect ratio)"""
        h, w = img.shape[:2]
        
        # Calculate scaling factor to fit
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        resized = cv2.resize(img, (new_w, new_h))
        return resized
    
    def _add_number(self, grid, text, position):
        """Add frame number to grid cell"""
        # Convert to PIL for better text
        grid_pil = Image.fromarray(cv2.cvtColor(grid, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(grid_pil)
        
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # Draw black background, white text
        bbox = draw.textbbox(position, text, font=font)
        draw.rectangle(bbox, fill=(0, 0, 0))
        draw.text(position, text, fill=(255, 255, 255), font=font)
        
        # Convert back
        grid[:] = cv2.cvtColor(np.array(grid_pil), cv2.COLOR_RGB2BGR)
    
    def _add_text_overlay(self, grid, text, position="top"):
        """Add text overlay to entire grid"""
        h, w = grid.shape[:2]
        
        grid_pil = Image.fromarray(cv2.cvtColor(grid, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(grid_pil)
        
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        # Position text
        if position == "top":
            x, y = 20, 20
        else:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            x = (w - text_width) // 2
            y = h - 50
        
        # Draw background
        bbox = draw.textbbox((x, y), text, font=font)
        draw.rectangle(bbox, fill=(0, 0, 0, 180))
        draw.text((x, y), text, fill=(255, 255, 255), font=font)
        
        # Convert back
        return cv2.cvtColor(np.array(grid_pil), cv2.COLOR_RGB2BGR)
    
    def save_grid(self, grid, filename):
        """Save grid to file"""
        cv2.imwrite(filename, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    
    def show_grid(self, grid):
        """Display grid"""
        cv2.imshow("Grid", cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# Quick test
if __name__ == "__main__":
    # Create test frames (color gradients)
    frames = []
    for i in range(12):
        frame = np.ones((400, 600, 3), dtype=np.uint8) * (i * 20)
        frames.append(frame)
    
    creator = GridCreator()
    grid = creator.create_grid(frames, location_text="I-95 NB @ MM 27")
    
    if grid is not None:
        creator.save_grid(grid, "test_grid.jpg")
        print("Grid saved to test_grid.jpg")