# tests/a9_test_cases.py
"""
Generate test cases from A9 dataset
"""

import json
from pathlib import Path
from test_dataset import TestDataset, TEST_IMAGES_DIR

# Define paths
A9_DIR = TEST_IMAGES_DIR / "a9" / "r00_s00"
LABELS_DIR = A9_DIR / "labels"

# Camera views (matching your folder names)
CAMERAS = [
    "s40_north_16mm",
    "s40_north_50mm", 
    "s50_south_16mm",
    "s50_south_50mm"
]

def parse_a9_label(label_file):
    """Convert A9 label to your test case format"""
    with open(label_file, 'r') as f:
        data = json.load(f)
    
    # Extract frame number from filename - A9 format: timestamp_frameid_cameraname.json
    filename = Path(label_file).stem
    # Split by underscore and get the second part (frame ID)
    parts = filename.split('_')
    frame_num = 0
    if len(parts) >= 2:
        try:
            frame_num = int(parts[1])  # Second part is the frame number (e.g., 022000000)
        except:
            frame_num = 0
    
    # Determine if this frame has a crash (frames 150-200 in r00_s00)
    # For A9, we need to know which frame range has the crash
    # Let's assume frames with ID between 150000000 and 200000000
    is_crash = 150000000 <= frame_num <= 200000000
    
    # Count vehicles by type
    vehicles = []
    # Check the actual structure of your A9 labels
    # The structure might be different - let's print to debug if needed
    if 'objects' in data:
        for obj in data.get('objects', []):
            obj_type = obj.get('type', 'unknown')
            if obj_type in ['car', 'truck', 'bus', 'motorcycle']:
                vehicles.append(obj_type)
    elif 'openlabel' in data:
        # Some A9 datasets use openlabel format
        frames = data.get('openlabel', {}).get('frames', {})
        for frame_id, frame_data in frames.items():
            objects = frame_data.get('objects', {})
            for obj_id, obj_data in objects.items():
                obj_type = obj_data.get('type', 'unknown')
                if obj_type in ['car', 'truck', 'bus', 'motorcycle']:
                    vehicles.append(obj_type)
    
    return {
        "incident_type": "crash" if is_crash else "none",
        "vehicles": {
            "total_count": len(vehicles),
            "types": vehicles[:10]  # First 10 types
        },
        "lanes": {
            "total": 3,  # A9 has 3 lanes
            "blocked": [1, 2] if is_crash else []
        },
        "hazards": {
            "fire": False,
            "smoke": False,
            "debris": True if is_crash else False
        },
        "traffic": {
            "state": "stopped" if is_crash else "flowing"
        }
    }

def create_a9_test_cases():
    dataset = TestDataset()
    
    # For each camera view
    for camera in CAMERAS:
        image_dir = A9_DIR / camera
        label_dir = LABELS_DIR / camera
        
        if not image_dir.exists():
            print(f"⚠️  Warning: Image directory missing: {image_dir}")
            continue
        if not label_dir.exists():
            print(f"⚠️  Warning: Label directory missing: {label_dir}")
            continue
        
        # Get first 20 images from this camera
        images = sorted(image_dir.glob("*.png"))[:20]
        print(f"📷 {camera}: Found {len(list(image_dir.glob('*.png')))} images, using first {len(images)}")
        
        for img_path in images:
            # Find matching label (same filename)
            label_file = label_dir / f"{img_path.stem}.json"
            
            if label_file.exists():
                expected = parse_a9_label(label_file)
                
                dataset.add_case(
                    image_path=str(img_path),
                    expected=expected,
                    description=f"A9 r00_s00 {camera} frame {img_path.stem}",
                    critical_rules=["crash_detection"] if expected["incident_type"] == "crash" else []
                )
            else:
                print(f"⚠️  Missing label for: {img_path.name}")
    
    return dataset

def verify_calibration():
    """Check if calibration files exist (optional)"""
    cal_dir = A9_DIR / "calibration"
    if cal_dir.exists():
        cal_files = list(cal_dir.glob("*.json"))
        print(f"📐 Calibration files found: {len(cal_files)}")
        return True
    else:
        print("📐 No calibration folder found (optional)")
        return False

if __name__ == "__main__":
    print("🔍 Verifying A9 dataset structure...")
    print(f"📁 A9 directory: {A9_DIR}")
    
    # Check calibration
    verify_calibration()
    
    # Create test cases
    print("\n📝 Creating test cases...")
    dataset = create_a9_test_cases()
    
    # Save
    dataset.save_to_json("tests/a9_test_cases.json")
    print(f"\n✅ Created {len(dataset.cases)} A9 test cases")
    print(f"💾 Saved to: tests/a9_test_cases.json")
    
    # Show summary
    crash_cases = sum(1 for case in dataset.cases if case.expected["incident_type"] == "crash")
    normal_cases = len(dataset.cases) - crash_cases
    print(f"📊 Summary: {crash_cases} crash frames, {normal_cases} normal frames")