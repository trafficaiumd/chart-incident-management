import requests
import cv2
import json

def get_weather_condition(lat, lon):
    if not lat or not lon: return "UNKNOWN"
    try:
        url = "[https://api.open-meteo.com/v1/forecast](https://api.open-meteo.com/v1/forecast)"
        params = {"latitude": float(lat), "longitude": float(lon), "current": "weather_code,is_day", "timezone": "auto"}
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        code = data.get("current", {}).get("weather_code")
        
        weather_map = {
            0: "Clear/Sunny", 1: "Mainly Clear", 2: "Partly Cloudy", 3: "Cloudy",
            45: "Foggy", 51: "Light Drizzle", 61: "Light Rain", 63: "Rainy",
            65: "Heavy Rain", 71: "Light Snow", 73: "Snowy", 95: "Thunderstorm"
        }
        return weather_map.get(code, "UNKNOWN")
    except Exception:
        return "UNKNOWN"

def get_city_from_lat_lon(lat, lon):
    if not lat or not lon: return "UNKNOWN"
    try:
        url = "[https://nominatim.openstreetmap.org/reverse](https://nominatim.openstreetmap.org/reverse)"
        params = {"lat": float(lat), "lon": float(lon), "format": "jsonv2"}
        headers = {"User-Agent": "chart-incident-management/1.0"}
        response = requests.get(url, params=params, headers=headers, timeout=5)
        response.raise_for_status()
        address = response.json().get("address", {})
        return address.get("city") or address.get("town") or address.get("county") or "UNKNOWN"
    except Exception:
        return "UNKNOWN"

def overlay_watermark(frame_bgr, logo_path, scale=0.1, margin=20):
    try:
        logo_rgba = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
        if logo_rgba is None: return frame_bgr
        
        fh, fw = frame_bgr.shape[:2]
        target_w = max(1, int(fw * scale))
        aspect = logo_rgba.shape[0] / logo_rgba.shape[1]
        target_h = max(1, int(target_w * aspect))
        
        logo_resized = cv2.resize(logo_rgba, (target_w, target_h), interpolation=cv2.INTER_AREA)
        
        x = fw - target_w - margin
        y = margin
        if x < 0 or y < 0: return frame_bgr
        
        output = frame_bgr.copy()
        logo_bgr = logo_resized[:, :, :3]
        alpha = logo_resized[:, :, 3] / 255.0
        alpha = alpha[:, :, None]
        
        roi = output[y:y+target_h, x:x+target_w]
        output[y:y+target_h, x:x+target_w] = (alpha * logo_bgr + (1 - alpha) * roi).astype("uint8")
        return output
    except Exception:
        return frame_bgr
