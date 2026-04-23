import argparse
import json
import os
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image as PILImage

import aimodel


def analyze_montage(montage_path: str, prompt_template: str, model_name: str) -> tuple[dict, str]:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found in environment/.env")

    genai.configure(api_key=api_key)
    model_candidates = [model_name]
    if model_name != "gemini-2.5-flash":
        model_candidates.append("gemini-2.5-flash")

    camera_info = {
        "id": "",
        "name": "Montage Test Input",
        "description": "2x3 montage diagnostic input",
        "publicVideoURL": "",
        "lat": "",
        "lon": "",
        "routePrefix": "",
        "routeNumber": "",
        "routeSuffix": "",
        "milePost": "",
        "opStatus": "",
        "commMode": "",
        "cctvIp": "",
        "cameraCategories": [],
        "lastCachedDataUpdateTime": "",
    }

    prompt = aimodel.build_prompt(prompt_template, camera_info)
    image = PILImage.open(montage_path)

    last_error = None
    for candidate in model_candidates:
        try:
            model = genai.GenerativeModel(candidate)
            response = model.generate_content([prompt, image])
            text = aimodel.clean_response(response.text)
            output = json.loads(text)
            return aimodel.normalize_output(output, camera_info), candidate
        except Exception as exc:
            last_error = exc
            continue

    raise RuntimeError(f"All model attempts failed: {last_error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Gemini montage diagnostic with existing prompt logic."
    )
    parser.add_argument(
        "--montage_path",
        required=True,
        help="Absolute path to 2x3 montage JPEG.",
    )
    parser.add_argument(
        "--output_path",
        default=str(Path(__file__).resolve().parent / "montage_test_results.json"),
        help="Where to save JSON output.",
    )
    parser.add_argument(
        "--model_name",
        default="gemini-2.0-flash",
        help="Primary Gemini model name (falls back to gemini-2.5-flash on failure).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    montage_path = str(Path(args.montage_path).resolve())
    output_path = str(Path(args.output_path).resolve())
    prompt_path = str((Path(__file__).resolve().parent / "prompt.txt").resolve())

    if not os.path.exists(montage_path):
        raise FileNotFoundError(f"Montage not found: {montage_path}")

    prompt_template = aimodel.load_text(prompt_path)
    result, used_model = analyze_montage(montage_path, prompt_template, args.model_name)

    Path(output_path).write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("Montage diagnostic complete.")
    print(f"montage_path={montage_path}")
    print(f"prompt_path={prompt_path}")
    print(f"model_used={used_model}")
    print(f"output_path={output_path}")
    print(f"incident={result.get('incident', {})}")


if __name__ == "__main__":
    main()
