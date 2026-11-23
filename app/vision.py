import os
import mimetypes
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from app.schema import Metadata

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

DEFAULT_MODEL = "gemini-3.6-flash"


def tag_image(image_path: str | Path, model: str = DEFAULT_MODEL) -> Metadata:
    """Send a local image to Gemini and get back schema-validated structured tags."""
    path = Path(image_path)
    with open(path, "rb") as f:
        image_bytes = f.read()

    mime_type, _ = mimetypes.guess_type(str(path))
    mime_type = mime_type or "image/jpeg"

    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            (
                "Analyze this wildlife photo and produce structured tags. "
                "Use snake_case for category (e.g. red_fox, wolf, dog, bear, moose)."
            ),
        ],
        config={
            "response_mime_type": "application/json",
            "response_schema": Metadata,
        },
    )

    if response.parsed is None:
        raise ValueError(f"Model returned invalid or empty structured output for {path}")

    return Metadata.model_validate(response.parsed)


if __name__ == "__main__":
    result = tag_image("data/images/bear/bear_01.jpg")
    print(result)