import io
from pathlib import Path

from PIL import Image


class ImageValidationError(ValueError):
    pass


def is_blank_rgb(image: Image.Image, threshold: int = 250) -> bool:
    if image.mode != "RGB" or len(image.getbands()) != 3:
        raise ValueError("blank detection requires an RGB three-band image")
    return all(low >= threshold for low, _high in image.getextrema())


def _is_rgb_jpeg(image: Image.Image) -> bool:
    return (
        image.format == "JPEG"
        and image.mode == "RGB"
        and len(image.getbands()) == 3
    )


def _validate_nonblank_jpeg_rgb(source, stage: str) -> None:
    try:
        with Image.open(source) as checked:
            checked.load()
            if not _is_rgb_jpeg(checked):
                raise ImageValidationError(
                    f"{stage} image is not JPEG RGB with three bands"
                )
            if is_blank_rgb(checked):
                raise ImageValidationError(f"{stage} image is blank")
    except ImageValidationError:
        raise
    except Exception as error:
        raise ImageValidationError(
            f"{stage} image decode failed: {error}"
        ) from error


def normalized_jpeg_bytes(payload: bytes) -> tuple[bytes, bool]:
    try:
        with Image.open(io.BytesIO(payload)) as source:
            source.load()
            preserves_original = _is_rgb_jpeg(source)
            if preserves_original:
                rgb = source
            elif (
                "A" in source.getbands()
                or "transparency" in source.info
            ):
                rgba = source.convert("RGBA")
                background = Image.new(
                    "RGBA", rgba.size, (255, 255, 255, 255)
                )
                background.alpha_composite(rgba)
                rgb = background.convert("RGB")
            else:
                rgb = source.convert("RGB")
            if is_blank_rgb(rgb):
                raise ImageValidationError("blank image")
            if preserves_original:
                candidate = payload
                converted = False
            else:
                output = io.BytesIO()
                rgb.save(output, format="JPEG", quality=95)
                candidate = output.getvalue()
                converted = True
    except ImageValidationError:
        raise
    except Exception as error:
        raise ImageValidationError(f"image decode failed: {error}") from error

    _validate_nonblank_jpeg_rgb(io.BytesIO(candidate), "final")

    return candidate, converted


def write_validated_image(payload: bytes, path: Path) -> bool:
    path = Path(path)
    candidate, converted = normalized_jpeg_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    part_path = path.with_name(f"{path.name}.part")
    part_path.unlink(missing_ok=True)
    try:
        part_path.write_bytes(candidate)
        _validate_nonblank_jpeg_rgb(part_path, "written")
        part_path.replace(path)
        return converted
    finally:
        part_path.unlink(missing_ok=True)
