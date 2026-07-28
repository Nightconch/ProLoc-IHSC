import io
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parent))

from download import (
    ImageValidationError,
    is_blank_rgb,
    normalized_jpeg_bytes,
    write_validated_image,
)


def image_bytes(mode, color, image_format="PNG"):
    image = Image.new(mode, (3, 2), color)
    output = io.BytesIO()
    image.save(output, format=image_format)
    return output.getvalue()


def palette_image_bytes():
    image = Image.new("P", (3, 2), 0)
    image.putpalette([10, 20, 30] + [0, 0, 0] * 255)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def rgba_image_bytes():
    image = Image.new("RGBA", (32, 16), (255, 0, 0, 0))
    for x in range(16, 32):
        for y in range(16):
            image.putpixel((x, y), (20, 30, 40, 255))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def palette_transparency_image_bytes():
    image = Image.new("P", (32, 16), 0)
    image.putpalette([255, 0, 0, 20, 30, 40] + [0, 0, 0] * 254)
    for x in range(16, 32):
        for y in range(16):
            image.putpixel((x, y), 1)
    image.info["transparency"] = 0
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


@contextmanager
def written_image(payload):
    with tempfile.TemporaryDirectory() as temp_dir:
        target = Path(temp_dir) / "image.jpg"
        converted = write_validated_image(payload, target)
        yield target, converted


class DownloadImageContractTest(unittest.TestCase):
    def test_all_channels_at_threshold_are_blank(self):
        image = Image.new("RGB", (2, 2), (250, 250, 250))

        self.assertTrue(is_blank_rgb(image))

    def test_any_channel_below_threshold_is_not_blank(self):
        image = Image.new("RGB", (2, 2), (249, 255, 255))

        self.assertFalse(is_blank_rgb(image))

    def test_rejects_image_at_blank_threshold_after_rgb_conversion(self):
        payload = image_bytes("RGB", (250, 250, 250))

        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "image.jpg"
            with self.assertRaisesRegex(ImageValidationError, "blank"):
                write_validated_image(payload, target)
            self.assertFalse(target.exists())

    def test_rejects_conversion_that_becomes_blank_after_jpeg_encoding(self):
        payload = image_bytes("RGB", (249, 251, 252))

        with self.assertRaisesRegex(ImageValidationError, "final image is blank"):
            normalized_jpeg_bytes(payload)

    def test_rejects_unrecognized_content_as_decode_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "image.jpg"
            with self.assertRaisesRegex(
                ImageValidationError, "image decode failed"
            ):
                write_validated_image(b"not an image", target)
            self.assertFalse(target.exists())

    def test_rejects_truncated_content_as_decode_failure(self):
        payload = image_bytes("RGB", (10, 20, 30), "JPEG")
        truncated = payload[: len(payload) // 2]

        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "image.jpg"
            with self.assertRaisesRegex(
                ImageValidationError, "image decode failed"
            ):
                write_validated_image(truncated, target)
            self.assertFalse(target.exists())

    def test_preserves_nonblank_rgb_jpeg_bytes(self):
        payload = image_bytes("RGB", (10, 20, 30), "JPEG")

        normalized, converted = normalized_jpeg_bytes(payload)

        self.assertFalse(converted)
        self.assertEqual(normalized, payload)

    def test_converts_grayscale_to_rgb_jpeg(self):
        payload = image_bytes("L", 40)

        with written_image(payload) as (target, converted):
            self.assertTrue(converted)
            with Image.open(target) as image:
                image.load()
                self.assertEqual(image.format, "JPEG")
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(len(image.getbands()), 3)

    def test_converts_cmyk_jpeg_to_rgb_jpeg(self):
        payload = image_bytes("CMYK", (0, 128, 255, 0), "JPEG")

        with written_image(payload) as (target, converted):
            self.assertTrue(converted)
            self.assertNotEqual(target.read_bytes(), payload)
            with Image.open(target) as image:
                image.load()
                self.assertEqual(image.format, "JPEG")
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(len(image.getbands()), 3)

    def test_converts_palette_image_to_rgb_jpeg(self):
        with written_image(palette_image_bytes()) as (target, converted):
            self.assertTrue(converted)
            with Image.open(target) as image:
                image.load()
                self.assertEqual(image.format, "JPEG")
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(len(image.getbands()), 3)

    def test_composites_alpha_transparency_on_white(self):
        with written_image(rgba_image_bytes()) as (target, converted):
            self.assertTrue(converted)
            with Image.open(target) as image:
                image.load()
                self.assertTrue(
                    all(channel >= 245 for channel in image.getpixel((2, 8)))
                )

    def test_composites_palette_transparency_on_white(self):
        with written_image(palette_transparency_image_bytes()) as (
            target,
            converted,
        ):
            self.assertTrue(converted)
            with Image.open(target) as image:
                image.load()
                self.assertTrue(
                    all(channel >= 245 for channel in image.getpixel((2, 8)))
                )

    def test_atomically_writes_and_preserves_qualified_jpeg_bytes(self):
        payload = image_bytes("RGB", (10, 20, 30), "JPEG")
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "nested" / "image.jpg"

            converted = write_validated_image(payload, target)

            self.assertFalse(converted)
            self.assertEqual(target.read_bytes(), payload)
            self.assertFalse(target.with_name("image.jpg.part").exists())
            with Image.open(target) as image:
                image.load()
                self.assertEqual(image.format, "JPEG")
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(len(image.getbands()), 3)

    def test_atomic_replace_failure_preserves_target_and_cleans_part_file(self):
        original = image_bytes("RGB", (10, 20, 30), "JPEG")
        replacement = image_bytes("RGB", (40, 50, 60), "JPEG")
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "image.jpg"
            target.write_bytes(original)
            part_path = target.with_name("image.jpg.part")

            with patch.object(
                type(target), "replace", side_effect=OSError("replace failed")
            ):
                with self.assertRaisesRegex(OSError, "replace failed"):
                    write_validated_image(replacement, target)

            self.assertEqual(target.read_bytes(), original)
            self.assertFalse(part_path.exists())

    def test_write_revalidation_rejects_invalid_staged_content(self):
        source = image_bytes("RGB", (10, 20, 30), "JPEG")
        staged_cases = {
            "non-JPEG": image_bytes("RGB", (10, 20, 30), "PNG"),
            "non-RGB": image_bytes("L", 40, "JPEG"),
            "corrupt": b"not an image",
        }
        for case, staged_payload in staged_cases.items():
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temp_dir:
                target = Path(temp_dir) / "image.jpg"
                part_path = target.with_name("image.jpg.part")

                def write_staged(path, _candidate):
                    with path.open("wb") as output:
                        return output.write(staged_payload)

                with patch.object(
                    type(target),
                    "write_bytes",
                    autospec=True,
                    side_effect=write_staged,
                ):
                    with self.assertRaisesRegex(
                        ImageValidationError, "written image"
                    ):
                        write_validated_image(source, target)

                self.assertFalse(target.exists())
                self.assertFalse(part_path.exists())

    def test_write_revalidation_rejects_blank_staged_jpeg(self):
        source = image_bytes("RGB", (10, 20, 30), "JPEG")
        blank_jpeg = image_bytes("RGB", (255, 255, 255), "JPEG")
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "image.jpg"
            part_path = target.with_name("image.jpg.part")

            def write_blank(path, _candidate):
                with path.open("wb") as output:
                    return output.write(blank_jpeg)

            with patch.object(
                type(target),
                "write_bytes",
                autospec=True,
                side_effect=write_blank,
            ):
                with self.assertRaisesRegex(
                    ImageValidationError, "written image is blank"
                ):
                    write_validated_image(source, target)

            self.assertFalse(target.exists())
            self.assertFalse(part_path.exists())


if __name__ == "__main__":
    unittest.main()
