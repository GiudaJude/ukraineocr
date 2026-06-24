import os
import shutil
from pathlib import Path

import cv2
import pytest

import gemini_ukr_ocr as ocr

SAMPLE_DATA_DIR = Path("sample_data")
IMAGE_SUFFIXES = {".jpg", ".jpeg"}


def _downscale_image_in_place(image_path: Path, max_dimension: int) -> None:
    image = cv2.imread(str(image_path))
    assert image is not None, f"OpenCV could not read {image_path}"

    height, width = image.shape[:2]
    current_max = max(height, width)
    if current_max <= max_dimension:
        return

    scale = max_dimension / current_max
    resized = cv2.resize(
        image,
        (int(width * scale), int(height * scale)),
        interpolation=cv2.INTER_AREA,
    )
    cv2.imwrite(str(image_path), resized)


def _sample_images() -> list[Path]:
    if not SAMPLE_DATA_DIR.exists():
        return []
    return sorted(
        path
        for path in SAMPLE_DATA_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def test_sample_data_images_are_readable() -> None:
    images = _sample_images()
    if not images:
        pytest.skip("sample_data is not present in this workspace")

    # Spot check a few images so the corpus can be validated without API usage.
    for image_path in images[: min(3, len(images))]:
        image = cv2.imread(str(image_path))
        assert image is not None, f"OpenCV could not read {image_path}"
        assert image.size > 0, f"OpenCV loaded an empty image from {image_path}"


@pytest.mark.integration
def test_live_ocr_smoke_on_sample_data(tmp_path: Path) -> None:
    if os.getenv("RUN_SAMPLE_OCR_LIVE") != "1":
        pytest.skip("Set RUN_SAMPLE_OCR_LIVE=1 to run live OCR smoke tests")

    images = _sample_images()
    if not images:
        pytest.skip("sample_data is not present in this workspace")

    limit = int(os.getenv("SAMPLE_OCR_LIMIT", "2"))
    max_dimension = int(os.getenv("SAMPLE_OCR_MAX_DIM", "1600"))
    selected_images = images[: max(1, min(limit, len(images)))]

    staged_dirs: set[Path] = set()
    for source_image in selected_images:
        relative_parent = source_image.parent.relative_to(SAMPLE_DATA_DIR)
        staged_dir = tmp_path / relative_parent
        staged_dir.mkdir(parents=True, exist_ok=True)
        staged_image = staged_dir / source_image.name
        shutil.copy2(source_image, staged_image)
        _downscale_image_in_place(staged_image, max_dimension)
        staged_dirs.add(staged_dir)

    for staged_dir in sorted(staged_dirs):
        ocr.process_dir(staged_dir)

    for source_image in selected_images:
        relative_parent = source_image.parent.relative_to(SAMPLE_DATA_DIR)
        staged_image = tmp_path / relative_parent / source_image.name
        output_dir = ocr.get_output_directory(staged_image.parent)
        parsed_output = output_dir / f"{staged_image.name}.parsed.txt"
        clean_output = output_dir / f"{staged_image.name}.txt"

        assert parsed_output.exists(), f"Missing parsed output for {source_image.name}"
        assert clean_output.exists(), f"Missing clean output for {source_image.name}"
        assert parsed_output.read_text(encoding="utf-8").strip()
        assert clean_output.read_text(encoding="utf-8").strip()
