import csv
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset
import random

# Optional dependency: mediapipe for landmark scores
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
LABEL_TO_INDEX = {"real": 0, "fake": 1}
INDEX_TO_LABEL = {0: "REAL", 1: "FAKE"}
MANIFEST_EXTENSIONS = {".csv"}
MANIFEST_REQUIRED_COLUMNS = {"path", "label"}


class DeepfakeDataset(Dataset):
    def __init__(self, data_dir, transform=None, sample_limit=20000, use_landmarks=True):
        """
        Deepfake Dataset Loader.

        Supported label sources:
        1. Explicit folder structure using exact folder names `real/` and `fake/`
        2. CSV manifest with explicit `path,label` columns

        Returns a tuple (image_tensor, label_tensor, landmark_score_tensor).

        Args:
            data_dir (str): Dataset root directory or CSV manifest path.
            transform: torchvision transforms.
            sample_limit (int): To limit dataset size for quick testing and memory limits. Default 20000.
            use_landmarks (bool): Whether to compute mediapipe facial-landmark score.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.sample_limit = sample_limit
        self.dataset_name = os.path.basename(os.path.abspath(data_dir))
        self.total_real = 0
        self.total_fake = 0
        self.samples = []
        self.label_source = None
        self.use_landmarks = use_landmarks and MEDIAPIPE_AVAILABLE
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) if self.use_landmarks else None

        self._build_samples()

        if self.sample_limit:
            real_samples = [s for s in self.samples if s["label"] == 0]
            fake_samples = [s for s in self.samples if s["label"] == 1]

            random.shuffle(real_samples)
            random.shuffle(fake_samples)

            half_limit = self.sample_limit // 2
            self.samples = real_samples[:half_limit] + fake_samples[:half_limit]
            random.shuffle(self.samples)

        self.total_real = sum(1 for s in self.samples if s["label"] == 0)
        self.total_fake = sum(1 for s in self.samples if s["label"] == 1)

        print(f"Dataset scan complete. Found {len(self.samples)} valid samples.")
        print(f"Label source: {self.label_source}")
        print(f"Class breakdown -> REAL: {self.total_real}, FAKE: {self.total_fake}")

    def _build_samples(self):
        source_path = Path(self.data_dir)
        if not source_path.exists():
            raise FileNotFoundError(f"Dataset source not found: {self.data_dir}")

        if source_path.is_file():
            suffix = source_path.suffix.lower()
            if suffix not in MANIFEST_EXTENSIONS:
                supported = ", ".join(sorted(MANIFEST_EXTENSIONS))
                raise ValueError(f"Unsupported manifest file type: {source_path.name}. Supported types: {supported}")
            self.label_source = f"manifest:{source_path.name}"
            self._load_from_manifest(source_path)
            return

        self.label_source = "folders:real-fake"
        self._load_from_explicit_folders(source_path)

    def _load_from_explicit_folders(self, root_dir: Path):
        print(f"Scanning directory: {root_dir} using explicit folder labels...")
        for path in root_dir.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            label = self._label_from_path(path)
            if label is None:
                continue

            self.samples.append({
                "path": str(path),
                "label": label,
                "type": "image"
            })

        if not self.samples:
            raise ValueError(
                "No labeled images found. Expected exact folder names `real/` and `fake/` somewhere under the dataset root, "
                "or pass a CSV manifest with explicit labels."
            )

    def _label_from_path(self, path: Path):
        for parent in path.parents:
            normalized = parent.name.strip().lower()
            if normalized in LABEL_TO_INDEX:
                return LABEL_TO_INDEX[normalized]
        return None

    def _load_from_manifest(self, manifest_path: Path):
        print(f"Scanning manifest: {manifest_path}")
        with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError("Manifest file is empty.")

            fieldnames = {field.strip().lower() for field in reader.fieldnames if field}
            missing = MANIFEST_REQUIRED_COLUMNS - fieldnames
            if missing:
                required = ", ".join(sorted(MANIFEST_REQUIRED_COLUMNS))
                raise ValueError(f"Manifest must include columns: {required}. Missing: {', '.join(sorted(missing))}")

            for row_number, row in enumerate(reader, start=2):
                resolved_row = {str(k).strip().lower(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
                rel_path = resolved_row.get("path", "")
                raw_label = resolved_row.get("label", "")

                if not rel_path:
                    raise ValueError(f"Manifest row {row_number} is missing `path`.")

                image_path = Path(rel_path)
                if not image_path.is_absolute():
                    image_path = (manifest_path.parent / image_path).resolve()

                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    raise ValueError(f"Manifest row {row_number} references unsupported file type: {image_path}")

                if not image_path.exists():
                    raise FileNotFoundError(f"Manifest row {row_number} references missing file: {image_path}")

                label = self._normalize_manifest_label(raw_label, row_number)
                self.samples.append({
                    "path": str(image_path),
                    "label": label,
                    "type": "image"
                })

        if not self.samples:
            raise ValueError("Manifest did not produce any valid labeled samples.")

    def _normalize_manifest_label(self, raw_label, row_number):
        normalized = str(raw_label).strip().lower()
        if normalized in LABEL_TO_INDEX:
            return LABEL_TO_INDEX[normalized]
        if normalized in {"0", "1"}:
            return int(normalized)
        raise ValueError(
            f"Manifest row {row_number} has invalid label `{raw_label}`. Use one of: real, fake, 0, 1."
        )

    def __len__(self):
        return len(self.samples)

    def _compute_landmark_score(self, pil_image):
        if not self.use_landmarks or self.mp_face_mesh is None:
            return 0.5

        image_np = np.array(pil_image)
        results = self.mp_face_mesh.process(image_np)
        if not results.multi_face_landmarks:
            return 0.5

        face_landmarks = results.multi_face_landmarks[0].landmark
        coords = np.array([[lm.x, lm.y] for lm in face_landmarks])

        eye_left = coords[33]
        eye_right = coords[263]
        nose = coords[1]
        mouth = coords[0]
        eye_dist = np.linalg.norm(eye_left - eye_right)
        nose_eye = np.linalg.norm(nose - (eye_left + eye_right) / 2)
        mouth_nose = np.linalg.norm(mouth - nose)

        if eye_dist == 0 or nose_eye == 0 or mouth_nose == 0:
            return 0.5

        ratio1 = nose_eye / eye_dist
        ratio2 = mouth_nose / eye_dist
        ideal1, ideal2 = 0.35, 0.45
        score = 1.0 - (abs(ratio1 - ideal1) + abs(ratio2 - ideal2)) / 1.0
        score = np.clip((score + 1) / 2, 0.0, 1.0)
        return float(score)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        path = sample["path"]
        label = sample["label"]

        try:
            img = Image.open(path).convert('RGB')
        except Exception as e:
            img = Image.new("RGB", (224, 224), color="black")

        landmark_score = self._compute_landmark_score(img)

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long), torch.tensor(landmark_score, dtype=torch.float32)
