from __future__ import annotations

from datetime import date
from pathlib import Path
from threading import RLock

import torch
from PIL import Image
from torchvision import transforms

from hanwoo.core.config import (
    GALLERY_DIR,
    MATCHING_MODEL_PATH,
    QDRANT_COLLECTION,
    QDRANT_URL,
)
from hanwoo.core.encoders.swin import SiameseViT
from hanwoo.core.vectorstore.qdrant_store import QdrantGalleryStore


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def choose_device(device_name: str = "auto") -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "HANWOO_DEVICE=cuda but CUDA is not available. Run on a GPU host with NVIDIA Container Toolkit."
        )
    return torch.device(device_name)


def torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


class MatchingService:
    def __init__(
        self,
        model_path: Path = MATCHING_MODEL_PATH,
        gallery_dir: Path = GALLERY_DIR,
        device_name: str = "auto",
        qdrant_url: str = QDRANT_URL,
        qdrant_collection: str = QDRANT_COLLECTION,
    ):
        self.model_path = model_path
        self.gallery_dir = gallery_dir
        self.qdrant_url = qdrant_url
        self.qdrant_collection = qdrant_collection
        self.device = choose_device(device_name)
        self.model: SiameseViT | None = None
        self.store: QdrantGalleryStore | None = None
        self.backbone = ""
        self.embedding_dim = 0
        self.image_size = 224
        self.checkpoint_metadata: dict = {}
        self.transform = self._build_transform(self.image_size)
        self.lock = RLock()

    @staticmethod
    def _build_transform(image_size: int):
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Matching checkpoint not found: {self.model_path}")

        checkpoint = torch_load(self.model_path, self.device)
        self.backbone = checkpoint.get("backbone", "swin")
        self.embedding_dim = int(checkpoint.get("embedding_dim", 256))
        self.image_size = int(checkpoint.get("image_size", 224))
        self.transform = self._build_transform(self.image_size)
        self.checkpoint_metadata = {
            "backbone": self.backbone,
            "embedding_dim": self.embedding_dim,
            "image_size": self.image_size,
            "epoch": checkpoint.get("epoch"),
            "metrics": checkpoint.get("metrics"),
        }

        model = SiameseViT(
            backbone=self.backbone,
            embedding_dim=self.embedding_dim,
            image_size=self.image_size,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.device)
        model.eval()
        self.model = model

        self.gallery_dir.mkdir(parents=True, exist_ok=True)
        self.store = QdrantGalleryStore(
            url=self.qdrant_url,
            collection=self.qdrant_collection,
            vector_size=self.embedding_dim,
        )
        self.store.ensure_collection()

    def _ensure_loaded(self) -> SiameseViT:
        if self.model is None:
            raise RuntimeError("Matching model is not loaded")
        return self.model

    def _ensure_store(self) -> QdrantGalleryStore:
        if self.store is None:
            raise RuntimeError("Qdrant gallery store is not initialized")
        return self.store

    @staticmethod
    def safe_stem(filename: str) -> str:
        stem = Path(filename).stem.strip().replace(" ", "_")
        allowed = []
        for ch in stem:
            if ch.isalnum() or ch in {"_", "-", "."}:
                allowed.append(ch)
        return "".join(allowed) or "image"

    @classmethod
    def safe_lot_id(cls, lot_id: str) -> str:
        if not lot_id.strip():
            raise ValueError("lot_id is required")
        return cls.safe_stem(lot_id)

    @staticmethod
    def normalize_capture_date(capture_date: str | None) -> str:
        if capture_date is None or not capture_date.strip():
            return date.today().isoformat()
        return date.fromisoformat(capture_date.strip()).isoformat()

    def embed_image(self, img: Image.Image) -> torch.Tensor:
        model = self._ensure_loaded()
        img_tensor = self.transform(img.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = model(img_tensor).squeeze(0).cpu()
        return emb

    def embed_image_dual(self, img: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        return self.embed_image(img), self.embed_image(img.rotate(180))

    def add_gallery_image(
        self,
        name: str,
        image: Image.Image,
        lot_id: str,
        capture_date: str | None = None,
        preprocessed: bool | None = None,
        rgba_image: Image.Image | None = None,
    ) -> dict:
        with self.lock:
            store = self._ensure_store()
            lot_id = self.safe_lot_id(lot_id)
            capture_date = self.normalize_capture_date(capture_date)
            existing = {
                item["name"]
                for item in store.list_images(
                    lot_id=lot_id,
                    capture_date=capture_date,
                )
            }
            base_name = self.safe_stem(name)
            final_name = base_name
            counter = 2
            while final_name in existing:
                final_name = f"{base_name}_{counter}"
                counter += 1

            save_dir = self.gallery_dir / lot_id / capture_date
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"{final_name}.png"
            image = image.convert("RGB")
            image.save(save_path)
            if rgba_image is not None:
                rgba_dir = save_dir / ".rgba"
                rgba_dir.mkdir(parents=True, exist_ok=True)
                rgba_image.convert("RGBA").save(rgba_dir / f"{final_name}.png")

            emb_orig, emb_rot = self.embed_image_dual(image)
            store.upsert_image(
                name=final_name,
                lot_id=lot_id,
                capture_date=capture_date,
                image_path=save_path,
                original_filename=name,
                original_vector=emb_orig.tolist(),
                rotated_vector=emb_rot.tolist(),
                preprocessed=preprocessed,
            )

        return {
            "name": final_name,
            "lot_id": lot_id,
            "capture_date": capture_date,
            "path": str(save_path),
        }

    def remove_gallery_image(
        self,
        name: str,
        lot_id: str,
        capture_date: str | None = None,
    ) -> bool:
        with self.lock:
            store = self._ensure_store()
            lot_id = self.safe_lot_id(lot_id)
            capture_date = self.normalize_capture_date(capture_date) if capture_date else None
            images = store.list_images(lot_id=lot_id, capture_date=capture_date)
            if name not in {item["name"] for item in images}:
                return False

            store.remove_image(name, lot_id=lot_id, capture_date=capture_date)

            for item in images:
                if item["name"] != name:
                    continue
                image_path = item.get("image_path")
                if image_path:
                    path = Path(str(image_path))
                    if path.exists():
                        path.unlink()
                    rgba_path = path.parent / ".rgba" / path.name
                    if rgba_path.exists():
                        rgba_path.unlink()
        return True

    def clear_gallery(self, lot_id: str, capture_date: str | None = None) -> int:
        with self.lock:
            store = self._ensure_store()
            lot_id = self.safe_lot_id(lot_id)
            capture_date = self.normalize_capture_date(capture_date) if capture_date else None
            images = store.list_images(lot_id=lot_id, capture_date=capture_date)
            count = len(images)
            store.clear(lot_id=lot_id, capture_date=capture_date)
            for item in images:
                image_path = item.get("image_path")
                if image_path:
                    path = Path(str(image_path))
                    if path.exists():
                        path.unlink()
                    rgba_path = path.parent / ".rgba" / path.name
                    if rgba_path.exists():
                        rgba_path.unlink()

            root = self.gallery_dir / lot_id
            if capture_date:
                root = root / capture_date
            if root.exists():
                for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
                    if path.is_dir() and not any(path.iterdir()):
                        path.rmdir()
                if root.is_dir() and not any(root.iterdir()):
                    root.rmdir()
            lot_root = self.gallery_dir / lot_id
            if lot_root.exists() and lot_root.is_dir() and not any(lot_root.iterdir()):
                lot_root.rmdir()
        return count

    def list_gallery(
        self,
        lot_id: str | None = None,
        capture_date: str | None = None,
    ) -> dict:
        with self.lock:
            if lot_id is not None:
                lot_id = self.safe_lot_id(lot_id)
            capture_date = self.normalize_capture_date(capture_date) if capture_date else None
            images = self._ensure_store().list_images(
                lot_id=lot_id,
                capture_date=capture_date,
            )
            return {
                "count": len(images),
                "filenames": [image["name"] for image in images],
                "images": images,
            }

    def find_matches(
        self,
        query_img: Image.Image,
        top_k: int,
        lot_id: str,
        capture_date: str | None = None,
    ) -> list[dict]:
        query_emb = self.embed_image(query_img)
        with self.lock:
            store = self._ensure_store()
            lot_id = self.safe_lot_id(lot_id)
            capture_date = self.normalize_capture_date(capture_date) if capture_date else None
            images = store.list_images(lot_id=lot_id, capture_date=capture_date)
            if not images:
                return []
            candidates = store.search(
                query_emb.tolist(),
                limit=min(len(images) * 2, top_k * 8),
                lot_id=lot_id,
                capture_date=capture_date,
            )

        best_by_name = {}
        for point in candidates:
            key = (point.lot_id, point.capture_date, point.name)
            current = best_by_name.get(key)
            if current is None or point.distance < current.distance:
                best_by_name[key] = point

        results = []
        for rank, point in enumerate(
            sorted(best_by_name.values(), key=lambda item: item.distance)[:top_k],
            start=1,
        ):
            similarity = max(0.0, min(1.0, 1.0 - point.distance / 2.0)) * 100.0
            results.append(
                {
                    "rank": rank,
                    "name": point.name,
                    "lot_id": point.lot_id,
                    "capture_date": point.capture_date,
                    "distance": point.distance,
                    "similarity": float(similarity),
                    "image_path": point.image_path
                    or str(self.gallery_dir / point.lot_id / point.capture_date / f"{point.name}.png"),
                    "matched_variant": point.variant,
                }
            )
        return results
