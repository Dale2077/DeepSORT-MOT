"""Re-ID feature extractor using torchreid or a simple CNN backbone."""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger


class ReIDExtractor:
    """Appearance feature extractor for Re-ID.

    Supports torchreid models (osnet_x0_25, osnet_x1_0, etc.) and
    falls back to a torchvision ResNet18 backbone if torchreid is unavailable.

    Parameters
    ----------
    model_name : str
        torchreid model name (e.g., "osnet_x0_25").
    weights : str or None
        Path to custom weights. None uses ImageNet-pretrained weights.
    input_size : tuple (height, width)
        Input image size for the model.
    batch_size : int
        Maximum batch size for inference.
    device : str
        Compute device.
    """

    def __init__(
        self,
        model_name: str = "osnet_x0_25",
        weights: str = None,
        input_size: tuple = (256, 128),
        batch_size: int = 32,
        device: str = "",
    ):
        self.model_name = model_name
        self.weights = self._resolve_weights_path(weights)
        self.input_size = input_size
        self.batch_size = batch_size

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self._build_model()

    def _resolve_weights_path(self, weights: str = None) -> str:
        """Resolve an explicit or model-default weights path."""
        repo_root = Path(__file__).resolve().parents[2]

        if weights:
            candidates = [Path(weights)]
            if not Path(weights).is_absolute():
                candidates.append(repo_root / weights)
            for candidate in candidates:
                if candidate.is_file():
                    return str(candidate.resolve())
            return str((repo_root / weights).resolve()) if not Path(weights).is_absolute() else str(Path(weights))

        default_weights = {
            "osnet_x0_25": repo_root / "models" / "osnet_x0_25_msmt17.pth",
            "osnet_x1_0": repo_root / "models" / "osnet_x1_0_msmt17.pth",
        }
        candidate = default_weights.get(self.model_name)
        if candidate is not None and candidate.is_file():
            return str(candidate.resolve())
        return None

    def _load_torchreid_weights(self, weights_path: str):
        """Load a torchreid checkpoint while ignoring classifier shape mismatch."""
        checkpoint = torch.load(weights_path, map_location="cpu")
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
                state_dict = checkpoint["state_dict"]
            elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint
        else:
            raise TypeError(f"Unsupported checkpoint type: {type(checkpoint).__name__}")

        model_state = self.model.state_dict()
        compatible_state = {}
        skipped_keys = []

        for key, value in state_dict.items():
            if not hasattr(value, "shape"):
                continue
            clean_key = key.replace("module.", "", 1)
            if clean_key in model_state and model_state[clean_key].shape == value.shape:
                compatible_state[clean_key] = value
            else:
                skipped_keys.append(clean_key)

        if not compatible_state:
            raise RuntimeError(f"No compatible tensors found in checkpoint: {weights_path}")

        incompatible = self.model.load_state_dict(compatible_state, strict=False)
        logger.info(
            f"Loaded torchreid weights from {weights_path} "
            f"({len(compatible_state)} tensors, skipped {len(skipped_keys)})"
        )
        if incompatible.missing_keys:
            logger.debug(f"Missing keys after partial load: {incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            logger.debug(f"Unexpected keys after partial load: {incompatible.unexpected_keys}")

    def _build_model(self):
        """Build the feature extraction model.

        The fallback to ImageNet ResNet18 exists only so unit tests and GUI
        previews work without the heavy ``torchreid`` dependency. When the
        fallback fires during real MOT evaluation the appearance features are
        generic object features (not pedestrian Re-ID), which is enough to
        silently drag DeepSORT's IDF1 below SORT — hence the loud warning
        below. Install ``torchreid`` to use the OSNet MSMT17 checkpoint.
        """
        try:
            import torchreid
            self.model = torchreid.models.build_model(
                name=self.model_name,
                num_classes=1,
                loss="softmax",
                pretrained=self.weights is None,
            )
            if self.weights is not None:
                self._load_torchreid_weights(self.weights)
                logger.info(f"Loaded torchreid model: {self.model_name} ({self.weights})")
            else:
                logger.info(f"Loaded torchreid model: {self.model_name} (pretrained backbone)")
        except ImportError:
            if self.weights is not None and Path(self.weights).is_file():
                logger.error(
                    "torchreid is not installed, but OSNet weights were found at "
                    f"{self.weights}. Falling back to torchvision ResNet18, which "
                    "will NOT use those weights — DeepSORT's IDF1 will underperform. "
                    "Run `pip install torchreid` to fix."
                )
            else:
                logger.warning(
                    "torchreid not available, using torchvision ResNet18 backbone. "
                    "Re-ID features will be generic ImageNet features, not pedestrian "
                    "Re-ID — install torchreid for paper-quality DeepSORT results."
                )
            import torchvision.models as models
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            # Remove classifier, use as feature extractor
            self.model = torch.nn.Sequential(*list(resnet.children())[:-1])
            logger.info("Loaded ResNet18 fallback feature extractor")

        self.model = self.model.eval().to(self.device)

    @torch.no_grad()
    def extract(self, image: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
        """Extract Re-ID features for bounding boxes in an image.

        Parameters
        ----------
        image : ndarray (H, W, 3) BGR
            Full frame image.
        bboxes : ndarray (N, 4)
            Bounding boxes in [x1, y1, x2, y2] format.

        Returns
        -------
        features : ndarray (N, D)
            L2-normalized feature vectors.
        """
        if len(bboxes) == 0:
            return np.array([])

        import cv2

        h, w = image.shape[:2]
        crops = []
        for bbox in bboxes:
            x1 = max(0, int(bbox[0]))
            y1 = max(0, int(bbox[1]))
            x2 = min(w, int(bbox[2]))
            y2 = min(h, int(bbox[3]))

            if x2 <= x1 or y2 <= y1:
                # Invalid bbox, create a zero crop
                crop = np.zeros((self.input_size[0], self.input_size[1], 3), dtype=np.uint8)
            else:
                crop = image[y1:y2, x1:x2]
                crop = cv2.resize(crop, (self.input_size[1], self.input_size[0]))

            # BGR -> RGB, normalize
            crop = crop[:, :, ::-1].astype(np.float32) / 255.0
            crop = (crop - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            crops.append(crop)

        # Process in batches
        all_features = []
        for i in range(0, len(crops), self.batch_size):
            batch = np.array(crops[i : i + self.batch_size])
            batch = torch.from_numpy(batch).permute(0, 3, 1, 2).float().to(self.device)

            features = self.model(batch)
            if features.dim() == 4:
                features = features.mean(dim=[2, 3])
            elif features.dim() == 3:
                features = features.mean(dim=2)
            features = features.view(features.size(0), -1)

            # L2 normalize
            features = F.normalize(features, p=2, dim=1)
            all_features.append(features.cpu().numpy())

        return np.concatenate(all_features, axis=0) if all_features else np.array([])
