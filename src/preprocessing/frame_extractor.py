"""
frame_extractor.py - Video frame extraction and face detection using MediaPipe.

Pipeline:
  1. Open video with OpenCV
  2. Extract every N-th frame
  3. Detect faces in each frame using MediaPipe Face Detection
  4. Crop and pad face regions
  5. Return list of face-cropped PIL Images or numpy arrays
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FaceExtractor:
    """
    Extracts face crops from video frames using MediaPipe Face Detection.

    MediaPipe's BlazeFace model is fast and accurate, ideal for
    real-time video processing. We use model_selection=1 for full-range
    detection (works for faces up to ~5 meters away).
    """

    def __init__(
        self,
        frame_interval: int = 10,
        max_frames: int = 30,
        min_frames: int = 5,
        face_margin: float = 0.3,
        min_face_size: int = 50,
        confidence_threshold: float = 0.5,
    ):
        """
        Initialize the FaceExtractor.

        Args:
            frame_interval (int): Extract 1 frame every N frames.
            max_frames (int): Maximum frames to extract per video.
            min_frames (int): Minimum frames required for valid processing.
            face_margin (float): Fractional padding around face bounding box.
            min_face_size (int): Minimum face width/height in pixels to keep.
            confidence_threshold (float): MediaPipe detection confidence cutoff.
        """
        self.frame_interval = frame_interval
        self.max_frames = max_frames
        self.min_frames = min_frames
        self.face_margin = face_margin
        self.min_face_size = min_face_size
        self.confidence_threshold = confidence_threshold

        # Initialize MediaPipe Face Detection (supports mp 0.9+ and 0.10+)
        try:
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detector = self.mp_face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=confidence_threshold,
            )
        except AttributeError:
            # Newer mediapipe API (>= 0.10.14)
            import mediapipe as _mp
            BaseOptions = _mp.tasks.BaseOptions
            FaceDetector = _mp.tasks.vision.FaceDetector
            FaceDetectorOptions = _mp.tasks.vision.FaceDetectorOptions
            VisionRunningMode = _mp.tasks.vision.RunningMode
            options = FaceDetectorOptions(
                base_options=BaseOptions(model_asset_path=None),
                running_mode=VisionRunningMode.IMAGE,
                min_detection_confidence=confidence_threshold,
            )
            self.mp_face_detection = None
            self.face_detector = None
            self._use_new_api = True
            logger.warning(
                "MediaPipe solutions API unavailable. Face detection will use fallback (full frame)."
            )

        logger.info(
            f"FaceExtractor initialized | "
            f"interval={frame_interval}, max_frames={max_frames}, "
            f"margin={face_margin}"
        )

    def extract_faces_from_video(
        self,
        video_path: str,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> List[np.ndarray]:
        """
        Extract face crops from a video file.

        Args:
            video_path (str): Path to input video file.
            target_size (tuple): (width, height) to resize face crops to.
                                  If None, returns original crop sizes.

        Returns:
            List[np.ndarray]: List of RGB face crops as numpy arrays (H, W, 3).

        Raises:
            FileNotFoundError: If video file doesn't exist.
            ValueError: If video cannot be opened or has too few faces.
        """
        video_path = str(video_path)
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0

        logger.info(
            f"Processing video: {Path(video_path).name} | "
            f"frames={total_frames}, fps={fps:.1f}, duration={duration:.1f}s"
        )

        face_crops = []
        frame_idx = 0
        extracted_count = 0

        try:
            while cap.isOpened() and extracted_count < self.max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                # Only process every N-th frame
                if frame_idx % self.frame_interval == 0:
                    # Convert BGR (OpenCV default) to RGB (MediaPipe expects RGB)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Detect and crop faces
                    crops = self._detect_and_crop_faces(rgb_frame, target_size)
                    face_crops.extend(crops)
                    extracted_count += 1

                frame_idx += 1

        finally:
            cap.release()

        logger.info(
            f"Extracted {len(face_crops)} face crops from "
            f"{extracted_count} frames sampled"
        )

        if len(face_crops) < self.min_frames:
            logger.warning(
                f"Only {len(face_crops)} faces found (minimum: {self.min_frames}). "
                "Consider using a lower frame_interval or checking video quality."
            )

        return face_crops

    def _detect_and_crop_faces(
        self,
        rgb_frame: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> List[np.ndarray]:
        """
        Detect faces in a single frame and return cropped face regions.

        Args:
            rgb_frame (np.ndarray): RGB image array (H, W, 3).
            target_size (tuple): Target resize dimensions (W, H).

        Returns:
            List[np.ndarray]: List of face crops (may be empty if no face found).
        """
        h, w = rgb_frame.shape[:2]
        if self.face_detector is None:
            # Fallback: return full frame resized to target_size
            if target_size is not None:
                resized = cv2.resize(rgb_frame, target_size, interpolation=cv2.INTER_LANCZOS4)
                return [resized]
            return [rgb_frame]
        results = self.face_detector.process(rgb_frame)
        if not results.detections:
            return []

        crops = []
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box

            # Convert relative coordinates to absolute pixels
            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            box_w = int(bbox.width * w)
            box_h = int(bbox.height * h)

            # Apply margin padding around face
            margin_x = int(box_w * self.face_margin)
            margin_y = int(box_h * self.face_margin)

            # Clamp coordinates to image boundaries
            x1_pad = max(0, x1 - margin_x)
            y1_pad = max(0, y1 - margin_y)
            x2_pad = min(w, x1 + box_w + margin_x)
            y2_pad = min(h, y1 + box_h + margin_y)

            # Skip faces smaller than minimum size
            crop_w = x2_pad - x1_pad
            crop_h = y2_pad - y1_pad
            if crop_w < self.min_face_size or crop_h < self.min_face_size:
                continue

            # Crop the face
            face_crop = rgb_frame[y1_pad:y2_pad, x1_pad:x2_pad]

            # Resize to target size if specified
            if target_size is not None:
                face_crop = cv2.resize(face_crop, target_size, interpolation=cv2.INTER_LANCZOS4)

            crops.append(face_crop)

        return crops

    def extract_faces_to_disk(
        self,
        video_path: str,
        output_dir: str,
        target_size: Optional[Tuple[int, int]] = (224, 224),
        label: Optional[str] = None,
    ) -> List[str]:
        """
        Extract face crops from a video and save them as JPEG files.

        Used during dataset preparation to convert raw videos into
        the image dataset format expected by the DataLoader.

        Args:
            video_path (str): Input video path.
            output_dir (str): Directory to save face images.
            target_size (tuple): Resize target (width, height).
            label (str): Subdirectory label ("real" or "fake").

        Returns:
            List[str]: Paths to saved face image files.
        """
        # Determine output subdirectory
        save_dir = Path(output_dir)
        if label:
            save_dir = save_dir / label
        save_dir.mkdir(parents=True, exist_ok=True)

        # Get video stem for filename prefix
        video_stem = Path(video_path).stem

        # Extract faces
        faces = self.extract_faces_from_video(video_path, target_size=target_size)

        saved_paths = []
        for idx, face in enumerate(faces):
            filename = f"{video_stem}_face_{idx:04d}.jpg"
            save_path = save_dir / filename

            # Convert RGB to BGR for OpenCV saving
            bgr_face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(save_path), bgr_face, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_paths.append(str(save_path))

        logger.info(
            f"Saved {len(saved_paths)} face crops from '{Path(video_path).name}' "
            f"to '{save_dir}'"
        )
        return saved_paths

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self.face_detector is not None:
            try:
                self.face_detector.close()
            except Exception:
                pass
        logger.debug("FaceExtractor resources released.")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def extract_raw_frames(
    video_path: str,
    frame_interval: int = 10,
    max_frames: int = 30,
    target_size: Optional[Tuple[int, int]] = None,
) -> List[np.ndarray]:
    """
    Extract raw video frames WITHOUT face detection.

    Useful as a fallback when no faces are detected.

    Args:
        video_path (str): Path to video file.
        frame_interval (int): Extract every N-th frame.
        max_frames (int): Maximum frames to extract.
        target_size (tuple): Resize each frame to this size (W, H).

    Returns:
        List[np.ndarray]: List of RGB frames as numpy arrays.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frames = []
    frame_idx = 0

    try:
        while cap.isOpened() and len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if target_size:
                    rgb = cv2.resize(rgb, target_size, interpolation=cv2.INTER_LANCZOS4)
                frames.append(rgb)

            frame_idx += 1
    finally:
        cap.release()

    return frames


# ─────────────────────────────────────────────────────────────────────────────
# Image-Mode Face Extraction (added for real dataset support)
# ─────────────────────────────────────────────────────────────────────────────

def extract_face_from_image(
    rgb_image: np.ndarray,
    extractor: "FaceExtractor" = None,
    target_size: Optional[Tuple[int, int]] = (224, 224),
    fallback_full_image: bool = True,
    face_margin: float = 0.3,
    min_face_size: int = 50,
    confidence_threshold: float = 0.5,
) -> List[np.ndarray]:
    """
    Detect and crop faces from a single RGB image.

    Used for IMAGE MODE inference and preprocessing.
    Falls back to the full resized image when no face is detected.

    Args:
        rgb_image: Input RGB image (H, W, 3) numpy array.
        extractor: Optional existing FaceExtractor instance to reuse.
                   If None, a temporary one is created.
        target_size: Resize output crops to this (W, H). None = no resize.
        fallback_full_image: If True and no face detected, return the full image.
        face_margin: Fractional padding around detected face.
        min_face_size: Minimum face dimension in pixels.
        confidence_threshold: MediaPipe detection confidence.

    Returns:
        List[np.ndarray]: List of RGB face crops (may contain one full image as fallback).
    """
    own_extractor = extractor is None
    if own_extractor:
        extractor = FaceExtractor(
            frame_interval=1,
            max_frames=1,
            face_margin=face_margin,
            min_face_size=min_face_size,
            confidence_threshold=confidence_threshold,
        )

    try:
        crops = extractor._detect_and_crop_faces(rgb_image, target_size=target_size)
    except Exception as e:
        logger.warning(f"Face detection failed on image: {e}")
        crops = []
    finally:
        if own_extractor:
            extractor.close()

    if not crops and fallback_full_image:
        logger.debug("No face detected in image — falling back to full image.")
        if target_size is not None:
            resized = cv2.resize(rgb_image, target_size, interpolation=cv2.INTER_LANCZOS4)
            crops = [resized]
        else:
            crops = [rgb_image]

    return crops


def preprocess_image_for_inference(
    image_input,
    target_size: int = 224,
    face_margin: float = 0.2,
    fallback_full_image: bool = True,
) -> Tuple[List[np.ndarray], bool]:
    """
    High-level image preprocessing for single-image inference.

    Accepts file path (str), PIL Image, or numpy array.

    Returns:
        (face_crops, face_detected)
        face_crops: List of RGB numpy arrays ready for model input.
        face_detected: True if at least one real face was found.
    """
    from PIL import Image as PILImage

    # Normalize input to numpy RGB array
    if isinstance(image_input, str):
        bgr = cv2.imread(image_input)
        if bgr is None:
            raise ValueError(f"Could not read image: {image_input}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, PILImage.Image):
        rgb = np.array(image_input.convert("RGB"))
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 3 and image_input.shape[2] == 3:
            rgb = image_input
        else:
            raise ValueError(f"Unexpected array shape: {image_input.shape}")
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    extractor = FaceExtractor(
        frame_interval=1,
        max_frames=1,
        face_margin=face_margin,
        min_face_size=40,
        confidence_threshold=0.4,
    )

    crops = extractor._detect_and_crop_faces(rgb, target_size=(target_size, target_size))
    face_detected = len(crops) > 0

    if not crops and fallback_full_image:
        resized = cv2.resize(rgb, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
        crops = [resized]

    extractor.close()
    return crops, face_detected
