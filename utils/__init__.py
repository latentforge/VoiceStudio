"""
Utility functions for synthesis pipeline.
"""

from pathlib import Path
from typing import Union

def validate_audio_file(file_path: Union[str, Path]) -> bool:
    """Basic validation for audio files.

    Args:
        file_path: Path to audio file

    Returns:
        True if file is valid, False otherwise
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)

    # Check if file exists
    if not file_path.exists():
        return False

    # Check if file is not empty
    if file_path.stat().st_size == 0:
        return False

    # Check file extension (basic check)
    valid_extensions = {'.wav', '.mp3', '.flac', '.m4a'}
    if file_path.suffix.lower() not in valid_extensions:
        return False

    return True


def create_output_filename(
    prefix: str,
    index: int,
    extension: str = ".wav",
    zero_pad: int = 3
) -> str:
    """Create standardized output filename.

    Args:
        prefix: Filename prefix (e.g., 'ref', 'syn')
        index: Numeric index
        extension: File extension
        zero_pad: Number of digits for zero padding

    Returns:
        Formatted filename
    """
    return f"{prefix}_{index:0{zero_pad}d}{extension}"


__all__ = [
    "validate_audio_file",
    "create_output_filename"
]