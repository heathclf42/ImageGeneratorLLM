"""
Output Management Utility for organizing generated images.

Automatically creates organized folder structures for image generation sessions.
"""

from pathlib import Path
from datetime import datetime
from typing import Optional
import re


class OutputManager:
    """
    Manages organized output directories for image generation.

    Creates structured folders based on session name, date, and type,
    ensuring all generated images are properly organized.

    Example:
        mgr = OutputManager(session_name="portrait_study")
        output_path = mgr.get_output_path("image_001.png")
        # Returns: outputs/portrait_study_20251018/image_001.png
    """

    def __init__(
        self,
        base_dir: str = "outputs",
        session_name: Optional[str] = None,
        add_timestamp: bool = True,
        create_subdirs: bool = True,
    ):
        """
        Initialize output manager.

        Args:
            base_dir: Base output directory (default: "outputs")
            session_name: Name for this generation session (e.g., "lincoln_study")
            add_timestamp: Add date timestamp to folder name (default: True)
            create_subdirs: Auto-create subdirectories like "images", "data" (default: True)
        """
        # Normalize and validate base_dir to prevent nested outputs directories
        base_path = Path(base_dir)

        # Prevent common mistake of nested "outputs/outputs" directories
        # Normalize the path by removing any redundant "outputs" components
        parts = base_path.parts
        if parts.count("outputs") > 1:
            # Remove duplicate "outputs" from path
            cleaned_parts = []
            seen_outputs = False
            for part in parts:
                if part == "outputs":
                    if not seen_outputs:
                        cleaned_parts.append(part)
                        seen_outputs = True
                    # Skip duplicate "outputs"
                else:
                    cleaned_parts.append(part)
            base_path = Path(*cleaned_parts) if cleaned_parts else Path("outputs")

        # If base_dir ends with "outputs", use it directly; otherwise default to "outputs"
        if base_path.name != "outputs" and str(base_path) != "outputs":
            # User specified a non-standard base dir, use it as-is
            self.base_dir = base_path
        else:
            # Standard case: use "outputs" as base
            self.base_dir = Path("outputs")

        self.session_name = session_name or "generation"
        self.add_timestamp = add_timestamp
        self.create_subdirs = create_subdirs

        # Create session directory
        self.session_dir = self._create_session_dir()

        # Create standard subdirectories
        self.images_dir = self.session_dir / "images" if create_subdirs else self.session_dir
        self.data_dir = self.session_dir / "data" if create_subdirs else self.session_dir

        for dir_path in [self.images_dir, self.data_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def _create_session_dir(self) -> Path:
        """Create and return the session directory."""
        # Sanitize session name
        safe_name = re.sub(r'[^\w\s-]', '', self.session_name).strip().replace(' ', '_')

        # Add timestamp if requested
        if self.add_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d")
            dir_name = f"{safe_name}_{timestamp}"
        else:
            dir_name = safe_name

        session_path = self.base_dir / dir_name
        session_path.mkdir(parents=True, exist_ok=True)

        return session_path

    def get_output_path(self, filename: str, subdir: str = "images") -> Path:
        """
        Get full output path for a file.

        Args:
            filename: Name of the file
            subdir: Subdirectory ("images" or "data")

        Returns:
            Full path to the output file
        """
        if subdir == "images":
            return self.images_dir / filename
        elif subdir == "data":
            return self.data_dir / filename
        else:
            return self.session_dir / filename

    def get_relative_path(self, filename: str, subdir: str = "images") -> str:
        """
        Get relative path from current directory.

        Args:
            filename: Name of the file
            subdir: Subdirectory ("images" or "data")

        Returns:
            Relative path as string
        """
        return str(self.get_output_path(filename, subdir))

    def __str__(self) -> str:
        """String representation showing the session directory."""
        return str(self.session_dir)

    def __repr__(self) -> str:
        """Detailed representation."""
        return f"OutputManager(session_dir='{self.session_dir}')"


def create_session_output(session_name: str, base_dir: str = "outputs") -> OutputManager:
    """
    Convenience function to create an output manager for a session.

    Args:
        session_name: Name for the generation session
        base_dir: Base output directory

    Returns:
        Configured OutputManager instance

    Example:
        output_mgr = create_session_output("thermal_test")
        image_path = output_mgr.get_output_path("test_001.png")
    """
    return OutputManager(base_dir=base_dir, session_name=session_name)
