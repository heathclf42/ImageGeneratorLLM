"""
Interactive text removal tool.

This tool lets you visually draw rectangles over text areas you want to remove.
Much more precise than hardcoded coordinates!
"""

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backend_bases import MouseButton
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InteractiveTextRemover:
    """Interactive tool for selecting text areas to remove."""

    def __init__(self, image_path: str):
        """
        Initialize the interactive text remover.

        Args:
            image_path: Path to the image file
        """
        self.image_path = image_path
        self.image = Image.open(image_path).convert('RGB')
        self.width, self.height = self.image.size

        # Store rectangles as (x1, y1, x2, y2)
        self.rectangles = []

        # Current drawing state
        self.drawing = False
        self.current_rect_start = None

        # Setup matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.imshow(self.image)
        self.ax.set_title(
            "INSTRUCTIONS:\n"
            "1. Click and drag to draw rectangles over text\n"
            "2. Press 'u' to undo last rectangle\n"
            "3. Press 'r' to reset all rectangles\n"
            "4. Close window when done to save",
            fontsize=10,
            pad=15
        )

        # Connect mouse events
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        # Current preview rectangle
        self.preview_rect = None

        logger.info("Interactive mask tool started")
        logger.info("Draw rectangles over text areas, then close window")

    def on_press(self, event):
        """Handle mouse press - start drawing rectangle."""
        if event.inaxes != self.ax:
            return
        if event.button != MouseButton.LEFT:
            return

        self.drawing = True
        self.current_rect_start = (event.xdata, event.ydata)
        logger.info(f"Started rectangle at ({event.xdata:.0f}, {event.ydata:.0f})")

    def on_motion(self, event):
        """Handle mouse motion - update preview rectangle."""
        if not self.drawing or event.inaxes != self.ax:
            return

        # Remove old preview
        if self.preview_rect:
            self.preview_rect.remove()

        # Draw new preview
        x1, y1 = self.current_rect_start
        x2, y2 = event.xdata, event.ydata

        width = abs(x2 - x1)
        height = abs(y2 - y1)
        left = min(x1, x2)
        bottom = min(y1, y2)

        self.preview_rect = patches.Rectangle(
            (left, bottom), width, height,
            linewidth=2, edgecolor='red', facecolor='white', alpha=0.3
        )
        self.ax.add_patch(self.preview_rect)
        self.fig.canvas.draw_idle()

    def on_release(self, event):
        """Handle mouse release - finalize rectangle."""
        if not self.drawing:
            return

        self.drawing = False

        if event.inaxes != self.ax:
            return

        x1, y1 = self.current_rect_start
        x2, y2 = event.xdata, event.ydata

        # Store rectangle (normalized coordinates)
        rect = (
            min(x1, x2),
            min(y1, y2),
            max(x1, x2),
            max(y1, y2)
        )
        self.rectangles.append(rect)

        # Remove preview
        if self.preview_rect:
            self.preview_rect.remove()
            self.preview_rect = None

        # Draw permanent rectangle
        width = rect[2] - rect[0]
        height = rect[3] - rect[1]

        permanent_rect = patches.Rectangle(
            (rect[0], rect[1]), width, height,
            linewidth=2, edgecolor='red', facecolor='white', alpha=0.5
        )
        self.ax.add_patch(permanent_rect)
        self.fig.canvas.draw()

        logger.info(f"Added rectangle: ({rect[0]:.0f}, {rect[1]:.0f}) to ({rect[2]:.0f}, {rect[3]:.0f})")
        logger.info(f"Total rectangles: {len(self.rectangles)}")

    def on_key(self, event):
        """Handle keyboard events."""
        if event.key == 'u':
            # Undo last rectangle
            if self.rectangles:
                self.rectangles.pop()
                logger.info(f"Undid last rectangle. Remaining: {len(self.rectangles)}")
                self.redraw()

        elif event.key == 'r':
            # Reset all rectangles
            self.rectangles = []
            logger.info("Reset all rectangles")
            self.redraw()

    def redraw(self):
        """Redraw the image with all rectangles."""
        self.ax.clear()
        self.ax.imshow(self.image)
        self.ax.set_title(
            "INSTRUCTIONS:\n"
            "1. Click and drag to draw rectangles over text\n"
            "2. Press 'u' to undo last rectangle\n"
            "3. Press 'r' to reset all rectangles\n"
            "4. Close window when done to save",
            fontsize=10,
            pad=15
        )

        # Redraw all rectangles
        for rect in self.rectangles:
            width = rect[2] - rect[0]
            height = rect[3] - rect[1]
            patch = patches.Rectangle(
                (rect[0], rect[1]), width, height,
                linewidth=2, edgecolor='red', facecolor='white', alpha=0.5
            )
            self.ax.add_patch(patch)

        self.fig.canvas.draw()

    def show(self):
        """Show the interactive window."""
        plt.show()

    def create_mask(self) -> np.ndarray:
        """
        Create a binary mask from the drawn rectangles.

        Returns:
            Binary mask (255 = remove text, 0 = keep)
        """
        mask = Image.new('L', (self.width, self.height), 0)
        draw = ImageDraw.Draw(mask)

        for rect in self.rectangles:
            draw.rectangle(rect, fill=255)

        logger.info(f"Created mask with {len(self.rectangles)} rectangles")
        return np.array(mask)

    def apply_mask_and_save(self, output_path: str, mask_output_path: str = None):
        """
        Apply the mask and save the result.

        Args:
            output_path: Where to save the final image
            mask_output_path: Where to save the mask (optional)
        """
        if not self.rectangles:
            logger.warning("No rectangles drawn - nothing to remove!")
            return

        # Create mask
        mask = self.create_mask()

        if mask_output_path:
            mask_img = Image.fromarray(mask)
            mask_img.save(mask_output_path)
            logger.info(f"Mask saved to: {mask_output_path}")

        # Apply mask (simple white fill)
        img_array = np.array(self.image)

        for channel in range(3):  # RGB channels
            img_array[:, :, channel][mask == 255] = 255

        # Save result
        result = Image.fromarray(img_array.astype('uint8'))
        result.save(output_path)
        logger.info(f"Result saved to: {output_path}")

        return result


def main():
    """Run the interactive text remover."""

    input_path = "outputs/JesusAndTrump/JesusAndTrump.jpg"
    output_path = "outputs/JesusAndTrump/JesusAndTrump_no_text_interactive.png"
    mask_path = "outputs/JesusAndTrump/mask_interactive.png"

    logger.info("=" * 70)
    logger.info("Interactive Text Removal Tool")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Loading image...")

    # Create interactive tool
    tool = InteractiveTextRemover(input_path)

    # Show interactive window
    logger.info("Opening interactive window...")
    logger.info("Draw rectangles over text areas you want to remove")
    logger.info("Close the window when done")
    logger.info("")

    tool.show()

    # After window closes, save result
    logger.info("")
    logger.info("Window closed. Saving result...")
    tool.apply_mask_and_save(output_path, mask_path)

    logger.info("")
    logger.info("=" * 70)
    logger.info("✓ Done!")
    logger.info(f"Output: {output_path}")
    logger.info(f"Mask: {mask_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
