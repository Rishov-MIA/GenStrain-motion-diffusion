import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
import torch
import os
import imageio.v2 as imageio
from io import BytesIO
from PIL import Image
from tqdm import tqdm
import math



def generate_displacement_quiver_gif(transformation_field, output_path, scale=1, fps=10):
    """
    Generates a GIF showing the displacement vectors across all frames without saving intermediate images.

    Parameters:
    - transformation_field: displacement field (shape: (1, 2, 20, 64, 64)).
    - output_gif: Name of the output GIF file.
    - scale: Scaling factor for the quiver plot.
    - fps: Frames per second for the GIF.
    """
    data = (
        transformation_field.cpu().numpy() if isinstance(transformation_field, torch.Tensor) else transformation_field
    )

    # Extract displacement components
    displacement_x = data[0, 0]  # Shape: (20, 64, 64)
    displacement_y = data[0, 1]  # Shape: (20, 64, 64)

    num_frames, h, w = displacement_x.shape

    # Create mesh grid for quiver plot
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    # Open an in-memory GIF writer with loop forever setting
    with imageio.get_writer(output_path, mode="I", fps=fps, loop=0) as writer:
        for frame_idx in range(num_frames):
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.set_facecolor("black")  # Set background to black
            ax.set_aspect("equal")

            # Plot the displacement vectors
            ax.quiver(x, y, displacement_x[frame_idx], displacement_y[frame_idx], color="y", scale=scale, units="xy")

            # Add frame number text like "Frame: 5 / 20"
            ax.text(
                1,
                3,
                f"Frame: {frame_idx + 1} / {num_frames}",
                color="white",
                fontsize=8,
                fontweight="bold",
                bbox=dict(facecolor="black", edgecolor="white", boxstyle="round,pad=0.3"),
            )

            # Save frame to an in-memory buffer
            buf = BytesIO()
            plt.axis("off")
            plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, facecolor="black")
            plt.close(fig)

            # Convert buffer to image
            buf.seek(0)
            frame = imageio.imread(buf)
            writer.append_data(frame)


def generate_displacement_quiver_gifs(
    milestone, displacement_field, filenames, output_dir="./sampled_quiver_videos_gifs", **kwargs
):
    """
    Load displacement field from .npy file and create multiple GIFs.

    Parameters:
        displacement_field (str): displacement field (shape: [num_samples, 1, 2, F, H, W])
        output_dir (str): Directory where output GIFs will be saved
        **kwargs: Additional arguments for generate_displacement_gridplot_gif
    """
    # Load displacement field
    num_samples = displacement_field.shape[0]

    # Create output directory if needed
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_samples):
        output_path = output_dir / f"milestone_{milestone}_disp_{filenames[i]}.gif"
        generate_displacement_quiver_gif(displacement_field[i], output_path, **kwargs)


def generate_displacement_quiver_gifs_with_gt(
    milestone, filenames, pred_disp, gt_disp, output_dir="./sampled_quiver_videos_gifs", **kwargs
):
    """
    Load displacement field from .npy file and create multiple GIFs.

    Parameters:
        pred_disp (str): displacement field (shape: [num_samples, 1, 2, F, H, W])
        gt_disp (str): displacement field (shape: [num_samples, 1, 2, F, H, W])
        output_dir (str): Directory where output GIFs will be saved
        **kwargs: Additional arguments for generate_displacement_gridplot_gif
    """
    # Load displacement field
    num_samples = pred_disp.shape[0]

    # Create output directory if needed
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_samples):
        output_path = output_dir / f"milestone_{milestone}_disp_{filenames[i]}.gif"
        generate_displacement_quiver_gif_comparison(pred_disp[i], gt_disp[i], output_path, **kwargs)


def generate_displacement_quiver_gif_comparison(
    transformation_field1, transformation_field2, output_path, scale=1, fps=10, titles=["Predicted", "Ground Truth"]
):
    """
    Generates a GIF showing the displacement vectors for two fields side by side across all frames.

    Parameters:
    - transformation_field1: First displacement field (shape: (1, 2, 20, 64, 64)).
    - transformation_field2: Second displacement field (shape: (1, 2, 20, 64, 64)).
    - output_path: Path for the output GIF file.
    - scale: Scaling factor for the quiver plots.
    - fps: Frames per second for the GIF.
    - titles: List of titles for the two plots, defaults to ["Field 1", "Field 2"].
    """
    # Process field 1
    data1 = (
        transformation_field1.cpu().numpy()
        if isinstance(transformation_field1, torch.Tensor)
        else transformation_field1
    )
    displacement_x1 = data1[0, 0]  # Shape: (20, 64, 64)
    displacement_y1 = data1[0, 1]  # Shape: (20, 64, 64)

    # Process field 2
    data2 = (
        transformation_field2.cpu().numpy()
        if isinstance(transformation_field2, torch.Tensor)
        else transformation_field2
    )
    displacement_x2 = data2[0, 0]  # Shape: (20, 64, 64)
    displacement_y2 = data2[0, 1]  # Shape: (20, 64, 64)

    # Default titles if not provided
    if titles is None:
        titles = ["Field 1", "Field 2"]

    num_frames, h, w = displacement_x1.shape

    # Create mesh grid for quiver plot
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    # Open an in-memory GIF writer with loop forever setting
    with imageio.get_writer(output_path, mode="I", fps=fps, loop=0) as writer:
        for frame_idx in range(num_frames):
            # Create figure with two subplots side by side
            fig, axs = plt.subplots(1, 2, figsize=(8, 4))

            # Set background to black for both subplots
            for ax in axs:
                ax.set_facecolor("black")
                ax.set_aspect("equal")
                ax.axis("off")

            # Plot the first displacement field
            axs[0].quiver(
                x, y, displacement_x1[frame_idx], displacement_y1[frame_idx], color="y", scale=scale, units="xy"
            )
            axs[0].set_title(titles[0], color="white")

            # Plot the second displacement field
            axs[1].quiver(
                x, y, displacement_x2[frame_idx], displacement_y2[frame_idx], color="y", scale=scale, units="xy"
            )
            axs[1].set_title(titles[1], color="white")

            # Add frame number text
            axs[0].text(
                1,
                3,
                f"Frame: {frame_idx + 1} / {num_frames}",
                color="white",
                fontsize=8,
                fontweight="bold",
                bbox=dict(facecolor="black", edgecolor="white", boxstyle="round,pad=0.3"),
            )

            # Save frame to an in-memory buffer
            buf = BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1, facecolor="black")
            plt.close(fig)

            # Convert buffer to image
            buf.seek(0)
            frame = imageio.imread(buf)
            writer.append_data(frame)

    print(f"Comparison GIF saved as {output_path}")


def generate_displacement_quiver_gif_predicted_reconstructed_gt(
    transformation_field1,
    transformation_field2,
    transformation_field3,
    output_path,
    scale=1,
    fps=10,
    titles=["Predicted", "Reconstructed", "Ground Truth"],
):
    """
    Generates a GIF showing the displacement vectors for three fields side by side across all frames.

    Parameters:
    - transformation_field1: First displacement field (shape: (1, 2, T, H, W)).
    - transformation_field2: Second displacement field (shape: (1, 2, T, H, W)).
    - transformation_field3: Third displacement field (shape: (1, 2, T, H, W)).
    - output_path: Path for the output GIF file.
    - scale: Scaling factor for the quiver plots.
    - fps: Frames per second for the GIF.
    - titles: List of titles for the three plots.
    """
    def to_numpy(data):
        return data.cpu().numpy() if isinstance(data, torch.Tensor) else data

    # Convert all inputs to numpy
    data1, data2, data3 = map(to_numpy, [transformation_field1, transformation_field2, transformation_field3])

    # Extract displacements
    dx1, dy1 = data1[0, 0], data1[0, 1]
    dx2, dy2 = data2[0, 0], data2[0, 1]
    dx3, dy3 = data3[0, 0], data3[0, 1]

    num_frames, h, w = dx1.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    with imageio.get_writer(output_path, mode="I", fps=fps, loop=0) as writer:
        for t in range(num_frames):
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            for i, ax in enumerate(axs):
                ax.set_facecolor("black")
                ax.set_aspect("equal")
                ax.axis("off")

            axs[0].quiver(x, y, dx1[t], dy1[t], color="y", scale=scale, units="xy")
            axs[1].quiver(x, y, dx2[t], dy2[t], color="y", scale=scale, units="xy")
            axs[2].quiver(x, y, dx3[t], dy3[t], color="y", scale=scale, units="xy")

            for i in range(3):
                axs[i].set_title(titles[i], color="white")

            axs[0].text(
                1,
                3,
                f"Frame: {t + 1} / {num_frames}",
                color="white",
                fontsize=8,
                fontweight="bold",
                bbox=dict(facecolor="black", edgecolor="white", boxstyle="round,pad=0.3"),
            )

            buf = BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1, facecolor="black")
            plt.close(fig)

            buf.seek(0)
            frame = imageio.imread(buf)
            writer.append_data(frame)

    print(f"Comparison GIF saved as {output_path}")


def generate_mask_disp_side_by_side_gif(
    mask_video,
    transformation_field,
    output_path,
    scale=1,
    fps=10,
    figsize=(10, 5),
    video_title="Mask Video",
    field_title="Displacement Field",
    video_cmap="gray",
    quiver_color="yellow",
):
    """
    Advanced version with more customization options for the side-by-side GIF.

    Parameters:
    - mask_video: Video tensor (shape: [1, F, H, W])
    - transformation_field: Displacement field (shape: [1, 2, F, H, W])
    - output_path: Path to save the combined GIF
    - scale: Scaling factor for the quiver plot
    - fps: Frames per second for the GIF
    - figsize: Figure size for the plot
    - video_title: Title for the video subplot
    - field_title: Title for the displacement field subplot
    - video_cmap: Colormap for the video
    - quiver_color: Color for the quiver arrows
    """
    # Convert to numpy if tensors
    video_data = mask_video.cpu().numpy() if isinstance(mask_video, torch.Tensor) else mask_video
    field_data = (
        transformation_field.cpu().numpy() if isinstance(transformation_field, torch.Tensor) else transformation_field
    )

    # Extract displacement components
    displacement_x = field_data[0, 0]  # Shape: (F, H, W)
    displacement_y = field_data[0, 1]  # Shape: (F, H, W)

    # Get dimensions
    _, num_frames, height, width = video_data.shape

    # Create mesh grid for quiver plot
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # Open an in-memory GIF writer
    with imageio.get_writer(output_path, mode="I", fps=fps, loop=0) as writer:
        for frame_idx in range(num_frames):
            # Create a figure with two subplots side by side with black background
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, facecolor="black")

            # Plot 1: Grayscale video
            frame = video_data[0, frame_idx]
            ax1.imshow(frame, cmap=video_cmap)
            ax1.set_title(video_title)
            # ax1.axis('off')

            # Plot 2: Displacement vectors
            ax2.set_facecolor("black")
            ax2.set_aspect("equal")
            ax2.quiver(
                x, y, displacement_x[frame_idx], displacement_y[frame_idx], color=quiver_color, scale=scale, units="xy"
            )
            ax2.set_title(field_title)
            # ax2.axis('off')

            # Add frame counter with white text
            fig.suptitle(f"Frame: {frame_idx + 1} / {num_frames}", fontsize=12, color="white")

            # Tight layout
            plt.tight_layout()

            # Save frame to an in-memory buffer
            buf = BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
            plt.close(fig)

            # Convert buffer to image
            buf.seek(0)
            frame = imageio.imread(buf)
            writer.append_data(frame)

    print(f"side-by-side GIF saved as {output_path}")
