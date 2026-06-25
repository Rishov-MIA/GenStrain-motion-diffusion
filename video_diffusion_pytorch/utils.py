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
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)  # image convention: row 0 at top

            # Plot the displacement vectors. Negate displacement_y to preserve
            # direction after flipping the y-axis to image convention.
            ax.quiver(x, y, displacement_x[frame_idx], -displacement_y[frame_idx], color="y", scale=scale, units="xy")

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
                ax.set_xlim(0, w)
                ax.set_ylim(h, 0)  # image convention: row 0 at top

            # Plot the first displacement field. Negate displacement_y to
            # preserve direction after flipping the y-axis to image convention.
            axs[0].quiver(
                x, y, displacement_x1[frame_idx], -displacement_y1[frame_idx], color="y", scale=scale, units="xy"
            )
            axs[0].set_title(titles[0], color="white")

            # Plot the second displacement field
            axs[1].quiver(
                x, y, displacement_x2[frame_idx], -displacement_y2[frame_idx], color="y", scale=scale, units="xy"
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
                ax.set_xlim(0, w)
                ax.set_ylim(h, 0)  # image convention: row 0 at top

            # Negate displacement_y to preserve direction after flipping the
            # y-axis to image convention.
            axs[0].quiver(x, y, dx1[t], -dy1[t], color="y", scale=scale, units="xy")
            axs[1].quiver(x, y, dx2[t], -dy2[t], color="y", scale=scale, units="xy")
            axs[2].quiver(x, y, dx3[t], -dy3[t], color="y", scale=scale, units="xy")

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
            ax2.set_xlim(0, width)
            ax2.set_ylim(height, 0)  # image convention: row 0 at top, matches the mask panel
            # Negate displacement_y to preserve direction after flipping the
            # y-axis to image convention.
            ax2.quiver(
                x, y, displacement_x[frame_idx], -displacement_y[frame_idx], color=quiver_color, scale=scale, units="xy"
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


def generate_displacement_gridplot_gif(
    transformation_field, output_path, fps=20, dpi=300, line_color="r", line_width=0.5, line_alpha=0.5, figsize=(2, 2)
):
    transformation_field_numpy = (
        transformation_field.cpu().numpy() if isinstance(transformation_field, torch.Tensor) else transformation_field
    )
    # Extract dimensions
    _, channels, frames, H, W = transformation_field_numpy.shape
    assert channels == 2, "Expected 2 channels for x and y displacement"

    # Create output directory if needed
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create base grid points
    x = np.linspace(0, W - 1, W)
    y = np.linspace(0, H - 1, H)
    X, Y = np.meshgrid(x, y)

    # Set up the figure
    fig = plt.figure(figsize=figsize)
    plt.xticks([])
    plt.yticks([])
    plt.axis("off")
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)

    def update(frame):
        plt.clf()
        plt.xticks([])
        plt.yticks([])
        plt.axis("off")
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.margins(0, 0)

        # Get displacement for current frame
        dx = transformation_field_numpy[0, 0, frame]
        dy = transformation_field_numpy[0, 1, frame]

        # Apply displacement
        X_d = X + dx
        Y_d = Y + dy

        # Plot deformed grid
        for i in range(W):
            plt.plot(X_d[:, i], Y_d[:, i], f"{line_color}-", linewidth=line_width, alpha=line_alpha)
        for i in range(H):
            plt.plot(X_d[i, :], Y_d[i, :], f"{line_color}-", linewidth=line_width, alpha=line_alpha)

        plt.axis("equal")
        plt.gca().invert_yaxis()

        return (fig,)

    # Create animation
    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=True)

    # Save as GIF
    writer = PillowWriter(fps=fps)
    anim.save(str(output_path), writer=writer, dpi=dpi)
    plt.close()


def generate_displacement_gridplot_gifs(milestone, displacement_field, output_dir, **kwargs):
    """
    Load displacement field from .npy file and create multiple GIFs.

    Parameters:
        displacement_field (str): displacement field (shape: [num_samples, 2, F, H, W])
        output_dir (str): Directory where output GIFs will be saved
        **kwargs: Additional arguments for generate_displacement_gridplot_gif
    """
    num_samples = displacement_field.shape[0]

    # Create output directory if needed
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_samples):
        output_path = output_dir / f"milestone_{milestone}_displacement_{i}.gif"
        generate_displacement_gridplot_gif(displacement_field[i], output_path, **kwargs)


def save_grayscale_videos(milestone, video_tensor, filenames, output_dir="./gt_videos", fps=10):
    """
    Saves grayscale videos as separate .gif files.

    Args:
        video_tensor (torch.Tensor or np.ndarray): Shape [num_samples, 1, F, H, W]
        output_dir (str): Directory to save videos
    """
    if isinstance(video_tensor, torch.Tensor):
        video_tensor = video_tensor.cpu().numpy()

    os.makedirs(output_dir, exist_ok=True)

    num_samples, _, num_frames, height, width = video_tensor.shape

    # Convert FPS to duration in milliseconds per frame
    duration = math.ceil(1000 / fps)

    # Process each sample
    for sample_idx in range(num_samples):
        frames = [Image.fromarray(video_tensor[sample_idx, 0, i]) for i in range(num_frames)]

        # Define GIF filename
        gif_filename = os.path.join(output_dir, f"milestone_{milestone}_gt_{filenames[sample_idx]}.gif")

        # Save as GIF
        frames[0].save(gif_filename, save_all=True, append_images=frames[1:], duration=duration, loop=0)
