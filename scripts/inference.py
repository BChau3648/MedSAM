import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from segment_anything import sam_model_registry
from skimage import io, transform
import torch.nn.functional as F
import nibabel as nib
import argparse

def get_bounding_box(mask_slice):
    """
    Calculates the bounding box coordinates for a binary mask slice.

    Args:
        mask_slice (np.ndarray): A 2D numpy array representing a single slice
                                 of the binary mask (0 for background, 1 for lesion).

    Returns:
        np.array: An array containing the bounding box coordinates (x_min, y_min, x_max, y_max),
               or None if no lesion is present in the slice.
    """
    rows = np.any(mask_slice, axis=1)
    cols = np.any(mask_slice, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]] if np.any(rows) else (None, None)
    x_min, x_max = np.where(cols)[0][[0, -1]] if np.any(cols) else (None, None)

    if x_min is not None and y_min is not None:
        return np.array([x_min, y_min, x_max, y_max])
    else:
        return None

parser = argparse.ArgumentParser(description='', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--images_path', help='Input path to images')
parser.add_argument('--labels_path', help='Input path to labels')
parser.add_argument('--output_path', help='Output path of predicted labels')
parser.add_argument('--model_path', help='Path to checkpointed MEDSAM model')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model and image
MedSAM_CKPT_PATH = args.model_path
medsam_model = sam_model_registry['vit_b'](checkpoint=MedSAM_CKPT_PATH)
medsam_model = medsam_model.to(device)
medsam_model.eval()

# Getting images and labels
images = [f for f in os.listdir(args.images_path) if f.endswith('.nii.gz')].sort()
labels = [f for f in os.listdir(args.labels_path) if f.endswith('.nii.gz')].sort()

# Iterating over each case
for i in range(len(images)):
    image = nib.load(os.path.join(args.images_path, images[i])).get_fdata()
    label = nib.load(os.path.join(args.labels_path, labels[i])).get_fdata()

    # Iterate through each slice of the label and find the bounding box
    bounding_boxes = []
    for j in range(label.shape[2]):
        label_slice = label[:, :, j]
        bbox = get_bounding_box(label_slice)
        bounding_boxes.append(bbox)
    
    H, W, C = image.shape
    # Clipping
    img_1024 = np.clip(image, -160, 240)
    # Min-Max normalization
    img_1024 = (img_1024 - img_1024.min()) / np.clip(
        img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
    )  # normalize to [0, 1], (H, W, 3)
    # Resize
    img_1024 = transform.resize(img_1024, (1024, 1024, C), order=3, preserve_range=True, anti_aliasing=True)

    # Getting channels where there are lesions
    lesion_channel_idx = np.unique(np.where(label == 1)[2]).astype(int)

    # Getting predicted labels
    pred = np.zeros(label.shape)
    for c in lesion_channel_idx:
        # Repeating 2D slices to get shape (1, 3, H, W)
        img_slice = img_1024[:, :, c]
        slice_tensor = torch.tensor(img_slice).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(device)

        # Converting bounding box to 1024x1024 size
        box_np = get_bounding_box(label[:,:,c])
        box_np = np.expand_dims(box_np, axis=0)
        box_1024 = box_np / np.array([W, H, W, H]) * 1024

    with torch.no_grad():
        image_embedding = medsam_model.image_encoder(slice_tensor.float()) # (1, 256, 64, 64)

    medsam_seg = medsam_inference(medsam_model, image_embedding, box_1024, H, W)
    pred[:,:,c] = medsam_seg

    np.save(os.path.join(args.output_path, labels[i]))

    # Evaluating dice score
    intersection = np.sum(label * pred)
    union = np.sum(label) + np.sum(pred)
    eps = 1e-8
    dice = (2.0 * intersection + eps) / (union + eps)

    with open(os.path.join(args.output_path, 'dice_scores.txt'), 'a') as f:
        f.write(f"{labels[i]} {dice}\n")