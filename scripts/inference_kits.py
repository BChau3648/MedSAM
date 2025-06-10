import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from segment_anything import sam_model_registry
from skimage import io, transform
import torch.nn.functional as F
import nibabel as nib
import argparse
from monai.metrics import compute_hausdorff_distance
from tqdm import tqdm

@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :] # (B, 1, 4)
    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed, # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
        multimask_output=False,
        )
    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)
    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg

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
images = sorted([f for f in os.listdir(args.images_path) if f.endswith('.nii.gz')])
labels = sorted([f for f in os.listdir(args.labels_path) if f.endswith('.nii.gz')])

# Writing to scores file
with open(os.path.join(args.output_path, 'scores.txt'), 'w') as f:
    f.write(f"File Dice HausdorffDistance\n")

# Iterating over each case
for i in tqdm(range(len(images))):
    kid_img = nib.load(os.path.join(args.images_path, images[i])).get_fdata()
    kid_label_nib = nib.load(os.path.join(args.labels_path, labels[i]))
    kid_label = kid_label_nib.get_fdata()

    # Mask label to only 0=background and 1=lesion
    kid_label[kid_label != 2] = 0
    kid_label[kid_label == 2] = 1

    # Rearrange img and label into H, W, C
    kid_img = kid_img.transpose((1, 2, 0))
    kid_label = kid_label.transpose((1, 2, 0))

    # Iterate through each slice of the label and find the bounding box
    bounding_boxes = []
    for j in range(kid_label.shape[2]):
        slice_mask = kid_label[:, :, j]
        bbox = get_bounding_box(slice_mask)
        bounding_boxes.append(bbox)
    
    H, W, C = kid_img.shape
    # Clipping
    img_1024 = np.clip(kid_img, -160, 240)
    # Min-Max normalization
    img_1024 = (img_1024 - img_1024.min()) / np.clip(
        img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
    )  # normalize to [0, 1], (H, W, 3)
    # Resize
    img_1024 = transform.resize(img_1024, (1024, 1024, C), order=3, preserve_range=True, anti_aliasing=True)

    # Getting channels where there are lesions
    lesion_channel_idx = np.unique(np.where(kid_label == 1)[2]).astype(int)

    # Getting predicted labels
    kid_pred = np.zeros(kid_label.shape)
    for c in lesion_channel_idx:
        img_slice = img_1024[:, :, c]
        slice_tensor = torch.tensor(img_slice).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(device)
        # Bounding box
        box_np = get_bounding_box(kid_label[:,:,c])
        box_np = np.expand_dims(box_np, axis=0)
        box_1024 = box_np / np.array([W, H, W, H]) * 1024
        with torch.no_grad():
            image_embedding = medsam_model.image_encoder(slice_tensor.float()) # (1, 256, 64, 64)
        medsam_seg = medsam_inference(medsam_model, image_embedding, box_1024, H, W)
        kid_pred[:,:,c] = medsam_seg

    np.save(os.path.join(args.output_path, labels[i].split('.')[0]), kid_pred)

    # Evaluating dice score
    intersection = np.sum(kid_label * kid_pred)
    union = np.sum(kid_label) + np.sum(kid_pred)
    eps = 1e-8
    dice = (2.0 * intersection + eps) / (union + eps)

    # Evaluating hausdorff distance
    # Calculating spacing of resampled scan
    og_shape = kid_label.shape
    og_spacing = kid_label_nib.header.get_zooms()
    # Changing og_spacing because rearranged axes earlier
    og_spacing = og_spacing[1:] + (og_spacing[0],)
    new_shape = (1024, 1024, og_shape[2])
    new_spacing = tuple(np.array(og_spacing) * np.array(og_shape) / np.array(new_shape))
    # one hot encode 3D tensor
    kid_label_onehot = np.stack([(kid_label == 0).astype(int), (kid_label == 1).astype(int)], axis=0)
    kid_label_onehot = np.expand_dims(kid_label_onehot, axis=0)

    kid_pred_onehot = np.stack([(kid_pred == 0).astype(int), (kid_pred == 1).astype(int)], axis=0)
    kid_pred_onehot = np.expand_dims(kid_pred_onehot, axis=0)
    hd = compute_hausdorff_distance(kid_pred_onehot, kid_label_onehot, spacing=new_spacing).item()

    with open(os.path.join(args.output_path, 'scores.txt'), 'a') as f:
        f.write(f"{labels[i]} {dice} {hd}\n")