import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
import numpy as np
import torchvision.transforms as T


# Define the UNet model
class UNET(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNET, self).__init__()
        self.conv1 = self.contract_block(in_channels, 32, 7, 3)
        self.conv2 = self.contract_block(32, 64, 3, 1)
        self.conv3 = self.contract_block(64, 128, 3, 1)
        self.upconv3 = self.expand_block(128, 64, 3, 1)
        self.upconv2 = self.expand_block(64 * 2, 32, 3, 1)
        self.upconv1 = self.expand_block(32 * 2, out_channels, 3, 1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        upconv3 = self.upconv3(conv3)
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))
        return upconv1

    def contract_block(self, in_channels, out_channels, kernel_size, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

    def expand_block(self, in_channels, out_channels, kernel_size, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

class MNISTDDDataset(Dataset):
    def __init__(self, images, masks):
        # Ensure images have the correct shape [N, 64, 64, 3] for HWC format
        self.images = images.reshape(-1, 64, 64, 3)
        self.masks = masks.reshape(-1, 64, 64)
        self.transform = T.Compose([
            T.ToTensor(),  # Converts HWC to CHW and normalizes to [0, 1]
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Ensure input to transform is in HWC format
        image = self.transform(self.images[idx].astype(np.float32)) 
        mask = torch.tensor(self.masks[idx], dtype=torch.long)  
        return image, mask


# Training function
def train_unet(train_images, train_masks, save_path='unet_checkpoint.pth'):
    dataset = MNISTDDDataset(train_images, train_masks)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNET(3, 11).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target) 
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved at {save_path}")


def detect_and_segment(test_images):
    import torchvision.ops
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNET(3, 11).to(device)
    model.load_state_dict(torch.load('unet_checkpoint.pth', map_location=device))
    model.eval()

    N = test_images.shape[0]  # Number of test images
    pred_class = np.zeros((N, 2), dtype=np.int32)  # Predictions for 2 classes
    pred_bboxes = np.zeros((N, 2, 4), dtype=np.float64)  # Bounding boxes for each class
    pred_seg = np.zeros((N, 4096), dtype=np.int32)  # Segmentation mask

    with torch.no_grad():
        for i in range(N):
            # Prepare image for model input
            img = test_images[i].reshape(64, 64, 3).transpose(2, 0, 1)  # Reshape and transpose
            img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)  # Normalize and add batch dimension

            # Forward pass through the model
            output = model(img)  # [1, 11, 64, 64]
            output = torch.argmax(output.squeeze(0), dim=0).cpu().numpy()  # [64, 64]

            # Store predicted segmentation mask
            pred_seg[i] = output.flatten()

            # Create binary masks for each class (0 to 9)
            pred_mask = [(output == cls).astype(np.uint8) for cls in range(10)]

            # Extract bounding boxes for each mask
            boxes = []
            for cls, mask in enumerate(pred_mask):
                 # Process only valid masks
                if mask.sum() > 0: 
                    mask_tensor = torch.tensor(mask).unsqueeze(0)
                    bbox = torchvision.ops.masks_to_boxes(mask_tensor)[0].cpu().numpy()
                    # Save class and its bounding box
                    boxes.append((cls, bbox))  
                else:
                    boxes.append((cls, [0, 0, 0, 0]))

            # Identify unique classes in the predicted mask
            unique_classes, counts = np.unique(output, return_counts=True)
            # Exclude background (class 10)
            unique_classes = unique_classes[unique_classes < 10] 

            # Ensure exactly 2 predicted classes
            if len(unique_classes) > 2:
                # Sort by frequency
                sorted_indices = np.argsort(-counts[:len(unique_classes)])  
                # Pick top 2 classes
                unique_classes = unique_classes[sorted_indices[:2]]  
            while len(unique_classes) < 2:
                # Pad with 0 if fewer than 2 classes
                unique_classes = np.append(unique_classes, 0)  
                # Ensure ascending order
            unique_classes = np.sort(unique_classes)  
            pred_class[i] = unique_classes

            # Use precomputed `boxes` for the 2 selected classes
            bboxes = []
            for cls in unique_classes:
                bbox = [b[1] for b in boxes if b[0] == cls]  # Find bounding box for the class
                if bbox:
                    bboxes.append(bbox[0])
                else:
                    bboxes.append([0, 0, 0, 0])  # Default if no bounding box found

            pred_bboxes[i] = np.array(bboxes, dtype=np.float64)

    return pred_class, pred_bboxes, pred_seg
