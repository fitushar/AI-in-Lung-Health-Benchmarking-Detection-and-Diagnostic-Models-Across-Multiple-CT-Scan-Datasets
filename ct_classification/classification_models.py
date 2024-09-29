# Import necessary libraries and modules
import monai
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import unet3d
from monai.networks.nets import resnet10, resnet18, resnet34, resnet50, resnet101, resnet152, resnet200

# Define a custom ResNet3D classifier model
class ResNet3DClassifier(nn.Module):
    def __init__(self, version='resnet50', n_input_channels=1, num_classes=2):
        super().__init__()
        # Dictionary to map version strings to MONAI ResNet implementations
        resnet_versions = {
            'resnet10': resnet10,
            'resnet18': resnet18,
            'resnet34': resnet34,
            'resnet50': resnet50,
            'resnet101': resnet101,
            'resnet152': resnet152,
            'resnet200': resnet200
        }
        # Initialize the selected ResNet model with 3D support
        resnet_model = resnet_versions[version](spatial_dims=3, n_input_channels=n_input_channels)
        # Extract all layers except the last pooling and fully connected layers
        self.encoder = nn.Sequential(*list(resnet_model.children())[:-2])

        # Define a new classification head
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))  # Global average pooling to reduce feature size
        # Determine the number of input channels for the final fully connected layer
        final_in_channels = {
            'resnet10': 512,
            'resnet18': 512,
            'resnet34': 512,
            'resnet50': 2048,
            'resnet101': 2048,
            'resnet152': 2048,
            'resnet200': 2048
        }[version]
        # Fully connected layer for classification
        self.fc = nn.Linear(final_in_channels, num_classes)

    def forward(self, x):
        # Pass input through the ResNet encoder
        x = self.encoder(x)
        # Apply global average pooling
        x = self.avgpool(x)
        # Flatten the output for the fully connected layer
        x = torch.flatten(x, 1)
        # Pass through the classification head
        x = self.fc(x)
        return x

# Define a 3D Target Network for classification
class TargetNet(nn.Module):
    def __init__(self, base_model, n_class):
        super(TargetNet, self).__init__()
        self.base_model = base_model
        # Define fully connected layers for further classification
        self.dense_1 = nn.Linear(512, 1024, bias=True)  # Intermediate dense layer
        self.dense_2 = nn.Linear(1024, n_class, bias=True)  # Output layer

    def forward(self, x):
        # Get base model features
        self.base_model(x)
        self.base_out = self.base_model.out512
        # Global average pooling for 3D input
        self.out_glb_avg_pool = F.avg_pool3d(self.base_out, kernel_size=self.base_out.size()[2:]).view(self.base_out.size()[0], -1)
        # Pass through dense layers with ReLU activation
        self.linear_out = self.dense_1(self.out_glb_avg_pool)
        final_out = self.dense_2(F.relu(self.linear_out))
        return final_out

# Load the Model Genesis pre-trained UNet3D and create a classifier model
def Model_Genesis_Classifier(n_class):
    base_model = unet3d.UNet3D()  # Initialize the base UNet3D model
    # Load pre-trained weights from the Model Genesis repository
    weight_dir = "Genesis_Chest_CT.pt"
    checkpoint = torch.load(weight_dir)
    state_dict = checkpoint['state_dict']
    # Remove any unwanted module prefixes in the checkpoint keys
    unParalled_state_dict = {}
    for key in state_dict.keys():
        unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
    base_model.load_state_dict(unParalled_state_dict)  # Load the pre-trained weights
    # Create the final TargetNet with the pre-trained UNet3D as the base
    target_model = TargetNet(base_model, n_class)
    return target_model

# Function to get the classification model based on input parameters
def get_classification_model(Model_name, spatial_dims, n_input_channels, num_classes, device):
    # Select the appropriate MONAI model based on the Model_name parameter
    if Model_name == 'resnet10':
        model = monai.networks.nets.resnet10(pretrained=False, spatial_dims=spatial_dims, n_input_channels=n_input_channels, num_classes=num_classes).to(device)
    elif Model_name == 'resnet18':
        model = monai.networks.nets.resnet18(pretrained=False, spatial_dims=spatial_dims, n_input_channels=n_input_channels, num_classes=num_classes).to(device)
    elif Model_name == 'resnet34':
        model = monai.networks.nets.resnet34(pretrained=False, spatial_dims=spatial_dims, n_input_channels=n_input_channels, num_classes=num_classes).to(device)
    elif Model_name == 'resnet50':
        model = monai.networks.nets.resnet50(pretrained=False, spatial_dims=spatial_dims, n_input_channels=n_input_channels, num_classes=num_classes).to(device)
    elif Model_name == 'resnet200':
        model = monai.networks.nets.resnet200(pretrained=False, spatial_dims=spatial_dims, n_input_channels=n_input_channels, num_classes=num_classes).to(device)
    elif Model_name == 'DenseNet121':
        model = monai.networks.nets.DenseNet121(pretrained=False, spatial_dims=spatial_dims, in_channels=n_input_channels, out_channels=num_classes).to(device)
    elif Model_name == 'Model_Genesis':
        model = Model_Genesis_Classifier(n_class=num_classes)  # Use the custom UNet3D classifier
        model = model.to(device)

    elif Model_name == 'resnet50_MedicalNet3D':
        model = monai.networks.nets.resnet50(pretrained=False, spatial_dims=spatial_dims, n_input_channels=n_input_channels, num_classes=num_classes)
        # Load MedicalNet 3D pre-trained weights
        model_dict = model.state_dict()
        weights_dict = torch.load('resnet_50_23dataset.pth')
        weights_dict = {k.replace('module.', ''): v for k, v in weights_dict['state_dict'].items()}
        model_dict.update(weights_dict)
        model.load_state_dict(model_dict)
        model = model.to(device)



    return model
