import torch.nn as nn
import torch


class PICO(nn.Module):
    def __init__(self, num_classes = 8):
        super().__init__()
        #input is 3x96x96
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1) #32x96x96
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1) #32x96x96
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.relu = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, padding=1, stride=2) #128x48x48
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1) #128x48x48
        self.bn2 = nn.BatchNorm2d(num_features=128) #ReLU

        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2) #128x24x24
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, padding=1)#512x24x24
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)#512x24x24
        self.bn3 = nn.BatchNorm2d(num_features=512) #ReLU

        #channel attention
        self.global_ap = nn.AdaptiveAvgPool2d((1,1)) #we want 1x1 output per input map, 512x1x1
        self.ca_conv1 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1) #128x1x1 -> ReLU
        self.ca_conv2 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=1) #512x1x1
        self.ca_sig = nn.Sigmoid()

        #inception style block (5x5 and 7x7)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2) #512x24x24 -> 512x12x12
        self.in_conv1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=5, padding=2) #512x12x12
        self.in_conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=7, padding=3) #512x12x12
        #conatenate them -> 1024x12x12

        self.conv7 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1) #512x12x12
        self.bn4 = nn.BatchNorm2d(num_features=512) #ReLU 

        #FC Layer
        self.fc = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self , x): #habe ignored the batch dimension in notations, except in the final output
        layer1_out = self.relu(self.bn1(self.conv2(self.conv1(x))))#32x96x96
        layer2_out = self.relu(self.bn2(self.conv4(self.conv3(layer1_out))))#128x48x48

        pool1_out = self.max_pool1(layer2_out)#128x24x24

        layer3_out = self.relu(self.bn3(self.conv6(self.conv5(pool1_out))))#512x24x24

        #channel attention
        weights = self.ca_sig(self.ca_conv2(self.relu(self.ca_conv1(self.global_ap(layer3_out)))))
        ca_out = layer3_out * weights #512x24x24

        pool2_out = self.max_pool2(ca_out) #512x12x12

        #inception
        in_conv1_out = self.in_conv1(pool2_out)#512x12x12
        in_conv2_out = self.in_conv2(pool2_out)#512x12x12
        in_out = torch.concatenate([in_conv1_out, in_conv2_out], dim = 1) #1024x12x12

        layer4_out = self.relu(self.bn4(self.conv7(in_out)))#512x12x12 used for channel mixing after inception style operation
        final_gap_out = self.global_ap(layer4_out) #512x1x1
        out = self.fc(torch.flatten(final_gap_out, start_dim = 1)) #(batch_size , num_classes)

        return out
    









