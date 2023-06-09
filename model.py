#model 3 99.45%
class Net(nn.Module):
    """
      Defines the architecture for a Convolutional Neural Network for image classification.
    """
    def __init__(self):
        super(Net, self).__init__()

        # Convolutional layer 1 - Takes in 1 input channel (as it's grayscale), outputs 10 channels, kernel size is 3 which means 3x3 filters are used
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        # Batch Normalization 1 - Stabilizes and accelerates the learning process
        self.bn1 = nn.BatchNorm2d(10)
        # Dropout layer 1 - Regularization method where input units are randomly dropped during training, which helps prevent overfitting
        self.do1 = nn.Dropout(0.10)

        # Convolutional layer 2 - Takes in 10 input channels, outputs 16 channels, with 3x3 filters
        self.conv2 = nn.Conv2d(10, 16, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(16)
        self.do2 = nn.Dropout(0.10)

        # Convolutional layer 3 - Takes in 16 input channels, outputs 32 channels, with 3x3 filters
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(32)
        self.do3 = nn.Dropout(0.10)

        # MaxPool layer - Applies a 2x2 max pooling over the input signal, effectively reducing the spatial dimensions by half (downsampling)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Convolutional layer 4 - A 1x1 convolution that acts as channel-wise fully connected layer, mixing the channels but keeping the spatial dimensions intact
        self.conv4 = nn.Conv2d(32, 10, kernel_size=1)
        self.bn4 = nn.BatchNorm2d(10)
        self.do4 = nn.Dropout(0.10)

        # Following the same pattern for layers 5, 6, 7
        self.conv5 = nn.Conv2d(10, 16, kernel_size=3)
        self.bn5 = nn.BatchNorm2d(16)
        self.do5 = nn.Dropout(0.10)

        self.conv6 = nn.Conv2d(16, 32, kernel_size=3)
        self.bn6 = nn.BatchNorm2d(32)
        self.do6 = nn.Dropout(0.10)

        # Another 1x1 convolution layer to reduce channel size
        self.conv7 = nn.Conv2d(32, 10, kernel_size=1)
        self.bn7 = nn.BatchNorm2d(10)
        self.do7 = nn.Dropout(0.10)

        # Final Convolutional layer - Applies a 7x7 filter to each input channel for global pooling, covering the entire size of the image, taking 10 input channels and outputting 10 channels
        self.conv8 = nn.Conv2d(10, 10, kernel_size=7)

    def forward(self, x):
        """
          Defines the forward pass for this model.
        """
      # Apply each layer with ReLU activation function, batch normalization, and dropout sequentially
      # Finally, pass through the 2nd max pooling layer, flatten the tensor, and apply log softmax for outputting probabilities

        x = F.relu(self.conv1(x))
        x = self.do1(self.bn1(x))

        x = F.relu(self.conv2(x))
        x = self.do2(self.bn2(x))

        x = F.relu(self.conv3(x))
        x = self.do3(self.bn3(x))

        x = self.pool1(x)

        x = F.relu(self.conv4(x))
        x = self.do4(self.bn4(x))

        x = F.relu(self.conv5(x))
        x = self.do5(self.bn5(x))

        x = F.relu(self.conv6(x))
        x = self.do6(self.bn6(x))

        x = F.relu(self.conv7(x))
        x = self.do7(self.bn7(x))

        x = F.relu(self.conv8(x))

        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.log_softmax(x, dim=1)
        
        return x

