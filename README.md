# PyTorch MNIST Classifier with Less Than 20k Parameters

This project builds a convolutional neural network (CNN) using PyTorch to classify images in the MNIST dataset. The specific challenge was to achieve a model accuracy of 99.4% under 20 epochs with less than 20k parameters.

## Model Architecture

The CNN is defined in the `model.py` file. The network architecture was carefully designed to minimize the number of parameters while maintaining high classification accuracy. 

Here's the architecture overview:

1. Two initial convolutional layers with 32 and 64 filters, each having a kernel size of 3x3.
2. A max pooling layer with a size of 2x2.
3. Additional convolutional layers with 128 and 256 filters, each with a kernel size of 3x3.
4. Another max pooling layer with a size of 2x2.
5. Final sequence of convolutional layers with 512, 1024, and 10 filters. All these layers use a kernel size of 3x3.

Each convolutional layer is followed by a ReLU activation function, and the output of the final layer is transformed with a log softmax function for class probabilities.

## Training and Evaluation

`utils.py` houses the functions necessary for model training and evaluation:

- `train` function iterates over the training dataset, updating model parameters according to computed gradients.
- `test` function evaluates the model on test data, yielding total test loss and accuracy.

Refer to `main.ipynb` for detailed training and evaluation process. The model is trained for 20 epochs, and performance metrics are printed after each epoch.

## Setup and Usage

1. Install the required dependencies, PyTorch and torchvision.
2. Execute `main.ipynb`. This trains and evaluates the model, printing results as it proceeds.

## Results

With this architecture and training regimen, the model achieved an accuracy of XX% on the test dataset within 20 epochs, using fewer than 20k parameters.

Sample Output:
Epoch:  19
Training...
loss=0.015238138847053051 batch_id=117: 100%|██████████| 118/118 [00:25<00:00,  4.57it/s]

Test set: Average loss: 0.0165, Accuracy: 9945/10000 (99.45%)

Running the notebook should yield similar output.

## Next Steps

This is a straightforward, lean CNN model used to demonstrate efficient image classification with PyTorch. Potential performance improvements could include:

- Exploring more complex or efficient CNN architectures
- Implementing data augmentation techniques
- Further tuning of hyperparameters, such as learning rate and batch size

## Contributing

Contributions to this project are welcome! Please review the contribution guidelines prior to submitting a pull request.


