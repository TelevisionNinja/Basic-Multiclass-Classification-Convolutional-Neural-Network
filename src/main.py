import torch
import torchvision
import matplotlib.pyplot
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import random
import mlxtend.plotting
import torchmetrics

from timeit import default_timer
from tqdm.auto import tqdm

device = 'cpu'

if torch.cuda.is_available():
    device = 'cuda'


class Model(torch.nn.Module):
    def __init__(self, num_inputs=2, num_outputs=2, hidden_layer_size=10, width=32, height=32):
        super().__init__()

        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=num_inputs,
                            out_channels=hidden_layer_size,
                            kernel_size=(3, 3),
                            stride=(1, 1),
                            padding=(1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=hidden_layer_size,
                            out_channels=hidden_layer_size,
                            kernel_size=(3, 3),
                            stride=(1, 1),
                            padding=(1, 1)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2))
        )

        new_width = self.compute_convolution_output_shape(x=width,
                                                  kernel=3,
                                                  stride=1,
                                                  padding=1)
        new_height = self.compute_convolution_output_shape(x=height,
                                                   kernel=3,
                                                   stride=1,
                                                   padding=1)

        new_width = self.compute_convolution_output_shape(x=new_width,
                                                  kernel=3,
                                                  stride=1,
                                                  padding=1)
        new_height = self.compute_convolution_output_shape(x=new_height,
                                                   kernel=3,
                                                   stride=1,
                                                   padding=1)

        new_width = self.compute_convolution_output_shape(x=new_width,
                                                  kernel=2,
                                                  stride=2)
        new_height = self.compute_convolution_output_shape(x=new_height,
                                                   kernel=2,
                                                   stride=2)

        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=hidden_layer_size,
                            out_channels=hidden_layer_size,
                            kernel_size=(3, 3),
                            stride=(1, 1),
                            padding=(1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=hidden_layer_size,
                            out_channels=hidden_layer_size,
                            kernel_size=(3, 3),
                            stride=(1, 1),
                            padding=(1, 1)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2))
        )

        new_width = self.compute_convolution_output_shape(x=new_width,
                                                  kernel=3,
                                                  stride=1,
                                                  padding=1)
        new_height = self.compute_convolution_output_shape(x=new_height,
                                                   kernel=3,
                                                   stride=1,
                                                   padding=1)

        new_width = self.compute_convolution_output_shape(x=new_width,
                                                  kernel=3,
                                                  stride=1,
                                                  padding=1)
        new_height = self.compute_convolution_output_shape(x=new_height,
                                                   kernel=3,
                                                   stride=1,
                                                   padding=1)

        new_width = self.compute_convolution_output_shape(x=new_width,
                                                  kernel=2,
                                                  stride=2)
        new_height = self.compute_convolution_output_shape(x=new_height,
                                                   kernel=2,
                                                   stride=2)

        size = hidden_layer_size * new_width * new_height

        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=size,
                            out_features=num_outputs)
        )

        self.loss_function = torch.nn.CrossEntropyLoss()
        self.learning_rate = 0.01
        self.optimizer = torch.optim.SGD(params=self.parameters(), lr=self.learning_rate)
        self.epochs = 8

        self.to(device=device)


    def compute_convolution_output_shape(self, x = 32, kernel = 3, stride = 1, padding = 0, dilation = 1):
        return (x + padding * 2 - dilation * (kernel - 1) - 1) // stride + 1


    def accuracy_function(self, true_values = None, predicted_values = None):
        number_of_correct = torch.eq(true_values, predicted_values).sum().item()
        return number_of_correct / len(predicted_values) * 100


    def classification_function(self, logits):
        return torch.softmax(logits, dim=1).argmax(dim=1)


    def forward(self, x):
        return self.classifier(self.block2(self.block1(x)))


    def save_model(self, name='model.pt'):
        print('saving')
        torch.save(obj=self.state_dict(), f=name)
        print('done')


    def load_model(self, name='model.pt'):
        self.load_state_dict(torch.load(name))
        self.to(device=device)


    def start_training(self, training_dataloader = None, test_dataloader = None, training_loss_values = None, test_loss_values = None, training_accuracy_values = None, test_accuracy_values = None):
        print('training')

        for _ in tqdm(range(self.epochs)):
            # training
            self.train()
            training_loss = 0
            training_accuracy = 0

            for x, y in tqdm(training_dataloader): # loop through batches
                x = x.to(device)
                y = y.to(device)
                # forward pass
                predicted_logits = self(x)
                predicted_classification = self.classification_function(predicted_logits)

                # loss calculation
                loss = self.loss_function(predicted_logits, y)
                accuracy = self.accuracy_function(true_values = y,
                                                  predicted_values = predicted_classification)

                training_loss += loss
                training_accuracy += accuracy

                # zero out the gradient
                self.optimizer.zero_grad()

                # backpropagation
                loss.backward()

                # gradient descent
                self.optimizer.step()

            training_loss_values.append(training_loss / len(training_dataloader))
            training_accuracy_values.append(training_accuracy / len(training_dataloader))

            # testing
            self.eval()
            test_loss = 0
            test_accuracy = 0

            with torch.inference_mode():
                for test_x, test_y in tqdm(test_dataloader):
                    test_x = test_x.to(device)
                    test_y = test_y.to(device)
                    # testing
                    predicted_logits = self(test_x)
                    predicted_classification = self.classification_function(predicted_logits)

                    # loss calculation
                    loss = self.loss_function(predicted_logits, test_y)
                    accuracy = self.accuracy_function(true_values = test_y,
                                                            predicted_values = predicted_classification)
                    test_loss += loss
                    test_accuracy += accuracy

            test_loss_values.append(test_loss / len(test_dataloader))
            test_accuracy_values.append(test_accuracy / len(test_dataloader))


def plot_loss(training_loss = None, test_loss = None, name = 'loss.png'):
    training_loss = torch.tensor(training_loss).numpy()
    test_loss = torch.tensor(test_loss).numpy()
    matplotlib.pyplot.figure(figsize=(10, 7))
    matplotlib.pyplot.title('loss curves')
    matplotlib.pyplot.xlabel('iterations')
    matplotlib.pyplot.ylabel('loss')
    training_iterations = range(len(training_loss))
    test_iterations = range(len(test_loss))
    matplotlib.pyplot.plot(training_iterations, training_loss, label='training loss')
    matplotlib.pyplot.plot(test_iterations, test_loss, label='testing loss')
    matplotlib.pyplot.legend()
    matplotlib.pyplot.savefig(name)


def plot_accuracy(training_accuracy = None, test_accuracy = None, name = 'accuracy.png'):
    training_accuracy = torch.tensor(training_accuracy).numpy()
    test_accuracy = torch.tensor(test_accuracy).numpy()
    matplotlib.pyplot.figure(figsize=(10, 7))
    matplotlib.pyplot.title('accuracy curves')
    matplotlib.pyplot.xlabel('iterations')
    matplotlib.pyplot.ylabel('accuracy')
    training_iterations = range(len(training_accuracy))
    test_iterations = range(len(test_accuracy))
    matplotlib.pyplot.plot(training_iterations, training_accuracy, label='training accuracy')
    matplotlib.pyplot.plot(test_iterations, test_accuracy, label='testing accuracy')
    matplotlib.pyplot.legend()
    matplotlib.pyplot.savefig(name)


def make_time_str(start = 0, end = 0):
    total_seconds = end - start
    s = total_seconds % 60
    total_seconds = total_seconds // 60
    min = total_seconds % 60
    total_seconds = total_seconds // 60
    hr = total_seconds % 24
    days = total_seconds // 24

    return f'{days}d {hr}h {min}min {s}s'


def load_model_and_generate_images():
    # seed = 314
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)

    print('downloading dataset')

    download_dataset = False

    training_set = torchvision.datasets.FashionMNIST(root='dataset',
                                      train=True,
                                      download=download_dataset,
                                      transform=torchvision.transforms.ToTensor())

    test_set = torchvision.datasets.FashionMNIST(root='dataset',
                                      train=False,
                                      download=download_dataset,
                                      transform=torchvision.transforms.ToTensor())

    # display dataset items

    image, label = training_set[0]

    matplotlib.pyplot.figure(figsize=(10, 7))
    matplotlib.pyplot.title(training_set.classes[label])
    matplotlib.pyplot.axis(False)
    matplotlib.pyplot.imshow(image.squeeze(), cmap='gray')
    matplotlib.pyplot.savefig('raw data.png')

    figure = matplotlib.pyplot.figure(figsize=(9,9))
    rows = 5
    columns = 5

    for i in range(1, rows * columns + 1):
        random_index = torch.randint(0, len(training_set), size=[1]).item()
        image, label = training_set[random_index]
        figure.add_subplot(rows, columns, i)
        matplotlib.pyplot.title(training_set.classes[label])
        matplotlib.pyplot.axis(False)
        matplotlib.pyplot.imshow(image.squeeze(), cmap='gray')

    figure.savefig('raw data grid.png')

    # make data loaders

    batch_size = 32

    training_dataloader = DataLoader(dataset=training_set,
                                     batch_size=batch_size,
                                     shuffle=True)
    test_dataloader = DataLoader(dataset=test_set,
                                 batch_size=batch_size)

    print(f'training data batches ({batch_size} images per batch): {len(training_dataloader)}')
    print(f'test data batches ({batch_size} images per batch): {len(test_dataloader)}')

    number_of_color_channels = image.shape[0] # channel, height, width
    height = image.shape[1]
    width = image.shape[2]
    number_of_classes = len(training_set.classes)
    brain = Model(num_inputs=number_of_color_channels,
                  num_outputs=number_of_classes,
                  width=width,
                  height=height)

    brain.load_model()

    batch_x, batch_y = next(iter(training_dataloader))
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)

    # visualize
    writer = SummaryWriter('tensorboard/') # python -m tensorboard.main --logdir=tensorboard
    writer.add_graph(brain, batch_x)
    writer.close()


def train_model_and_generate_images():
    # seed = 314
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)

    print('downloading dataset')

    download_dataset = True

    training_set = torchvision.datasets.FashionMNIST(root='dataset',
                                      train=True,
                                      download=download_dataset,
                                      transform=torchvision.transforms.ToTensor())

    test_set = torchvision.datasets.FashionMNIST(root='dataset',
                                      train=False,
                                      download=download_dataset,
                                      transform=torchvision.transforms.ToTensor())

    # display dataset items

    image, label = training_set[0]

    matplotlib.pyplot.figure(figsize=(10, 7))
    matplotlib.pyplot.title(training_set.classes[label])
    matplotlib.pyplot.axis(False)
    matplotlib.pyplot.imshow(image.squeeze(), cmap='gray')
    matplotlib.pyplot.savefig('raw data.png')

    figure = matplotlib.pyplot.figure(figsize=(9,9))
    rows = 5
    columns = 5

    for i in range(1, rows * columns + 1):
        random_index = torch.randint(0, len(training_set), size=[1]).item()
        image, label = training_set[random_index]
        figure.add_subplot(rows, columns, i)
        matplotlib.pyplot.title(training_set.classes[label])
        matplotlib.pyplot.axis(False)
        matplotlib.pyplot.imshow(image.squeeze(), cmap='gray')

    figure.savefig('raw data grid.png')

    # make data loaders

    batch_size = 32

    training_dataloader = DataLoader(dataset=training_set,
                                     batch_size=batch_size,
                                     shuffle=True)
    test_dataloader = DataLoader(dataset=test_set,
                                 batch_size=batch_size)

    print(f'training data batches ({batch_size} images per batch): {len(training_dataloader)}')
    print(f'test data batches ({batch_size} images per batch): {len(test_dataloader)}')

    # number_of_pixels = 28 * 28
    number_of_color_channels = image.shape[0] # channel, height, width
    height = image.shape[1]
    width = image.shape[2]
    number_of_classes = len(training_set.classes)
    brain = Model(num_inputs=number_of_color_channels,
                  num_outputs=number_of_classes,
                  width=width,
                  height=height)
    
    training_loss_values = []
    test_loss_values = []
    training_accuracy_values = []
    test_accuracy_values = []

    start_time = default_timer()

    brain.start_training(training_dataloader=training_dataloader,
                         test_dataloader=test_dataloader,
                         training_loss_values = training_loss_values,
                         test_loss_values = test_loss_values,
                         training_accuracy_values = training_accuracy_values,
                         test_accuracy_values = test_accuracy_values)

    end_time = default_timer()

    print('training time: ' + make_time_str(start=start_time, end=end_time))

    brain.save_model()

    plot_loss(training_loss = training_loss_values,
              test_loss = test_loss_values)
    plot_accuracy(training_accuracy = training_accuracy_values,
                  test_accuracy = test_accuracy_values)
    
    print(f'training accuracy: {training_accuracy_values[-1]}\ntest accuracy: {test_accuracy_values[-1]}')
    print(f'training loss: {training_loss_values[-1]}\ntest loss: {test_loss_values[-1]}')

    batch_x, batch_y = next(iter(training_dataloader))
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)

    # visualize
    writer = SummaryWriter('tensorboard/') # python -m tensorboard.main --logdir=tensorboard
    writer.add_graph(brain, batch_x)
    writer.close()


def make_predictions(model = None, data = None):
    model.eval()
    predictions = []

    with torch.inference_mode():
        for input in data:
            input = torch.unsqueeze(input, dim=0).to(device=device)

            logit = model(input)
            probability = torch.softmax(logit.squeeze(), dim=0)

            predictions.append(probability)

    return torch.stack(predictions).argmax(dim=1).cpu()


def load_model_and_predict():
    # seed = 314
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # random.seed(seed)

    print('downloading dataset')

    download_dataset = False

    test_set = torchvision.datasets.FashionMNIST(root='dataset',
                                      train=False,
                                      download=download_dataset,
                                      transform=torchvision.transforms.ToTensor())

    grid_size = 3

    test_input = []
    test_output = []
    for input, output in random.sample(list(test_set), k=grid_size*grid_size):
        test_input.append(input)
        test_output.append(output)

    print('labeling images')

    image, label = test_set[0]
    number_of_color_channels = image.shape[0] # channel, height, width
    height = image.shape[1]
    width = image.shape[2]
    number_of_classes = len(test_set.classes)
    brain = Model(num_inputs=number_of_color_channels,
                  num_outputs=number_of_classes,
                  width=width,
                  height=height)
    brain.load_model()
    predictions = make_predictions(model=brain,
                                   data=test_input)

    matplotlib.pyplot.figure(figsize=(10, 10))
    for i, input in enumerate(test_input):
        matplotlib.pyplot.subplot(grid_size, grid_size, i + 1)

        matplotlib.pyplot.imshow(input.squeeze(), cmap='gray')

        predicted_label = test_set.classes[predictions[i]]
        true_label = test_set.classes[test_output[i]]

        title = f'predicted: {predicted_label} | actual: {true_label}'

        if predicted_label == true_label:
            matplotlib.pyplot.title(title, fontsize=10, c='g')
        else:
            matplotlib.pyplot.title(title, fontsize=10, c='r')
    matplotlib.pyplot.savefig('results.png')

    # confusion matrix

    batch_size = 32
    test_dataloader = DataLoader(dataset=test_set,
                                 batch_size=batch_size)
    predictions = []
    with torch.inference_mode():
        for input, output in tqdm(test_dataloader):
            input = input.to(device=device)
            output = output.to(device=device)

            logit = brain(input)
            prediction = brain.classification_function(logit)
            predictions.append(prediction)
    predictions_tensor = torch.cat(predictions).cpu()

    confusion_matrix = torchmetrics.ConfusionMatrix(task='multiclass',
                                                    num_classes=len(test_set.classes))
    confusion_matrix_tensor = confusion_matrix(preds=predictions_tensor,
                                               target=test_set.targets)

    figure, axes = mlxtend.plotting.plot_confusion_matrix(conf_mat=confusion_matrix_tensor.numpy(),
                                                           class_names=test_set.classes,
                                                           figsize=(10, 7))
    figure.savefig('confusion matrix.png')


def main():
    train_model_and_generate_images()
    # load_model_and_generate_images()
    load_model_and_predict()


if __name__ == '__main__':
    main()
