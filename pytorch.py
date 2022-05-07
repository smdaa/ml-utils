import time
import numpy as np
import torch
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedKFold

def create_model():
    model = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3)),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=(2, 2)),
        torch.nn.Flatten(),
        torch.nn.Linear(in_features=5408, out_features=100),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=100, out_features=10),
        torch.nn.Softmax(dim = 1)
    )
    return model

def summarize_diagnostics(histories):
	for i in range(len(histories)):
		plt.subplot(2, 1, 1)
		plt.title('Cross Entropy Loss')
		plt.plot(histories[i]['loss'], color='blue', label='train')
		plt.subplot(2, 1, 2)
		plt.title('Classification Accuracy')
		plt.plot(histories[i]['accuracy'], color='blue', label='train')
	plt.show()

def summarize_performance(scores):
	print('Accuracy: mean=%.3f std=%.3f, n=%d' % (np.mean(scores)*100, np.std(scores)*100, len(scores)))
	plt.boxplot(scores)
	plt.show()

# device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# load data
x = np.load('x.npy')
y = np.load('y.npy')

print('working with', y.size, 'samples')

# shuffle data
x, y = shuffle(x, y)

# 5 fold cross validation
n_splits  = 5
n_repeats = 1
scores    = []
histories = []

kf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=None)

start = time.time()
for train_index, test_index in kf.split(x):
    # train test split
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # expand dim (1 channel)
    x_train = np.expand_dims(x_train, axis=1)
    x_test  = np.expand_dims(x_test,  axis=1)

    x_train = torch.from_numpy(x_train)
    x_test  = torch.from_numpy(x_test)

    y_train = torch.from_numpy(y_train)
    y_test  = torch.from_numpy(y_test)

    train_data = torch.utils.data.TensorDataset(x_train, y_train)
    test_data  = torch.utils.data.TensorDataset(x_test, y_test)

    # load model 
    model = create_model().to(device)
    learning_rate = 0.01
    momentum      = 0.9
    optimizer     = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    loss_fn       = torch.nn.CrossEntropyLoss()

    # train model
    epochs     = 10
    batch_size = 32
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    total_step = len(train_loader)
    history    = {'loss':[], 'accuracy':[]}

    for epoch in range(epochs):
        running_loss = 0
        correct      = 0
        total        = 0

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted  = outputs.max(1)
            total        += labels.size(0)
            correct      += predicted.eq(labels).sum().item()

        train_loss = running_loss / total_step
        accuracy       = 100.*correct/total

        history['loss'].append(train_loss)
        history['accuracy'].append(accuracy)

        print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.3f}, Accuracy: {:.3f}' 
                .format(epoch+1, epochs, i+1, total_step, train_loss, accuracy))

    histories.append(history)

    # test model
    model.eval() 
    with torch.no_grad():
        correct = 0
        total   = 0
        for images, labels in test_loader:
            images = images.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)

            outputs      = model(images)
            _, predicted = outputs.max(1)
            total        += labels.size(0)
            correct      += predicted.eq(labels).sum().item()

        accuracy       = correct/total
        scores.append(accuracy)
        print('Test accuracy : %.3f' % (accuracy * 100.0))

end = time.time()
summarize_diagnostics(histories)
summarize_performance(scores)
print('elapsed time', end - start)
