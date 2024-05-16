import matplotlib.pyplot as plt
import pickle

def plot_loss(info):
    if 'loss' in info.history and 'val_loss' in info.history:
        # Loss performance
        fig = plt.figure()
        plt.plot(info.history['loss'], color='blue', label='loss')
        plt.plot(info.history['val_loss'], color='orange', label='val_loss')
        fig.suptitle('Loss', fontsize=20)
        plt.legend(loc='upper left')
        plt.show()
    else:
        print("The loaded history does not contain loss information.")

def plot_accuracy(info):
    if 'accuracy' in info.history and 'val_accuracy' in info.history:
        # Accuracy performance
        fig = plt.figure()
        plt.plot(info.history['accuracy'], color='blue', label='accuracy')
        plt.plot(info.history['val_accuracy'], color='orange', label='val_accuracy')
        fig.suptitle('Accuracy', fontsize=20)
        plt.legend(loc='upper left')
        plt.show()
    else:
        print("The loaded history does not contain accuracy information.")

# Load the training history whenever needed
try:
    with open('training_history.pkl', 'rb') as f:
        loaded_info = pickle.load(f)
except EOFError:
    print("EOFError: The file is empty or corrupted.")
except FileNotFoundError:
    print("File not found. Please check the file path.")
except Exception as e:
    print("An error occurred:", e)
else:
    plot_loss(loaded_info)
    plot_accuracy(loaded_info)
