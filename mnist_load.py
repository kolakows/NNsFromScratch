import numpy as np
# img_path = r'.\MNIST\train-images.idx3-ubyte'
# label_path =  r'.\MNIST\train-labels.idx1-ubyte'

def read_images(img_path, byteorder = 'big'):
    '''
    Reads MNIST pixel data, returned format is a list of 28x28 images
    '''
    with open(img_path, "rb") as f:
        magic_number = f.read(4)
        imgs_count = f.read(4)
        rows, cols = f.read(4), f.read(4)

        magic_number = int.from_bytes(magic_number, byteorder)
        imgs_count = int.from_bytes(imgs_count, byteorder)
        rows = int.from_bytes(rows, byteorder)
        cols = int.from_bytes(cols, byteorder)

        imgs = []
        readimgs = 0
        pixel = f.read(1)
        while pixel:
            img = []
            for i in range(rows):
                row = []
                for j in range(cols):
                    row.append(int.from_bytes(pixel, byteorder))
                    pixel = f.read(1)
            imgs.append(img)
            readimgs += 1
        print(f'Read {readimgs}, expected {imgs_count}, file magic_number={magic_number}')
    return np.array(imgs)

def read_labels(label_path, byteorder = 'big'):
    '''
    Reads MNIST labels, returned format is a list of int labels
    '''
    with open(label_path, "rb") as f:
        magic_number = f.read(4)
        label_count = f.read(4)

        magic_number = int.from_bytes(magic_number, byteorder)
        label_count = int.from_bytes(label_count, byteorder)

        labels = []
        readlabels = 0
        label = f.read(1)
        while label:
            labels.append(int.from_bytes(label, byteorder))
            readlabels += 1
            label = f.read(1)
        print(f'Read {readlabels}, expected {label_count}, file magic_number={magic_number}')
    return np.array(labels)
