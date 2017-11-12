# Convolutional Neural Network

To run the program
```
pip install -r requirements.txt

python cnnTF.py

```

## 3 different convolutional models have been implemented.

1.
- 7x7 Conv with stride 2
- Relu activation
- Fully connected that map to 10 output classes

### Results:
![Simple model](https://github.com/ayesha92ahmad/convolutional-neural-network/blob/master/simple-model.png)

2.
- 7x7 Conv with stride 2
- Relu activation
- 2x2 Max Pooling
- Fully connected with 1024 hidden neurons § Relu activation
- Fully connected that map to 10 output classes

### Results:
![Complex network](https://github.com/ayesha92ahmad/convolutional-neural-network/blob/master/complex-model.png)

3.
- 7x7 Conv with stride 2
- Relu activation
- 2x2 Max Pooling
- 5x5 Conv with stride 1
- Relu activation
- 2x2 Max Pooling
- Fully connected with 1024 hidden neurons
- Relu activation
- Fully connected with 128 hidden neurons
- Relu activation
- Fully connected that map to 10 output classes

### Results:
![another complex model](https://github.com/ayesha92ahmad/convolutional-neural-network/blob/master/ownmodel-without-dropoff.png)

Since I have a MAC book without GPU this entire program takes a long time to run (about 8-9 hours).
