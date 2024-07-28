### Fractal DC Gan

This is a toy repo for training a GAN to generate Fractal images (Mandelbrot's set). 

#### Generating Images

The images are generated based on Mandelbrot's set using colab notebook free-tier, and saved to
Google drive. Here is the [corresponding notebook](https://colab.research.google.com/drive/1MlhzkKnppuJw9iWEPKRrKaeEn4H2zZMi?usp=sharing), with credits from various GPTs for helping to optimise the generation of the images. Some sample images of this generation code have been included in `data/images/mandelbrot_{x}.png`.

#### DCGAN Training Code

The DCGAN Training code originally references [pytorch tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html). However it has been heavily refactored, notably in the build and training scripts.


 
