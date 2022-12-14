# Neural Style Transfer

This Jupyter notebook allows users to; create semantic segmentation masks on a pair of content and style images, run the original NST algorithm, run a spatially controlled extension and run images through a canny edge detector.

## Instructions

Note: The imagenet-vgg-verydeep-19.mat file can be found here https://www.vlfeat.org/matconvnet/pretrained/

Follow these steps to run a complete experiment:
1. First, choose a model back bone to use for the segmentation model in the MODELNAME variable.
2. Find a content image and copy the image address into the imageurl variable to generate a segmentation mask and repeat this step for the style image.
3. If you are using manually annotated masks, make sure to resize them to the width and dimensions of the images you will pass through the CNN before placing them in the directory.
4. Set the adjustable parameters and be sure to give the output directory a name that is unique or the previous file will be overwritten with new results.
5. Once all desired parameters have been set. Execute train() to begin the training procedure.
6. Once the number of epochs specified for training has been performed. Save the loss values and plot them against the number of iterations.
7. Run the canny edge detector to pass the original and stylised image through the detector and store the edge detection maps.
8. Go to the named output directory for the experiment where the inputs and outputs are automatically saved including the loss graph, time taken and the stylised image at each step.

## Installation

Create a Conda environment:

```bash
conda create --name myEnv
```

Activate the environment using the command shown in the console,

Install the packages mentioned in the 'requirements.txt' file, e.g. for Tensorflow:

```bash
conda install -c conda-forge tensorflow
```

To set this Conda environment on your Jupyter notebook, install ipykernel:

```bash
conda install -c anaconda ipykernel
```

Type the below to add the Conda environment to the Jupyter notebook kernels drop down:

```bash
python -m ipykernel install --user --name=myEnv
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
