# adversarial-noise
A coding package to implement adverserial noise example, where we trick a machine learning model to make a false prediction using small, indetectable changes. In this package, we are attacking the ResNet50 pre-trained model using examples from ImageNet.

## create conda environment
Move into the folder containing this file, and run the following command to create a conda environment

```shell
conda env create -f environment.yml
```

Activate the conda environment:

```shell
conda activate adversarial_noise
```

## running the code
```shell
python AdverserialNoise.py --image_path data/n01443537_goldfish.jpeg --target_class "magpie"
```

The successful adversarial attack will be saved in the data/results folder. Here you will see the original image alongside the perturbed image.

You can change the target_class to another ImageNet classification, or point to another image on your local system to run on another image.

