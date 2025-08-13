# import necessary libraries
import numpy as np
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
import torch
from PIL import Image
import matplotlib.pyplot as plt
import argparse

class AdversarialNoise:
    '''A class to create adversarial examples'''
    def __init__(self, model, input_data, target_label, epsilon):
        '''Initializes the AdversarialNoise class with a model, input data, and target.'''
        self.model = model
        self.input_data = input_data
        self.target_label = torch.tensor([target_label])
        self.epsilon = epsilon

    def preprocess_image(self, image):
        '''Preprocesses the input image for the model.'''
        # try to understand why these settings are used
        
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        return preprocess(image).unsqueeze(0)
    
    def fgsm_attack_targetted(self, image, data_grad):
        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_image = image - (self.epsilon * sign_data_grad)

        # Clip the perturbed image values to ensure they stay within the valid range
        perturbed_image = torch.clamp(perturbed_image, 0, 1).detach()
        
        return perturbed_image

    def classify(self):
        '''Classifies the input image using the model.'''
        return
    
    def altered_image(self):
        '''Returns the altered image with adversarial noise.'''

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

        # preprocess the image
        img = self.preprocess_image(self.input_data)

        # enables us to compute gradients with respect to the input image
        img.requires_grad = True

        # forward pass through the model
        original_output = self.model((img - mean) / std)
        original_predicted_label = torch.argmax(original_output, 1).item()
        original_prob = torch.softmax(original_output, dim=1)

        conf_orig_pred = original_prob[0, original_predicted_label].item()
        conf_orig_target = original_prob[0, self.target_label.item()].item()

        print(f"Before attack: predicted={original_predicted_label} "
            f"(conf={conf_orig_pred:.4f}), target={self.target_label.item()} "
            f"(conf={conf_orig_target:.4f})")

        loss_function = torch.nn.CrossEntropyLoss()
        loss = loss_function(original_output, self.target_label)

        # backwards pass to compute gradients
        self.model.zero_grad()
        loss.backward()
        data_grad = img.grad.data #/ std # correct scaling to pixel space

        # generate adversarial noise using FGSM
        perturbed_image = self.fgsm_attack_targetted(img, data_grad)

        # classify the altered image
        perturbed_output = self.model((perturbed_image - mean) / std)
        perturbed_predicted_label = torch.argmax(perturbed_output, 1).item()
        perturbed_prob = torch.softmax(perturbed_output, dim=1)

        conf_perturbed_pred = perturbed_prob[0, perturbed_predicted_label].item()
        conf_perturbed_target = perturbed_prob[0, self.target_label.item()].item()
        print(f"After attack: predicted={perturbed_predicted_label} "
            f"(conf={conf_perturbed_pred:.4f}), target={self.target_label.item()} "
            f"(conf={conf_perturbed_target:.4f})")

        # ensure the altered image matches the target class
        if perturbed_predicted_label != self.target_label.item():
            print(f"Warning: The predicted label {perturbed_predicted_label} of the altered image does not match the target class {self.target_label.item()}.")
        
        return perturbed_image, img, original_predicted_label, perturbed_predicted_label
    
    
def plot_images(original, original_label, altered, altered_label, target_label):
    '''Plots the original and altered images side by side.'''

    image_np = np.transpose(original.squeeze().detach().numpy(), (1, 2, 0))
    perturbed_image_np = np.transpose(altered.squeeze().detach().numpy(), (1, 2, 0))

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(image_np)
    ax[0].set_title('Original Image, Predicted: {}'.format(original_label))
    ax[0].axis('off')

    ax[1].imshow(perturbed_image_np)
    ax[1].set_title('Altered Image with Adversarial Noise, Predicted: {}, Target: {}'.format(altered_label, target_label))
    ax[1].axis('off')

    return fig


def main():

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Adversarial Noise Generation")
    parser.add_argument('--image_path', type=str, default='data/n01443537_goldfish.jpeg', help='Path to the input image')
    parser.add_argument('--target_class', type=str, default='hen', help='Target class for adversarial attack')
    args = parser.parse_args()

    # Load an image from the local filesystem
    image = Image.open(args.image_path).convert("RGB")  # Convert image to RGB format
    target_class = args.target_class

    # Initialize model with the best available weights
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.eval()
    
    # could make this a function to avoid repetition
    target_class_index = weights.meta["categories"].index(target_class)

    # Define the epsilon values for noise - these should be in the range [0,1]
    epsilons = [0, .05, .1, .15, .2, .25, .3, .35, .4, .45, .5]
    #epsilons = [0, .01, .02, .03, .04, .5, .06, .07]

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Loop through each epsilon value and generate adversarial noise
    for epsilon in epsilons:

        altered_image, preprocessed_image, predicted_label, new_predicted_label = AdversarialNoise(model, image, target_class_index, epsilon).altered_image()
        fig = plot_images(
            preprocessed_image, 
            weights.meta["categories"][predicted_label], 
            altered_image, 
            weights.meta["categories"][new_predicted_label], 
            weights.meta["categories"][target_class_index]
            )
        
        plt.savefig(f"data/adversarial_noise_epsilon_{epsilon}.png")

if __name__ == "__main__":
    main()