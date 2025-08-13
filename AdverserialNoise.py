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
        '''Initializes the AdversarialNoise class with a model, input data, and target and epsilon.
        
        Args:
            model: The model to be attacked.
            input_data: The input image data to be perturbed.
            target_label: The class which we are aiming to misclassify as.
            epsilon: The perturbation magnitude for the adversarial noise.
        '''
        self.model = model
        self.input_data = input_data
        self.target_label = torch.tensor([target_label])
        self.epsilon = epsilon

    def preprocess_image(self, image):
        '''Preprocesses the input image for the model (without the normalisation - we'll do this when we put the image into the model).
        
        Args:
            image: The input image to be preprocessed.

        Returns:
            A preprocessed image tensor.
        '''
        
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        return preprocess(image).unsqueeze(0)
    
    def fgsm_attack_targetted(self, image, data_grad):
        '''Generates a perturbed image using the Fast Gradient Sign Method (FGSM) for targeted attacks.

        Args:
            image: The input image tensor.
            data_grad: The gradient of the loss with respect to the input image.    

        Returns:
            perturbed_image: The perturbed image tensor after applying adversarial noise.
        '''

        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_image = image - (self.epsilon * sign_data_grad)

        # Clip the perturbed image values to ensure they stay within the valid range
        perturbed_image = torch.clamp(perturbed_image, 0, 1).detach()
        
        return perturbed_image
    
    def fgsm_attack_targetted_iterative_step(self, perturbed_image, image, data_grad, alpha=0.01):
        '''Moving one step of the iterative approach using the Fast Gradient Sign Method (FGSM) for targeted attacks.

        Args:
            image: The original input image tensor.
            perturbed_image: The current perturbed image tensor.
            data_grad: The gradient of the loss with respect to the input image.   
            alpha: The step size for the iterative attack. 

        Returns:
            perturbed_image: The perturbed image tensor after applying the iterative adversarial noise step.
        '''
        # take a step in the direction of the gradient
        perturbed_image = perturbed_image - alpha * data_grad.sign()

        # Clip pertubation to ensure they stay within the valid range
        pertubation = torch.clamp(perturbed_image - image, -self.epsilon, self.epsilon)

        # Apply the perturbation to the original image
        perturbed_image = image + pertubation
        
        # Clip the perturbed image values to ensure they stay within the valid range
        perturbed_image = torch.clamp(perturbed_image, 0, 1).detach()

        return perturbed_image

    def classify(self, image, mean, std):
        '''Classifies an input image using the model.
        
        Args:
            image: The input image tensor to be classified.
            mean: The mean values for normalisation.
            std: The standard deviation values for normalisation.
            
        Returns:
            output: The model's output logits for the input image.
            predicted_label: The predicted label of the input image.
            prob: The probabilities of each class for the input image.
        '''

        # input normalised image
        output = self.model((image - mean) / std)
        predicted_label = torch.argmax(output, 1).item()
        probabilities = torch.softmax(output, dim=1)

        return output, predicted_label, probabilities

    def altered_image(self):
        '''Runs the adversarial noise generation process and returns the altered image, original image, and their predicted labels.
        
        Returns:
            perturbed_image: The image after applying the adversarial attack.
            img: The original preprocessed image tensor.
            original_predicted_label: The predicted label of the original image.
            perturbed_predicted_label: The predicted label of the image after the adversarial attack.
        '''

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

        # preprocess the image
        img = self.preprocess_image(self.input_data)

        # enables us to compute gradients with respect to the input image
        img.requires_grad = True

        # forward pass through the model
        original_output, original_predicted_label, original_prob = self.classify(img, mean, std)

        print(f"Before attack: predicted={original_predicted_label} "
            f"(conf={original_prob[0, original_predicted_label].item():.4f}), target={self.target_label.item()} "
            f"(conf={original_prob[0, self.target_label.item()].item():.4f})")

        loss_function = torch.nn.CrossEntropyLoss()
        loss = loss_function(original_output, self.target_label)

        # backwards pass to compute gradients
        self.model.zero_grad()
        loss.backward()
        data_grad = img.grad.data

        # generate adversarial noise using FGSM
        perturbed_image = img.detach().clone()

        Niter = 10  # Number of iterations for iterative FGSM
        for _ in range(Niter):
            perturbed_image.requires_grad = True
            model_output = self.model((perturbed_image - mean) / std)
            loss = torch.nn.CrossEntropyLoss()(model_output, self.target_label)
            self.model.zero_grad()
            loss.backward()

            # take a step in the direction of the gradient
            perturbed_image = self.fgsm_attack_targetted_iterative_step(perturbed_image, img, perturbed_image.grad.data)

        # classify the altered image
        _, perturbed_predicted_label, perturbed_prob = self.classify(perturbed_image, mean, std)

        print(f"After attack: predicted={perturbed_predicted_label} "
            f"(conf={perturbed_prob[0, perturbed_predicted_label].item():.4f}), target={self.target_label.item()} "
            f"(conf={perturbed_prob[0, self.target_label.item()].item():.4f})")

        # ensure the altered image matches the target class
        if perturbed_predicted_label != self.target_label.item():
            print(f"Warning: The predicted label {perturbed_predicted_label} of the altered image does not match the target class {self.target_label.item()}.")
        
        return perturbed_image, img, original_predicted_label, perturbed_predicted_label
    
    
def plot_images(original, original_label, altered, altered_label, target_label):
    '''Function to plot the results of the adversarial noise generation.
    Args:
        original: The original preprocessed image tensor.
        original_label: The predicted label of the original image.
        altered: The perturbed image tensor after the adversarial attack.
        altered_label: The predicted label after the adversarial attack.
        target_label: The target class label for the adversarial attack.
        
    Returns:
        fig: A matplotlib figure containing the original and perturbed images with their predicted labels.
    '''

    # Convert tensors to numpy arrays for plotting
    image_np = np.transpose(original.squeeze().detach().numpy(), (1, 2, 0))
    perturbed_image_np = np.transpose(altered.squeeze().detach().numpy(), (1, 2, 0))

    # plot as two subplots with predicted/target labels in titles
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(image_np)
    ax[0].set_title('Original Image, Predicted: {}'.format(original_label))
    ax[0].axis('off')

    ax[1].imshow(perturbed_image_np)
    ax[1].set_title('Altered Image with Adversarial Noise, Predicted: {}, Target: {}'.format(altered_label, target_label))
    ax[1].axis('off')

    return fig


def main():
    '''Main function to run the adversarial noise generation process.'''

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
    
    target_class_index = weights.meta["categories"].index(target_class)

    # Define the epsilon values for noise - these should be in the range [0,1]
    epsilons = [0, .01, .02, .03, .04, .05, .06, .07]

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Loop through each epsilon value and generate adversarial noise
    for epsilon in epsilons:

        adversarial_noise_generator = AdversarialNoise(model,image, target_class_index, epsilon)
        altered_image, preprocessed_image, predicted_label, new_predicted_label = adversarial_noise_generator.altered_image()

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