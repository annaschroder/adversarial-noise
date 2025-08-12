# import necessary libraries
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
import torch
from PIL import Image

class AdversarialNoise:
    '''A class to create adversarial examples'''
    def __init__(self, model, input_data, true_label, target, epsilon):
        '''Initializes the AdversarialNoise class with a model, input data, and target.'''
        self.model = model
        self.input_data = input_data
        self.true_label = torch.tensor([true_label]) 
        self.target = target
        self.epsilon = epsilon

    def preprocess_image(self, image):
        '''Preprocesses the input image for the model.'''
        # try to understand why these settings are used
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        return preprocess(image).unsqueeze(0)

    def fgsm_attack(self, image, data_grad):
        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_image = image + self.epsilon * sign_data_grad
        
        # Clip the perturbed image values to ensure they stay within the valid range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        
        return perturbed_image

    def classify(self):
        '''Classifies the input image using the model.'''
        return

    def loss_function(self, model_output, true_label):
        '''Computes the loss between the model output and the target.'''
        return torch.nn.CrossEntropyLoss()(model_output, true_label)
    
    def altered_image(self):
        '''Returns the altered image with adversarial noise.'''

        # preprocess the image
        pre_proc_img = self.preprocess_image(self.input_data)
        # enables us to compute gradients with respect to the input image
        pre_proc_img.requires_grad = True

        # forward pass through the model
        output = self.model(pre_proc_img)
        predicted_label = torch.argmax(output, 1).item()

        loss = self.loss_function(output, self.true_label)

        # backwards pass to compute gradients
        self.model.zero_grad()
        loss.backward()
        data_grad = pre_proc_img.grad.data

        # generate adversarial noise using FGSM
        perturbed_image = self.fgsm_attack(pre_proc_img, data_grad)

        # classify the altered image
        output = self.model(perturbed_image)
        new_predicted_label = torch.argmax(output, 1).item()

        # ensure the altered image matches the target class

        # return the altered image
        return perturbed_image


def main():

    # Load an image from the local filesystem
    image = Image.open("data/n01443537_goldfish.jpeg").convert("RGB")  # Convert image to RGB format
    target_class = "hen"

    # Initialize model with the best available weights
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.eval()

    # Initialize the inference transforms
    preprocess = weights.transforms()

    # Define the epsilon values for noise - these should be in the range [0,1]
    #epsilons = [0, .05, .1, .15, .2, .25, .3]
    # Set random seed for reproducibility
    torch.manual_seed(42)

    altered_image = AdversarialNoise(model, image, 1, target_class, 0.1).altered_image()


if __name__ == "__main__":
    main()