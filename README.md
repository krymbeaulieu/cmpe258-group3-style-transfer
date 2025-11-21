### cmpe258-group3-style-transfer
# Neural Style Transfer: Understanding Content and Style Representations in CNNs
Group 3 Members: Krystle Beaulieu: 00741572 | Esther Hammon: 017530032 | Jason Lee: 015293408

## Project Topic and Research Question
Our project explores Neural Style Transfer (NST) for Artistic and Texture Learning, using convolutional neural networks (CNNs) to separate and recombine content and style representations to generate new artistic images.
#### Our main research question is:
For Artistic and Texture Learning, how do different style representation methods (Gram matrices vs. adaptive feature statistics) affect the quality and efficiency in Neural Style Transfer?

In this project, we will use the methodology proposed in “A Neural Algorithm of Artistic Style” [1] by Gatys et al. (2016) to set a baseline for the traditional style and content loss Gram matrix calculations referenced in the article. We are proposing to use the dataset from WikiArt [2] that is available on Kaggle [3] to capture visual artistic style of famous painters and try to generate new images by combining the artistic style with images from the MSCOCO dataset [4]. As an extension, we plan to compare the methods referenced in the article written by Gatys et al. [1], with the faster mean and std calculations in the style and content loss calculations technique proposed in “A Neural Algorithm of Artistic Style with Adaptive Instance Normalization (AdaIN)” [5] by Huang & Belongie (2017). 

We plan to evaluate the results through qualitative and quantitative metrics. First we will  analyze the outcome visually by analyzing how well the artistic style was transferred over to the new images. Some quantitative metrics we can use is calculating the content and style loss over what a picture loses when transferring the style of the picture, which can be extracted from the training loss function. In the article written by Gatys et al. [1], they demonstrated that higher CNN layers store semantic content but discard texture details, validating that with the use of activations from higher layers you are able to essentially capture the content of an image. The Gram matrix is already used in the loss function for calculating the style loss and can be used to evaluate both Gram and AdaIN training methods. There are a couple of other proxy methods that can be used: 
Learned Perceptual Image Patch Similarity (LPIPS): LPIPS uses a pretrained CNN’s extracted features between the input image and style transferred output to see how closely they are related on an overall scale. This method could be more applicable compared to methods like Structural Similarity Index Measure (SSIM) that looks for crisper edges/gradients, which might be more useful for other tasks like super resolution.
Using a pretrained style classifier to evaluate the style transferred output data on how well a network classifies the output style. This is popular with the extension topic AdaIN. If the model weights are easily utilizable, then this evaluation method can be used within the scope of the project. 


### References

[1] A Neural Algorithm of Artistic Style (Gatys et al., 2016) https://arxiv.org/pdf/1508.06576

[2] Paintings from WikiArt.org

[3] A collection of WikiArt images on Kaggle https://www.kaggle.com/datasets/steubk/wikiart

[4] Pictures from MS COCO https://cocodataset.org/#home
[5] Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization (Xun Huang and Serge Belongie, 2017) https://arxiv.org/pdf/1703.06868 
