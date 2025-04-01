##  Dialect Identification
The dataset used for training is a set of wav file recordings of different speakers of eight different dialects. Importantly, the distributions of dialects are dissimilar between the train and dev sets (Fig. 1), which required weighting the loss function to prevent the model from defaulting to the most frequent class in the training dataset. Additionally, there is zero overlap in speakers between the training and dev datasets, so generalization to new speakers of each dialect is crucial.

<img src="https://github.com/user-attachments/assets/bdc10b06-647e-4f32-8463-02d2aa93b799" width="400" /> \
Figure 1: Distribution of dialects in the train and dev set. 

The model I used was a convolutional neural network composed of 7 convolutional layers and 3 max pooling layers, with BatchNorm after each convolutional layer, and ReLU after each convolutional or linear layer.
Feature extraction was done through converting waveforms to dB scaled mel spectrograms. MFCC spectrograms were tested as well, but the model struggled to learn the training set in that case. The final model used 40 mel filter banks, a window length of 400 frames (25 ms) and a hop length of 160 frames (10 ms). 

<img src="https://github.com/user-attachments/assets/4db5c71b-edf9-44ea-a70b-8bdea0b389b9" width="400" /> \
Figure 2.A: dB scaled mel spectrogram with no augmentations.

<img src="https://github.com/user-attachments/assets/913b229b-2744-4cfc-b6cf-51e87476ef37" width="400" /> \
Figure 2.B: dB scaled mel spectrogram with augmentations (Guassian noise, random time scaling, random crop with padding, and frequency masking).


<img src="https://github.com/user-attachments/assets/9034f972-0f9c-46f2-b0b7-becf9f7b5797" width="600" /> \
Figure 3: Final model

Since these images are very high dimensional (40 mel filters x 400 STFT windows) and the training dataset size is small (only 3730 samples) the model is prone to overfitting. Without regularization, the validation accuracy would eventually fall below that of random guessing (0.125%). In order to remedy this, I took two approaches: data augmentation and regularization. I tested multiple data augmentation strategies. The combination that  worked the best was a combination of methods including adding Gaussian noise, randomly cropping audio files, and applying frequency masking [1] (Fig. 2.B.). The model was regularized through dropout layers (p = 0.5) on both linear output layers. After 400 epochs of training, the best dev accuracy was around 20%. 

## Next steps
The next thing to try is to explore other architectures such as RNNs. Regularization helped but created a problem of underfitting on the training dataset. Artificial data generation through a GAN would be an interesting avenue to explore.

## References
[1] Park, D. S., Chan, W., Zhang, Y., Chiu, C., Zoph, B., Cubuk, E. D., & Le, Q. V. (2019). SpecAugment: a simple data augmentation method for automatic speech recognition. Interspeech 2022. https://doi.org/10.21437/interspeech.2019-2680
