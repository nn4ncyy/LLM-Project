# LLM Project

## Project Task
In this project, I focused on performing sentiment analysis using DistilBERT, a lighter version of BERT, to classify movie reviews from the IMDB dataset as either positive or negative. The task involves training a model that can accurately predict the sentiment (positive or negative) of a given text (movie review).

## Dataset
The dataset used in this project is the IMDB dataset from Hugging Face's datasets library.The dataset is well-known in sentiment analysis tasks and is widely used for benchmarking models. The goal was to classify the reviews based on the sentiment expressed in the text.
## Pre-trained Model
For this project, I selected the DistilBERT model for sentiment classification. DistilBERT is a smaller, faster, and more efficient version of BERT, retaining 97% of BERT's language understanding capabilities while being 60% faster and 30% smaller. This makes it an ideal choice for sentiment analysis tasks where resource efficiency is important.
I utilized the distilbert-base-uncased pre-trained model, which was fine-tuned on a large body of text, allowing it to effectively handle the nuances of natural language.

Throughout the project, the GPU usage was the main challenge to navigate. After trial and error and playing around, I found that distilbert was the model i was going to use.
The reasons for selecting DistilBERT are:

Speed and Efficiency: Compared to larger models like BERT, DistilBERT offers a good trade-off between accuracy and computational efficiency, making it suitable for scenarios with limited GPU resources.

Proven Performance: DistilBERT has been widely tested and shown to perform well on tasks like sentiment analysis with less resource consumption than BERT.

## Performance Metrics
The performance of the model was evaluated using the following metrics:

Accuracy: The proportion of correctly predicted labels (positive or negative).
F1-Score: The harmonic mean of precision and recall, offering a more balanced measure of performance, especially when dealing with imbalanced classes.

### Results:
Epoch 1:
- Training Loss: 0.4175
- Validation Loss: 0.3747
- Accuracy: 84.5%
- F1-Score: 84.47%

Epoch 2:
- Training Loss: 0.2307 (significant improvement)
- Validation Loss: 0.5010 (an increase)
- Accuracy: 85.0% (a slight improvement)
- F1-Score: 85.01% (a slight improvement)

These metrics were computed using the evaluate library, and the model’s performance was reported on the validation set after each training epoch.
Accuracy improved from 84.5% to 85% from Epoch 1 to Epoch 2, showing that the model is learning effectively and generalizing somewhat well.

F1-Score also improved from 84.47% to 85.01%, indicating that the model is not just accurate but is also balancing precision and recall well.

The relatively small gap between training loss and validation loss suggests that the model is not overfitting and is generalizing well to unseen data. Though there is a slight increase in validation loss, next iterations may involve decreasing the learning rate to help combat any overfitting that may be occurring.


## Hyperparameters
Several key hyperparameters were optimized to ensure the best performance for the sentiment analysis task:

**Learning Rate: **A learning rate of 2e-5 was used. This is a standard learning rate for fine-tuning transformer-based models like DistilBERT. Fine-tuning with a lower learning rate ensures that the model adjusts its weights gradually, improving generalization.

**Batch Size:** A batch size of 4 was used to minimize GPU memory usage while still ensuring efficient training. Reducing the batch size helps avoid running out of GPU memory and provides a better trade-off for small datasets.

**Epochs:** The model was trained for 2 epochs. Since this is a relatively simple task and the dataset is not excessively large, 2 epochs were sufficient to achieve good performance without overfitting.

**Evaluation Strategy:** The evaluation was performed at the end of each epoch to ensure consistent monitoring of the model’s performance on the validation set.

**Mixed Precision Training:** fp16=True was used to leverage mixed precision, which reduces memory usage and speeds up training without significantly affecting the model's performance.


