# Alzheimer's Disease Image Classification

This project aims to classify brain MRI images into four categories: MildDemented, ModerateDemented, NonDemented, and VeryMildDemented. It utilizes TensorFlow and Keras for deep learning-based image classification.

## Requirements
- Python 3.x
- TensorFlow 2.x
- scikit-learn
- matplotlib
- pandas

## Installation
1. Clone this repository to your local machine.
2. Place your dataset in the `data` directory. Ensure that your dataset is structured appropriately with subdirectories for each category.

## Usage
1. Modify the `dataPath` variable in the script to point to your dataset directory.
2. Ensure that your dataset is structured with subdirectories for each category (`MildDemented`, `ModerateDemented`, `NonDemented`, `VeryMildDemented`).
3. Run the script `train.py`.
4. The model will be trained, and the training progress will be displayed.
5. After training, the model will be saved as `AlzModel.keras` in the current directory.
6. Training and validation loss/accuracy plots will be displayed to evaluate the model's performance.
7. To load and evaluate the model using a Jupyter Notebook, open `loader.ipynb`.


## File Description
- `train.py`: Python script containing the code for data preprocessing, model training, and evaluation.
- `loader.ipynb`: Jupyter Notebook for loading the saved model and evaluating its performance.
- `AlzModel.keras`: Trained model saved in Keras format.
- `README.md`: Documentation file.

## Acknowledgments
- Special thanks to Dr. Fangning Hu for supervision and guidance throughout this project.
- The dataset used in this project is from [Kaggle](https://kaggle.com).


## Author
Raey Getahun

## License
This project is licensed under the [MIT License](LICENSE).
