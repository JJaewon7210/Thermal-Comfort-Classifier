Facial Landmark Extraction and Pixel Extraction for Thermal comfort Classifier
============================

Getting Started
---------------
To use the this Project, follow these steps:

1. Clone this repository to your local machine

2. Download the dataset 'Charlotte thermalFace' with images and labels.

3. Get the 68-point face landmarks for each image and save it into one single excel file. In this project, the excel file name is `D:/ThermalData/Charlotte_ThermalFace/SPIGA_results_charlotte_RAINBOW.xlsx`

    > *<i class="fa fa-info-circle" aria-hidden="true"></i>*Note**: In this project, I used the face-landmark detection model to find 68-point face landmarks for each images. Then, I combine theses results to the orignal Charlotte thermalFace dataset to make the excel file. You need additional process to this, not included in this folder.

4. Run `extractPixel.py` to extract the skin temperature features from ROIs of face component based on the 68-point face landmark. It will make the file `'D:/ThermalData/Charlotte_ThermalFace/S_3m_one_temp.csv'`. Each row contains the data including skin temperature features from ROIs of face component.

5. Run `Classifiers.py` to train and test the machine learning models.


6. Review the results from the `Classifiers\Classifier_~.xlsx` and `Classifiers\Label_~.xlsx`  and choose the best machine learning algorithms and input features.

Folder structure
------------

```lua
src/
├── Classifiers/
│   ├── Classifier_~.xlsx
│   └── Label_~.xlsx
├── Classifiers.py
├── configs.py
├── Dataset.py
├── extractPixel.py
├── pose_analyze.py
└── .gitignore
```

- `Classifiers/`: This folder contains the results file from the `Classifiers.py`. 

    - `Classifiers\Classifier_~.xlsx` includes the performance of machine learning algorithms with hyperparameters.

    - `Classifiers\Label_~.xlsx` includes the predicted label and ground truth label one by one for each row.

- `configs.py` contains the list of specific columns used for the thermal comfort prediction analysis. This script is the dependency to run the `Dataset.py` script. Regarding the excel file that contains not only the label of 'Charlotte thermalFace' but also the extracted skin temperature features from ROIs of face component based on the 68-point face landmark, the specific columns were selected to configure the final input data features and label for training machine learning models.

- `Classifiers.py` contains a function run_classifiers_and_evaluate that trains and evaluates different machine learning classifiers using a dataset. The dataset is loaded from a CSV file (`'D:/ThermalData/Charlotte_ThermalFace/S_3m_one_temp.csv'`) and preprocessed before being split into training and testing sets. The program then trains the classifiers using the training data, predicts the labels of the testing data, and evaluates the accuracy of the predictions. Finally, the program generates an evaluation report that contains the accuracy scores of each classifier, as well as the parameters used for each classifier.

- `Dataset.py` contains a 'MyDataController' class, which can be used to preprocess and split data for machine learning models. This script is the dependency to run the `Classifiers.py` script. You should determine the two input variables for constructing 'MydataController' class. First input variable 'base_df' is the output csv file from the `extractPixel.py` script. In the code, the output csv file is representeded in '`'D:/ThermalData/Charlotte_ThermalFace/S_3m_one_temp.csv'` '. Second input variable 'split' is the number of splits for the k-fold cross-validation.

- `extractPixel.py` provides a set of functions that can be used to extract pixels from specific parts of a facial image based on 68-point face landmarks.
    - The `PART_INDICES` dictionary contains the indices of the face landmark points for various parts of the face such as nose, chin, cheek, periorbital, mouth, and the entire face.
    - `extract_polygon_points()` function extracts the polygon points for a given part of the face based on the face landmark indices provided in the `PART_INDICES` dictionary.
    - The `crop_face_element()` function creates a mask using the polygon points and applies it to the image, and returns the pixels within the polygon.
    - The `extract_pixel_by_element()` function extracts pixels from a specific part of the face based on the image, face landmark points, and the part of the face.
  - The script reads an Excel file (`D:/ThermalData/Charlotte_ThermalFace/SPIGA_results_charlotte_RAINBOW.xlsx`) using pandas library containing the 68-point face landmarks, and it extracts pixels from various parts of the face using the functions defined in the script. The extracted pixels are saved in a CSV file (`'D:/ThermalData/Charlotte_ThermalFace/S_3m_one_temp.csv'`).

- `pose_analyze.py`  loads two dataframes, one from an Excel file (`Classifiers\Label_~.xlsx`) and one from a CSV file (`'D:/ThermalData/Charlotte_ThermalFace/S_3m_one_temp.csv'`), and merges them together based on their respective indexes. It then drops specific columns from the merged dataframe, creates new columns based on the values of certain columns, and categorizes these values into ranges. The output includes counts for both the total number of rows in each range and the number of rows where a certain condition is True in each range. This sciprt is for adding the headpose to the Excel file.

License
------------

This project is licensed under the MIT License.

README from chatGPT
