# Pulse Train Data Processor and CNN Model

This Python application processes CSV files containing pulse train data, preprocesses this data, and applies a Convolutional Neural Network (CNN) model for analysis.


## Contributors
- Austin Taylor
- Jonathan Ochoa Monzon
- Kaitlyn Olinger 
- Kenneth Cisneros 
- Jason Jordan

## Features

- **Data Processing**: Efficiently processes and filters CSV data.
- **Data Preprocessing**: Reshapes data into a suitable format for CNN.
- **CNN Model**: Utilizes a Convolutional Neural Network for data analysis.
- **Performance Metrics**: Provides test loss and accuracy metrics.

## Requirements

- Python 3.x
- TensorFlow
- Keras
- Pandas
- NumPy
- scikit-learn

## Installation

1. Ensure that Python 3.x is installed.
2. Install the required libraries:
    ```
    pip install tensorflow pandas numpy scikit-learn
    ```

## Usage

1. Place your CSV data files inside the `PulseTrainData` directory.
2. Run the script:
    ```
    python <script_name>.py
    ```

### Function Descriptions

- `process_csv_files()`: Processes CSV files from a specified directory and returns a dictionary of DataFrames.
- `process_csv_to_dataframe()`: Converts a CSV file into a DataFrame, skipping initial rows and filtering buffers.
- `preprocess_data()`: Preprocesses the DataFrame for CNN input.
- `convert_to_categorical()`: Converts angle values to categorical labels.
- `create_cnn_model()`: Creates and compiles the CNN model.
- `main()`: Main function to process data, train, and evaluate the CNN model.

### CSV File Format

The expected CSV file format should have data starting from the 5th row, and any rows with 'BUFFER' in the first column should be ignored.

## Contributing

Contributions to this project are welcome. Please ensure that any pull requests are well documented and tested.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For any queries or issues, please open an issue in the project repository.
