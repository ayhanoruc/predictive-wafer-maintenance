import pandas as pd
import numpy as np

from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek
from sklearn.base import BaseEstimator, TransformerMixin

import os, sys


from src.exception_handler import CustomException, handle_exceptions
from src.log_handler import AppLogger
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import (
    DataValidationArtifact,
    DataTransformationArtifact,
)
from src.utility.generic import save_numpy_array_data, save_object


# valid_train_dataset_dir = "../valid_feature_store/valid_training_data/" # test purpose
# valid_predict_dataset_dir = "../valid_feature_store/valid_predict_data/" # test purpose


class TrainingPreprocessor(BaseEstimator, TransformerMixin):
    """
    Custom data preprocessor.

    This class is responsible for preprocessing training data, including handling missing values,
    scaling features, and addressing class imbalance.
    Args:
        use_y (bool, optional): Whether to use the 'y' values. Defaults to True.
        important_cols (list, optional): List of important columns to keep. Defaults to None.

    Methods:
        fit(X, y=None):
            Fit the preprocessor to the training data.

        transform(X, is_testing=False):
            Transform the training data and optionally the testing data.

    """

    def __init__(self, use_y=True, important_cols=None):
        if important_cols != None:  # or if important_cols:
            self.important_cols = important_cols
        self.use_y = use_y

    def fit(self, X, y=None):
        if (
            self.use_y
        ):  # if preprocessor is initialized with use_t = True: simply take y as an attribute.
            self.y = y

        return self

    def transform(self, X, is_testing=False):
        """
        Transform the input data.

        This method applies a series of data transformations to the input data, including:
        - Dropping columns with zero standard deviation.
        - Dropping duplicated columns.
        - Selecting important columns if specified.
        - Handling missing values using a constant fill strategy.
        - Scaling the data using RobustScaler.

        Args:
            X (pd.DataFrame): The input DataFrame to be transformed.
            is_testing (bool, optional): Whether the data is for testing. Defaults to False.

        Returns:
            pd.DataFrame or tuple: If `is_testing` is False, it returns a tuple containing the transformed feature matrix (X_transformed)
            and target labels (self.y). If `is_testing` is True, it returns only the transformed feature matrix (X_transformed).
        """
        X_transformed = DataTransformationComponent.drop_zero_std(X)
        X_transformed = DataTransformationComponent.drop_duplicated_cols(X_transformed)

        if self.important_cols != None:
            X_transformed = X_transformed[self.important_cols]

        imputer = SimpleImputer(
            strategy="constant", fill_value=0
        )  # comes from EDA, but its not a good practice to hardcode this selection/decision
        X_transformed = imputer.fit_transform(X_transformed)

        if not is_testing:
            # Handle class imbalance only for training data to avoid leakage
            X_transformed, self.y = DataTransformationComponent.handle_imbalance(
                X_transformed, self.y
            )
            r_scaler = RobustScaler()
            X_transformed = r_scaler.fit_transform(X_transformed)
            return X_transformed, self.y

        # if testing:
        r_scaler = RobustScaler()
        X_transformed = r_scaler.fit_transform(X_transformed)

        return X_transformed


class DataTransformationComponent:

    """
    Component for data transformation operations.

    This class performs various data transformation tasks on training data, including
    handling missing values, dropping zero standard deviation columns, dropping highly correlated
    columns, and addressing class imbalance.

    Args:
        data_transformation_config (DataTransformationConfig): Configuration for data transformation.
        data_validation_artifact (DataValidationArtifact): Artifact from data validation.

    Attributes:
        data_transformation_config (DataTransformationConfig): Configuration for data transformation.
        data_validation_artifact (DataValidationArtifact): Artifact from data validation.
        valid_train_dataset_dir (str): Directory path to the valid training dataset.
        important_cols (list): List of important columns.

    Methods:
        - restore_original_data(): Restore the original training data.
        - drop_zero_std(dataframe): Drop columns with zero standard deviation.
        - drop_highly_correlated_columns(dataframe, corr_threshold=0.95, count_threshold=3):
          Drop highly correlated columns based on correlation thresholds.
        - drop_duplicated_cols(dataframe): Drop duplicated columns.
        - handle_imbalance(X, y): Handle class imbalance using SMOTETomek.
        - create_train_test(dataframe): Create train and test sets.
        - set_important_cols(dataframe): Set important columns based on target correlation.
        - get_preprocessor(): Get the training data preprocessor.
        - run_data_transformation(): Perform data transformation and return an artifact.

    For detailed information on each method, refer to the individual method docstrings.
    """

    def __init__(
        self,
        data_transformation_config: DataTransformationConfig,
        data_validation_artifact: DataValidationArtifact,
    ):
        self.data_transformation_config = data_transformation_config
        self.data_validation_artifact = data_validation_artifact
        # self.valid_train_dataset_dir = "./valid_feature_store/valid_training_data/" # test purpose
        self.important_cols = None

        self.valid_train_dataset_dir = (
            self.data_validation_artifact.valid_data_dir
        )  # <- added

        self.log_writer = AppLogger("DataTransformation")

    @handle_exceptions
    def restore_original_data(
        self,
    ) -> pd.DataFrame:
        """
        Restore the original training data by merging CSV files and preprocessing.

        This method reads CSV files from the specified directory containing valid training data,
        merges them into a single DataFrame, and performs preprocessing steps such as dropping
        unnecessary columns and converting the 'Good/Bad' target column to binary labels (1 for 'Good' and 0 for 'Bad').

        Returns:
            pd.DataFrame: A DataFrame containing the restored and preprocessed training data.
        """

        csv_file_list = os.listdir(self.valid_train_dataset_dir)
        df_merged = pd.DataFrame()
        for file in csv_file_list:
            file_path = os.path.join(self.valid_train_dataset_dir, file)
            df = pd.read_csv(file_path)
            df_merged = pd.concat(
                objs=[df_merged, df], ignore_index=True
            )  # merged around axis=0
            df_merged.drop(columns=["Wafer"], inplace=True)
            filt = df_merged["Good/Bad"] == 1
            df_merged["Good/Bad"] = np.where(filt, 1, 0)

        return df_merged

    @staticmethod
    @handle_exceptions
    def drop_zero_std(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Drop columns with zero standard deviation from a DataFrame.

        This static method takes a DataFrame as input and identifies columns with zero
        standard deviation (constant columns). It then removes these columns from the DataFrame
        and returns the modified DataFrame without the constant columns.

        Args:
            dataframe (pd.DataFrame): The input DataFrame to remove constant columns from.

        Returns:
            pd.DataFrame: A DataFrame with constant columns removed.

        """
        zero_std_cols = dataframe.columns[dataframe.std() == 0]
        dataframe2 = dataframe.drop(columns=zero_std_cols)
        return dataframe2

    @handle_exceptions
    def drop_highly_correlated_columns(
        self, dataframe: pd.DataFrame, corr_threshold=0.95, count_threshold=3
    ):
        """
        Drop highly correlated columns based on specified correlation thresholds.

        This method identifies highly correlated columns in the input DataFrame and drops them
        based on the provided correlation thresholds. Columns are considered highly correlated if
        their absolute correlation coefficient exceeds the given `corr_threshold`, and if they
        are correlated with more than `count_threshold` other columns.

        Args:
            dataframe (pd.DataFrame): The input DataFrame to process.
            corr_threshold (float, optional): The correlation threshold (between 0 and 1) to consider columns as highly correlated.
                Defaults to 0.95.
            count_threshold (int, optional): The count threshold for the number of correlated columns to consider a column as highly correlated.
                Defaults to 3.

        Returns:
            pd.DataFrame: A DataFrame with highly correlated columns removed.
        """
        corr_matrix = dataframe.corr(method="pearson")
        filt = (abs(corr_matrix) > corr_threshold) & (abs(corr_matrix) < 1.00)
        corr_counts = filt.sum(axis=1)
        # highly_correlated = corr_counts[cor_counts > count_threshold].sort_values(ascending=False)
        highly_correlated_cols = dataframe.columns[corr_counts > count_threshold]

        return dataframe.drop(columns=highly_correlated_cols)

    @staticmethod
    @handle_exceptions
    def drop_duplicated_cols(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Drop duplicated columns from the input DataFrame.

        This method identifies and drops duplicated columns in the input DataFrame, keeping only the first occurrence of each column.

        Args:
            dataframe (pd.DataFrame): The input DataFrame to process.

        Returns:
            pd.DataFrame: A DataFrame with duplicated columns removed, keeping the first occurrence of each.
        """
        duplicated_cols = dataframe.T[dataframe.T.duplicated()].index
        return dataframe.drop(columns=duplicated_cols)

    @staticmethod
    @handle_exceptions
    def handle_imbalance(X, y):
        """
        Handle class imbalance in the dataset using SMOTETomek.

        This method uses the SMOTETomek technique to handle class imbalance in the dataset. It oversamples the minority class
        using SMOTE and then cleans the resulting dataset using Tomek links to remove potential noisy samples.

        Args:
            X (array-like or pd.DataFrame): The feature matrix of the dataset.
            y (array-like or pd.Series): The target labels of the dataset.

        Returns:
            tuple: A tuple containing the resampled feature matrix (X_resampled) and the corresponding target labels (y_resampled).
        """
        smt = SMOTETomek(random_state=11, sampling_strategy="minority")
        return smt.fit_resample(X, y)

    @handle_exceptions
    def create_train_test(self, dataframe):
        """
        Create train and test sets from the given DataFrame.

        This method splits the input DataFrame into train and test sets for machine learning.
        It separates the feature matrix (X) from the target labels (y) and ensures that the class distribution
        is preserved by using stratified sampling.

        Args:
            dataframe (pd.DataFrame): The DataFrame containing both features and target labels.

        Returns:
            tuple: A tuple containing the following elements:
                - X_train (pd.DataFrame): The feature matrix for the training set.
                - X_test (pd.DataFrame): The feature matrix for the testing set.
                - y_train (pd.Series): The target labels for the training set.
                - y_test (pd.Series): The target labels for the testing set.
        """
        X = dataframe.drop("Good/Bad", axis="columns")
        y = dataframe["Good/Bad"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=11, stratify=y
        )

        return X_train, X_test, y_train, y_test

    @handle_exceptions
    def set_important_cols(self, dataframe: pd.DataFrame) -> list:
        """
        Set the list of important columns based on correlation with the target.

        This method calculates the correlation between each feature column and the target column 'Good/Bad'
        in the provided DataFrame. It selects the top 200 features with the highest absolute correlation
        and stores their column names as the list of important columns.

        Args:
            dataframe (pd.DataFrame): The DataFrame containing feature columns and the 'Good/Bad' target column.

        Raises:
            CustomException: If an error occurs during correlation calculation or handling exceptions.
        """
        df_corr = pd.DataFrame()
        df_corr["Sensor_id"] = dataframe.columns[:-1].tolist()
        corr_score = [
            abs(round(dataframe[[col, "Good/Bad"]].corr().iloc[0, 1], 4))
            for col in dataframe.columns[:-1]
        ]
        df_corr["corr"] = corr_score
        self.important_cols = df_corr.sort_values(by="corr", ascending=False)[:200][
            "Sensor_id"
        ].to_list()

    @handle_exceptions
    def get_preprocessor(self):
        """
        Get the training data preprocessor.

        This method returns an instance of the 'TrainingPreprocessor' class configured with the list of
        important columns. The preprocessor is used for data transformation during training/prediction pipeline phases.

        Returns:
            TrainingPreprocessor: An instance of the data preprocessor.
        """

        preprocessor = TrainingPreprocessor(important_cols=self.important_cols)

        return preprocessor

    def run_data_transformation(
        self,
    ) -> DataTransformationArtifact:
        """
        Perform data transformation on the training dataset and return an artifact.

        This method executes a series of data transformation steps on the training dataset:
        1. Restores the original data by merging CSV files and preprocessing.
        2. Splits the data into training and testing sets.
        3. Identifies and sets important columns based on correlation with the target.
        4. Initializes and fits the training data preprocessor.
        5. Transforms the train and test dataframes using the preprocessor.
        6. Saves the transformed datasets and preprocessor object to specified file paths.
        7. Returns a DataTransformationArtifact containing file paths.

        Returns:
            DataTransformationArtifact: An artifact containing file paths for transformed data and the preprocessor.
        """

        self.log_writer.handle_logging(
            "-------------ENTERED DATA TRANSFORMATION STAGE------------"
        )

        train_df_merged = self.restore_original_data()
        self.log_writer.handle_logging("Dataset gathered.")

        X_train, X_test, y_train, y_test = self.create_train_test(train_df_merged)
        self.log_writer.handle_logging("Train/test split applied succesfully.")

        self.set_important_cols(train_df_merged)
        preprocessor = self.get_preprocessor()
        self.log_writer.handle_logging(
            "Preprocessor pipeline object initialized succesfully."
        )

        preprocessor_obj = preprocessor.fit(X_train, y_train)
        self.log_writer.handle_logging(
            "Preprocessor object fitted succesfully. mean_ and scale_ are stored inside the object."
        )

        X_train_transformed, y_train_transformed = preprocessor_obj.transform(
            X_train, is_testing=False
        )
        X_test_transformed = preprocessor_obj.transform(X_test, is_testing=True)
        self.log_writer.handle_logging(
            "Train/test dataframes are transformed succesfully!"
        )

        # save numpy array data
        train_arr = np.c_[np.array(X_train_transformed), np.array(y_train_transformed)]
        test_arr = np.c_[np.array(X_test_transformed), np.array(y_test)]

        save_numpy_array_data(
            self.data_transformation_config.data_transformation_transformed_train_file_path,
            array=train_arr,
        )

        save_numpy_array_data(
            self.data_transformation_config.data_transformation_transformed_test_file_path,
            array=test_arr,
        )
        self.log_writer.handle_logging("Transformed datasets saved succesfully!")

        # save transformer object
        save_object(
            self.data_transformation_config.data_transformation_object_file_path,
            obj=preprocessor_obj,
        )
        self.log_writer.handle_logging("Transformation object saved succesfully!")

        # return transformation artifact
        data_transformation_artifact = DataTransformationArtifact(
            transformed_train_file_path=self.data_transformation_config.data_transformation_transformed_train_file_path,
            transformed_test_file_path=self.data_transformation_config.data_transformation_transformed_test_file_path,
            preprocessor_object_file_path=self.data_transformation_config.data_transformation_object_file_path,
        )
        self.log_writer.handle_logging(
            "DataTransformation artifact updated succesfully!"
        )

        return data_transformation_artifact
