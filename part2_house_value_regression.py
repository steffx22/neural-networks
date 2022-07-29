from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics, preprocessing
from part1_nn_lib import Preprocessor, PreprocessorZStandardisation
from part1_nn_lib import MultiLayerNetwork
from part1_nn_lib import Trainer
from enum import Enum

import pickle
import numpy as np
import pandas as pd


class Normaliser(Enum):
    MIN_MAX_NORMALISATION = 1
    Z_NORMALISATION = 2


class Regressor(BaseEstimator, ClassifierMixin):

    def __init__(
            self,
            x,
            nb_epoch=100,
            learning_rate=0.01,
            batch_size=512,
            activation='tanh',
            layer_numbers=3,
            layer_size=10,
            normaliser=Normaliser.MIN_MAX_NORMALISATION
    ):
        """
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epoch to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        self.x = x

        # Initialize the mapping
        self.mapping = dict()

        # Initialize the normalisers
        self.preprocessor_x = None
        self.preprocessor_y = None

        self.normaliser = normaliser

        # process the input x
        processed_x, _ = self._preprocessor(x, training=True)

        self.input_size = processed_x.shape[1]

        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.activation = activation
        self.layer_numbers = layer_numbers
        self.layer_size = layer_size
        self.trainer = None
        self.neurons = None
        self.activations = None

    def _preprocessor(self, x, y=None, training=False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size).
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Reset the indexes of the DataFrame
        # This is important later in the code where x and y might have different indexes
        x = x.reset_index().iloc[:, 1:]

        if y is not None:
            y = y.reset_index().iloc[:, 1:]

        # Fill the missing values with the mean on the column
        filled_dataset = x.fillna(x.mean(numeric_only=True, axis='index'))

        lb = preprocessing.LabelBinarizer()

        if training:
            # Get the column with the strings
            cols_strings = filled_dataset['ocean_proximity']

            # Get the unique labels and sort them (sorting will remove inconsistencies
            # caused by the order in which the one hot vectors are created
            unique_strings = cols_strings.drop_duplicates()
            unique_strings = unique_strings.sort_values()

            lb.fit(cols_strings)

            # Get the one hot vectors for every string in the column ocean_proximity
            one_hot_vectors = lb.transform(cols_strings)

            # Add column names to one hot vectors
            # This will be important later when we do pd.concat() because the other
            # columns already have the right label
            one_hot_vectors_df = pd.DataFrame(one_hot_vectors, columns=unique_strings)

            # Get the one hot vectors for every unique string
            # This will be used to create the mapping
            unique_one_hot_vectors = lb.transform(unique_strings)

            # Create the mapping with the key being a unique string and the value being
            # its corresponding one hot vector defined in unique_one_hot_vectors
            for unique_string, one_hot_vector in zip(unique_strings, unique_one_hot_vectors):
                self.mapping[unique_string] = one_hot_vector

            # Remove the string column
            dataset_without_strings = filled_dataset.drop(columns=['ocean_proximity'])

            # Add the one hot vectors so that they replace the string column
            processed_x = pd.concat([dataset_without_strings, one_hot_vectors_df], axis=1)

            # Initialise the normaliser for x
            if self.normaliser == Normaliser.MIN_MAX_NORMALISATION:
                self.preprocessor_x = Preprocessor(processed_x.to_numpy())
            elif self.normaliser == Normaliser.Z_NORMALISATION:
                self.preprocessor_x = PreprocessorZStandardisation(processed_x.to_numpy())

            # Normalise x
            normalised_x = self.preprocessor_x.apply(processed_x.to_numpy())

            if y is not None:
                # Initialise the normaliser for y
                if self.normaliser == Normaliser.MIN_MAX_NORMALISATION:
                    self.preprocessor_y = Preprocessor(y.to_numpy())
                elif self.normaliser == Normaliser.Z_NORMALISATION:
                    self.preprocessor_y = PreprocessorZStandardisation(y.to_numpy())

                # Normalise y
                normalised_y = self.preprocessor_y.apply(y.to_numpy())

                return normalised_x, normalised_y
            return normalised_x, y
        else:
            # Get the column with the strings
            cols_strings = filled_dataset['ocean_proximity']

            # Use the mapping defined during training to get the same one hot vectors
            one_hot_vectors = []
            for col_string in cols_strings:
                one_hot_vectors.append(self.mapping[col_string])

            # Get the unique strings from the mapping
            unique_strings = self.mapping.keys()

            # Add column names to one hot vectors
            one_hot_vectors_df = pd.DataFrame(one_hot_vectors, columns=unique_strings)

            # Remove the string column
            dataset_without_strings = filled_dataset.drop(columns=['ocean_proximity'])

            # Add the one hot vectors so that they replace the string column
            processed_x = pd.concat([dataset_without_strings, one_hot_vectors_df], axis=1)

            # Normalise x
            normalised_x = self.preprocessor_x.apply(processed_x.to_numpy())

            if y is not None:
                # Normalise y
                normalised_y = self.preprocessor_y.apply(y)

                return normalised_x, normalised_y
            return normalised_x, y

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Initialise the activations to be
        # [self.activation, ..., self.activation, identity]
        self.activations = list(np.repeat(self.activation, self.layer_numbers))
        self.activations.append('identity')

        # Initialise the neurons to be of form [self.layer_size, ..., self.layer_size, 1]
        self.neurons = list(np.repeat(self.layer_size, self.layer_numbers))
        self.neurons.append(1)

        # Initialise the network
        multi_layer_network = MultiLayerNetwork(
            self.input_size, neurons=self.neurons, activations=self.activations
        )

        # Initialise the trainer
        self.trainer = Trainer(
            network=multi_layer_network,
            batch_size=self.batch_size,
            nb_epoch=self.nb_epoch,
            learning_rate=self.learning_rate,
            loss_fun="mse",
            shuffle_flag=True,
        )

        print("nb_epoch = " + str(self.nb_epoch))
        print("learning_rate = " + str(self.learning_rate))
        print("batch_size = " + str(self.batch_size))
        print("activation = " + str(self.activation))
        print("layer_numbers = " + str(self.layer_numbers))
        print("layer_size = " + str(self.layer_size))
        print("normaliser = " + str(self.normaliser))
        print()

        # Process the data
        processed_x, processed_y = self._preprocessor(x, y=y, training=True)

        # Train the trainer
        self.trainer.train(processed_x, processed_y)

        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def predict(self, x):
        """
        Ouput the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.darray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Process the data
        X, _ = self._preprocessor(x, training=False)

        # Get the predicted output and denormalise it
        return self.preprocessor_y.revert(self.trainer.network(X))

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw ouput array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Predict the output
        y_pred = self.predict(x)

        # Compute the mean absolute error using the predicted output and the target
        mean_error = metrics.mean_absolute_error(y_pred, y)

        return mean_error

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_regressor(trained_model):
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor():
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model


def RegressorHyperParameterSearch(regressor, train_x, train_y):
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    # The model that GridSearchCV is using is our regressor, which implements
    # ClassifierMixin
    model = regressor

    # This dictionary stores all the combinations we want to try in order to get the best
    # model
    grid = dict(
        nb_epoch=[20, 100, 600, 1000],
        learning_rate=[0.05, 0.01, 0.001, 0.0001],
        batch_size=[512, 1024],
        activation=['relu', 'sigmoid', 'tahn'],
        layer_numbers=[5],
        layer_size=[10, 20],
        normaliser=[Normaliser.MIN_MAX_NORMALISATION, Normaliser.Z_NORMALISATION]
    )

    # Initialise the grid search
    randomSearch = GridSearchCV(
        estimator=model,
        param_grid=grid,
        scoring='neg_mean_absolute_error'
    )

    # The grid search runs fit for the model with every combination of the parameters
    # defined in grid
    searchResult = randomSearch.fit(train_x, train_y)
    bestModel = searchResult.best_estimator_

    print("\nFound best model:")

    # Print the best score
    print("R2: {:.2f}".format(bestModel.score(train_x, train_y)))

    # Print the best model
    print(searchResult.best_params_)

    return searchResult.best_params_

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################


def example_main():
    output_label = "median_house_value"

    data = pd.read_csv("housing.csv")

    # Split the data in train and test data with the percentage 0.2
    data_train, data_test = train_test_split(data, test_size=0.2)

    # We need to use reset index to avoid inconsistencies between the indexes of x and y
    # that could appear in _preprocessor()
    x_train = data_train.loc[:, data_train.columns != output_label].reset_index().iloc[:, 1:]
    y_train = data_train.loc[:, [output_label]].reset_index().iloc[:, 1:]

    x_test = data_test.loc[:, data_test.columns != output_label].reset_index().iloc[:, 1:]
    y_test = data_test.loc[:, [output_label]].reset_index().iloc[:, 1:]

    # Initialise the regressor, with the parameters set for the best model that we found
    # using our RegressorHyperParameterSearch
    regressor = Regressor(
        x_train,
        nb_epoch=1000,
        learning_rate=0.01,
        batch_size=512,
        activation='tanh',
        layer_numbers=10,
        layer_size=10,
        normaliser=Normaliser.MIN_MAX_NORMALISATION
    )

    # Comment these 2 lines if you want to run the regressor on the sets defined above
    # Uncomment them if you want to find the best model
    # RegressorHyperParameterSearch(regressor, x_train, y_train)
    # return

    # Train the regressor
    regressor = regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # Print the score
    mean_error = regressor.score(x_test, y_test)
    print("Mean error: " + str(mean_error))


if __name__ == "__main__":
    example_main()
