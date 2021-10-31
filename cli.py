from abc import abstractmethod


class ModelClient:
    """
    Base client for interacting with generated models.
    """
    def __init__(self):
        self.model = self.__load_model__()
        pass

    @abstractmethod
    def __train_data__(self):
        """
        Returns training data.
        :return: Any
        """
        pass

    @abstractmethod
    def __test_data__(self):
        """
        Returns testing data.
        :return: Any
        """
        pass

    @abstractmethod
    def __model_from_path__(self):
        """
        Loads model from a directory.
        :return: model
        """
        pass

    @abstractmethod
    def __build_and_save__(self):
        """
        Builds and saves model.
        :return: model
        """
        pass

    def __load_model__(self):
        """
        Attempts to load model from path. If model does not exist a new
        one will be generated.
        :return: model
        """
        try:
            return self.__model_from_path__()
        except:
            return self.__build_and_save__()

    @abstractmethod
    def predict(self, content):
        """
        Preforms a prediction based on the passed in content.
        """
        pass
