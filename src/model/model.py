from abc import ABC, abstractmethod


class Model(ABC):
    @abstractmethod
    def predict_tile(self, new_tile):
        """
         Implementation of this method should take in a tile through argument
         new_tile and return a prediction using the model

        :param new_tile: input tile
        :return: prediction
        """
        pass
