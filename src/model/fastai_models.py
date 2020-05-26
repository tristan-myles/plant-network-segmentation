"""
fastai_models.py
Script to train fast-ai models used for binary image segmentation
"""
__author__ = "Tristan Naidoo"
__maintainer__ = "Tristan Naidoo"
__version__ = "0.0.1"
__email__ = "ndxtri015@myuct.ac.za"
__status__ = "development"

from fastai.vision import *
import matplotlib.pyplot as plt
from src.helpers.extract_dataset import chip_range
from src.model.model import Model


class FastaiUnetLearner(Model):
    def __init__(self, data_bunch: vision.data.DataBunch = None):
        if data_bunch:
            self.data_bunch = data_bunch
        else:
            self.data_bunch = None

        self.learn = None
        self.min_grad_lr = None

    def add_databunch(self, data_bunch: vision.data.DataBunch = None):
        self.data_bunch = data_bunch

    def prep_fastai_data(self, paths_df: pd.DataFrame,
                         leaf_folder_path: str,
                         batch_sizes: int,
                         split_func=ItemList.split_from_df,
                         plot: bool = False,
                         mask_col_name:str = "mask_path",
                         codes: List[int] = [0, 1]):

        segmentation_image_list = (SegmentationItemList
                                   .from_df(paths_df, leaf_folder_path))
        segmentation_image_list = split_func(segmentation_image_list)
        segmentation_image_list = segmentation_image_list.label_from_df(
            mask_col_name, classes=codes)

        # Note no transformations
        data_bunch = (segmentation_image_list
                      .databunch(bs=batch_sizes)
                      .normalize(imagenet_stats))

        if plot:
            data_bunch.show_batch(rows=2, figsize=(10, 7))
            plt.show()

        self.data_bunch = data_bunch

    def create_learner(self, data_bunch: vision.data.DataBunch = None,
                       model: vision.models = models.resnet18):
        if data_bunch is None:
            data_bunch = self.data_bunch

        self.learn = unet_learner(data_bunch, model)

    def train(self, epochs: int, save: bool, save_path: str = None,
              lr: float = None):
        if lr is None:
            if self.min_grad_lr:
                lr = self.min_grad_lr
            else:
                raise ValueError("No lr provided and min_grad_lr "
                                 "is also none")
        self.learn.fit_one_cycle(epochs, lr)

        if save:
            self.learn.save(save_path, return_path=True)
            self.learn.export(f'{save_path}.pkl')

    def find_lr(self, learn: vision.learner):
        self.learn.lr_find()
        self.learn.recorder.plot(suggestion=True)
        # suggested LR is the point at which the gradient is the steepest
        self.min_grad_lr = learn.recorder.min_grad_lr

    def load_weights(self, path: str):
        self.learn.load(path)

    def predict_tile(self, new_tile: np.array = None, tile_number: int = None):
        if tile_number:
            input_image = self.data.x[tile_number]
        elif new_tile is not None:
            input_image = Image(
                pil2tensor(new_tile, dtype=np.float32).div_(255))
            # confirm the roll of div_(255)
            _, val_tfms = get_transforms()
            input_image = input_image.apply_tfms(
                val_tfms.append(normalize_funcs(*imagenet_stats)),
                input_image,size=new_tile.shape)
        else:
            raise Exception("No input image :(")

        prediction = self.learn.predict(input_image)

        return prediction

    def predict_full_leaf(self, x_length: int, y_length: int,
                          x_tile_length: int, y_tile_length: int):

        prediction = np.zeros((y_length, x_length))
        counter = 0

        for y_range in chip_range(0, y_length, y_tile_length):
            for x_range in chip_range(0, x_length, x_tile_length):
                pred_tile = self.predict_tile(self.data.x[counter])
                pred_tile = (pred_tile[0].px.numpy() * 255)
                pred_tile = pred_tile.reshape(y_tile_length, x_tile_length)

                if ((y_range[1] - y_range[0]) != y_tile_length or
                        (x_range[1] - x_range[0]) != x_tile_length):
                    pred_tile = pred_tile[0:(y_range[1]-y_range[0]), 0:(x_range[1]-x_range[0])]

                prediction[y_range[0]:y_range[1], x_range[0]:x_range[1]] = \
                    pred_tile

                counter += 1

        return prediction

    def plot_leaf(self, image_index: int, plot_pred: bool = False,
                  prediction: vision.image.ImageSegment = None):

        if plot_pred:
            rows = 2
        else:
            rows = 1
        # Code obtained from: https://docs.fast.ai/vision.image.html
        _, axs = plt.subplots(rows, 3, figsize=(20, 15))
        axs = axs.flatten()
        self.data.x[image_index].show(ax=axs[0], title='no mask')
        self.data.x[image_index].show(ax=axs[1], y=self.data.y[image_index],
                                      title='masked', cmap="gray")
        # Masks are quite small could be the reason it looks weird
        self.data.y[image_index].show(ax=axs[2], title='mask only',
                                      alpha=1.,
                                      cmap="gray")
        if plot_pred:
            self.data.x[image_index].show(ax=axs[3], title='no mask')
            self.data.x[image_index].show(ax=axs[4],
                                          y=prediction[0],
                                          title='masked', cmap="gray")
            # Masks are quite small could be the reason it looks weird
            prediction[0].show(ax=axs[5], title='mask only',
                               alpha=1., cmap="gray")
        plt.show()


if __name__ == "__main__":
    fai_unet_learner = FastaiUnetLearner()

    temp_df = pd.read_csv("/home/tristan/Documents/MSc_Dissertation/"
                          "plant-network-segmentation/out_leaf_train.csv")
    temp_df.leaf_name = temp_df.leaf_name.apply(
        lambda x: str.rsplit(x, "/", 1)[1])
    paths_df = pd.DataFrame({"path": temp_df.leaf_name[2408: 2471],
                             "mask_path": temp_df.mask_name[2408: 2471]})

    fai_unet_learner.prep_fastai_data(
        paths_df,
        "/mnt/disk3/thesis/data/1_qk3/tristan/diff-chips/",
        4, plot=True)
    fai_unet_learner.create_learner()

    fai_unet_learner.load_weights(
        "/home/tristan/Documents/MSc_Dissertation/plant-network-segmentation"
        "/mini_train")
    prediction = fai_unet_learner.predict_tile(tile_number=4)
    fai_unet_learner.plot_leaf(4, True, prediction)

    fai_unet_learner.predict_full_leaf(2616, 1949, 300, 300)
