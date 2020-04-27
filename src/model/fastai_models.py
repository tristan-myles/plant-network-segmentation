"""
fastai_models.py
Script to train models used for binary image segmentation
"""
__author__ = "Tristan Naidoo"
__maintainer__ = "Tristan Naidoo"
__version__ = "0.0.1"
__email__ = "ndxtri015@myuct.ac.za"
__status__ = "development"

from fastai.vision import *
import matplotlib.pyplot as plt


class fastaiUnetLearner():
    def __init__(self):
        self.data = None
        self.learn = None
        self.min_grad_lr = None

    def prep_fastai_data(self, paths_df: pd.DataFrame,
                         leaf_folder_path: str, batch_sizes: int,
                         plot: bool = False,
                         codes: List[int] = [0, 1]):
        segmentation_image_list = (SegmentationItemList
                                   .from_df(paths_df, leaf_folder_path)
                                   .split_none()
                                   .label_from_df("mask_path", classes=codes))

        # Note no transformations
        data = (segmentation_image_list
                .databunch(bs=batch_sizes)
                .normalize(imagenet_stats))

        if plot:
            data.show_batch(rows=2, figsize=(12, 9))
            plt.show()

        self.data = data

    def create_learner(self, data: vision.data.DataBunch = None,
                       model: vision.models = models.resnet18):
        if data is None:
            data = self.data

        self.learn = unet_learner(data, model)

    def train_unet_fastai(self, epochs: int,
                          lr: float,
                          save: bool, save_path: str):
        if lr is None:
            lr = self.min_grad_lr
        self.learn.fit_one_cycle(epochs, lr)
        if save:
            self.learn.save(save_path)

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
            input_image = new_tile
        else:
            raise Exception("No input image :(")

        prediction = self.learn.predict(input_image)

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
    fai_unet_learner = fastaiUnetLearner()

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
