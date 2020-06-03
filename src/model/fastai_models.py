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

        # y_idx = 1 and weighted binary by default
        self.learn = unet_learner(
            data_bunch, model, metrics=
            [Precision(), Recall(), FBeta(beta=1), accuracy])

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
                    pred_tile = pred_tile[0:(y_range[1]-y_range[0]),
                                0:(x_range[1]-x_range[0])]

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


# Copied from fastai git repo:
# https://github.com/fastai/fastai/blob/master/fastai/metrics.py
@dataclass
class ConfusionMatrix(Callback):
    """Computes the confusion matrix."""
    # The location of the target (y) is difference for classification and
    # semantic segmentation (-1 in the case of classification and 1 in the
    # case of semantic segmentation). See:
    # https://github.com/dronedeploy/dd-ml-segmentation-benchmark/blob/
    # 284f729899495b2ea853d3c364155dd0d5cae56e/libs/util.py#L147
    # and https://forums.fast.ai/t/
    # confusionmatrix-metrics-dont-work-for-semantic-segmentation/45166
    y_idx = 1

    def on_train_begin(self, **kwargs):
        self.n_classes = 0

    def on_epoch_begin(self, **kwargs):
        self.cm = None

    def on_batch_end(self, last_output: Tensor, last_target: Tensor, **kwargs):
        preds = last_output.argmax(self.y_idx).view(-1).cpu()
        # argmax of the predicted probablites, i.t.o shapes:
        # [batch_size, num_classes, tile_len1, tile_len2] -> [batch_size, 1,
        # tile_len1, tile_len2] ... view(-1) squashes this all
        # into shape [tile_len1 x tile_len2 x batch size] (i.e. a vector)

        targs = last_target.view(-1).cpu()
        # usually classification will only have target per image but in the
        # case of semantic segmentation we have a map of pixels => we need
        # to squash this to match the shape above.

        if self.n_classes == 0:
            self.n_classes = last_output.shape[self.y_idx]

        if self.cm is None:
            # make an empty confusion matrix
           self.cm = torch.zeros((self.n_classes, self.n_classes),
                                  device=torch.device('cpu'))

        cm_temp_numpy = self.cm.numpy()

        # use predictions and targets as indices and add 1 each time
        np.add.at(cm_temp_numpy, (targs, preds), 1)

        self.cm = torch.from_numpy(cm_temp_numpy)

    def on_epoch_end(self, **kwargs):
        self.metric = self.cm


@dataclass
class CMScores(ConfusionMatrix):
    """Base class for metrics which rely on the calculation of the precision
     and/or recall score."""
    average: Optional[str] = "binary"
    # `binary`, `micro`, `macro`, `weighted` or None
    pos_label: int = 1                     # 0 or 1
    eps: float = 1e-9

    def _recall(self):
        rec = torch.diag(self.cm) / self.cm.sum(dim=1)
        if self.average is None:
            return rec
        else:
            if self.average == "micro":
                weights = self._weights(avg="weighted")
            else:
                weights = self._weights(avg=self.average)

            return (rec * weights).sum()

    def _precision(self):
        prec = torch.diag(self.cm) / self.cm.sum(dim=0)

        if self.average is None:
            return prec
        else:
            weights = self._weights(avg=self.average)
            return (prec * weights).sum()

    def _weights(self, avg:str):
        if self.n_classes != 2 and avg == "binary":
            avg = self.average = "macro"
            warn("average=`binary` was selected for a non binary case. "
                 "Value for average has now been set to `macro` instead.")
        if avg == "binary":
            if self.pos_label not in (0, 1):
                self.pos_label = 1
                warn("Invalid value for pos_label. It has now been set to 1.")
            if self.pos_label == 1:
                return Tensor([0,1])
            else:
                return Tensor([1,0])
        elif avg == "micro":
            return self.cm.sum(dim=0) / self.cm.sum()
        elif avg == "macro":
            return torch.ones((self.n_classes,)) / self.n_classes
        elif avg == "weighted":
            return self.cm.sum(dim=1) / self.cm.sum()


class Recall(CMScores):
    """Computes the Recall."""
    def on_epoch_end(self, last_metrics, **kwargs):
        return add_metrics(last_metrics, self._recall())


class Precision(CMScores):
    """Computes the Precision."""
    def on_epoch_end(self, last_metrics, **kwargs):
        return add_metrics(last_metrics, self._precision())


@dataclass
class FBeta(CMScores):
    """Computes the F`beta` score."""
    beta: float = 2

    def on_train_begin(self, **kwargs):
        self.n_classes = 0
        self.beta2 = self.beta ** 2
        self.avg = self.average

        if self.average != "micro":
            self.average = None

    def on_epoch_end(self, last_metrics, **kwargs):
        prec = self._precision()
        rec = self._recall()
        metric = ((1 + self.beta2) * prec * rec /
                  (prec * self.beta2 + rec + self.eps))
        metric[metric != metric] = 0  # removing potential "nan"s

        if self.avg:
            metric = (self._weights(avg=self.avg) * metric).sum()

        return add_metrics(last_metrics, metric)

    def on_train_end(self, **kwargs):
        self.average = self.avg


# binary accuracy
def accuracy(input: Tensor, targs: Tensor) -> Rank0Tensor:
    """Computes accuracy with `targs` when `input` is bs * n_classes."""
    input = input.argmax(1).view(-1)
    targs = targs.view(-1)
    return (input == targs).float().mean()
