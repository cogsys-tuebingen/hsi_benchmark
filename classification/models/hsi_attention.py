import numpy as np
import torch
import torch.nn.functional as f
from skimage.transform import resize


"""
    Hyperspectral Band Selection Using Attention-Based Convolutional Neural Networks
    based on the official code: https://github.com/ESA-PhiLab/hypernet/blob/master/python_research/experiments/hsi_attention/
"""


def build_convolutional_block(input_channels: int, output_channels: int) -> torch.nn.Sequential:
    """
    Create convolutional block for the network.

    :param input_channels: Number of input feature maps.
    :param output_channels: Number of designed output feature maps.
    :return: Sequential module.
    """
    return torch.nn.Sequential(
        torch.nn.Conv1d(in_channels=input_channels, out_channels=output_channels, kernel_size=5, padding=2),
        torch.nn.ReLU(),
        torch.nn.BatchNorm1d(output_channels),
        torch.nn.MaxPool1d(2)
    )


def build_classifier_block(input_size: int, number_of_classes: int) -> torch.nn.Sequential:
    """
    Build classifier block designed for obtaining the prediction.

    :param input_size: Input number of features.
    :param number_of_classes: Number of classes.
    :return: Sequential module.
    """
    return torch.nn.Sequential(
        torch.nn.Linear(input_size, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, number_of_classes)
    )


def build_softmax_module(input_channels: int) -> torch.nn.Sequential:
    """
    Build softmax module for attention mechanism.

    :param input_channels: Number of input feature maps.
    :return: Sequential module.
    """
    return torch.nn.Sequential(
        torch.nn.Conv1d(in_channels=input_channels, out_channels=1, kernel_size=1),
        torch.nn.ReLU(),
        torch.nn.Softmax(dim=2)
    )


def build_classifier_confidence(input_channels: int) -> torch.nn.Sequential:
    """
    Create sequential module for classifier confidence score.

    :param input_channels: Number of input activation maps.
    :return: Sequential module.
    """
    return torch.nn.Sequential(
        torch.nn.Linear(input_channels, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 1),
        torch.nn.Tanh()
    )


class AttentionBlock(torch.nn.Module):
    def __init__(self, input_channels: int, num_channels: int, num_classes: int):
        """
        Attention module constructor.

        :param input_channels: Number of input activation maps.
        :param num_channels: Size of input spectral dimensionality.
        :param num_classes: Number of classes.
        """
        super(AttentionBlock, self).__init__()
        self._softmax_block_1 = build_softmax_module(input_channels)
        self._confidence_net = torch.nn.Sequential(
            torch.nn.Linear(num_channels, 1),
            torch.nn.Tanh()
        )
        self._attention_net = torch.nn.Sequential(
            torch.nn.Linear(num_channels, num_classes)
        )
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Feed forward method for attention module.

        :param z: Input tensor.
        :param y: Labels.
        :param infer: Boolean variable indicating whether to save attention heatmap which is later used in the
                      band selection process.
        :return: Weighted attention module hypothesis.
        """
        heatmap = self._softmax_block_1(z)
        cross_product = torch.einsum("ijk,ilk->ijlk", (heatmap.clone(), z.clone())) \
            .reshape(heatmap.shape[0], -1, heatmap.shape[2])
        cross_product = f.avg_pool1d(cross_product.permute(0, 2, 1), cross_product.shape[1])
        cross_product = cross_product.squeeze()
        return self._attention_net(cross_product) * self._confidence_net(cross_product)


class HSIAttentionModel2(torch.nn.Module):

    def __init__(self, num_classes: int, num_channels: int, uses_attention: bool = True):
        """
        Initializer of model with two attention modules.

        :param num_classes: Number of classes.
        :param num_channels: Input spectral size.
        :param uses_attention: Boolean variable indicating whether the model uses attention mechanism or not.
        """
        super().__init__()
        self._conv_block_1 = build_convolutional_block(1, 96)
        self._conv_block_2 = build_convolutional_block(96, 54)
        assert int(num_channels / 4) > 0, "The spectral size is to small for model with two attention modules."
        self._classifier = build_classifier_block(54 * int(num_channels / 4), num_classes)
        if uses_attention:
            self._attention_block_1 = AttentionBlock(96, int(num_channels / 2), num_classes)
            self._attention_block_2 = AttentionBlock(54, int(num_channels / 4), num_classes)
            self._classifier_confidence = build_classifier_confidence(54 * int(num_channels / 4))
        self.uses_attention = uses_attention

    def forward(self, x: torch.Tensor, meta_data) -> torch.Tensor:
        """
       Feed forward method for model with two attention modules.

       :param x: Input tensor.
       :param y: Labels.
       :param infer: Boolean variable indicating whether to save attention heatmap which is later used in the
                     band selection process.
       :return: Weighted classifier hypothesis.
       """
        global first_module_prediction, second_module_prediction
        x = x[:, None, :, x.shape[2] // 2, x.shape[3] // 2]  # center pixel
        z = self._conv_block_1(x)
        if self.uses_attention:
            first_module_prediction = self._attention_block_1(z)
        z = self._conv_block_2(z)
        if self.uses_attention:
            second_module_prediction = self._attention_block_2(z)
        prediction = self._classifier(z.view(z.shape[0], -1))
        if self.uses_attention:
            prediction *= self._classifier_confidence(z.view(z.shape[0], -1))
        if self.uses_attention:
            return prediction + first_module_prediction + second_module_prediction
        return prediction

    def get_heatmaps(self, input_size: int) -> np.ndarray:
        """
        Return averaged heatmaps for model with two attention modules.

        :param input_size: Designed spectral size.
        :return: Array containing averaged scores for interpolated heatmaps.
        """
        return np.mean([self._attention_block_1.get_heatmaps(input_size).squeeze(),
                        self._attention_block_2.get_heatmaps(input_size).squeeze()], axis=0).squeeze()


class HSIAttentionModel3(torch.nn.Module):

    def __init__(self, num_classes: int, num_channels: int, uses_attention: bool = True):
        """
        Initializer of model with three attention modules.

        :param num_classes: Number of classes.
        :param num_channels: Input spectral size.
        :param uses_attention: Boolean variable indicating whether the model uses attention mechanism or not.
        """
        super().__init__()
        self._conv_block_1 = build_convolutional_block(1, 96)
        self._conv_block_2 = build_convolutional_block(96, 54)
        self._conv_block_3 = build_convolutional_block(54, 36)
        assert int(num_channels / 8) > 0, "The spectral size is to small for model with three attention modules."
        self._classifier = build_classifier_block(36 * int(num_channels / 8), num_classes)
        if uses_attention:
            self._attention_block_1 = AttentionBlock(96, int(num_channels / 2), num_classes)
            self._attention_block_2 = AttentionBlock(54, int(num_channels / 4), num_classes)
            self._attention_block_3 = AttentionBlock(36, int(num_channels / 8), num_classes)
            self._classifier_confidence = build_classifier_confidence(36 * int(num_channels / 8))
        self.uses_attention = uses_attention

    def forward(self, x: torch.Tensor, meta_data) -> torch.Tensor:
        """
       Feed forward method for model with three attention modules.

       :param x: Input tensor.
       :param y: Labels.
       :param infer: Boolean variable indicating whether to save attention heatmap which is later used in the
                     band selection process.
       :return: Weighted classifier hypothesis.
        """
        global first_module_prediction, second_module_prediction, third_module_prediction
        x = x[:, None, :, x.shape[2] // 2, x.shape[3] // 2]  # center pixel
        z = self._conv_block_1(x)
        if self.uses_attention:
            first_module_prediction = self._attention_block_1(z)
        z = self._conv_block_2(z)
        if self.uses_attention:
            second_module_prediction = self._attention_block_2(z)
        z = self._conv_block_3(z)
        if self.uses_attention:
            third_module_prediction = self._attention_block_3(z)
        prediction = self._classifier(z.view(z.shape[0], -1))
        if self.uses_attention:
            prediction *= self._classifier_confidence(z.view(z.shape[0], -1))
        if self.uses_attention:
            return prediction + first_module_prediction + second_module_prediction + third_module_prediction
        return prediction

    def get_heatmaps(self, input_size: int) -> np.ndarray:
        """
       Return averaged heatmaps for model with three attention modules.

       :param input_size: Designed spectral size.
       :return: Array containing averaged scores for interpolated heatmaps.
       """
        return np.mean([self._attention_block_1.get_heatmaps(input_size).squeeze(),
                        self._attention_block_2.get_heatmaps(input_size).squeeze(),
                        self._attention_block_3.get_heatmaps(input_size).squeeze()], axis=0).squeeze()




class HSIAttentionModel4(torch.nn.Module):

    def __init__(self, num_classes: int, num_channels: int, uses_attention: bool = True):
        """
        Initializer of model with four attention modules.

        :param num_classes: Number of classes.
        :param num_channels: Input spectral size.
        :param uses_attention: Boolean variable indicating weather the model uses attention mechanism or not.
        """
        super().__init__()
        self._conv_block_1 = build_convolutional_block(1, 96)
        self._conv_block_2 = build_convolutional_block(96, 54)
        self._conv_block_3 = build_convolutional_block(54, 36)
        self._conv_block_4 = build_convolutional_block(36, 24)
        assert int(num_channels / 16) > 0, "The spectral size is to small for model with four attention modules."
        self._classifier = build_classifier_block(24 * int(num_channels / 16), num_classes)
        if uses_attention:
            self._attention_block_1 = AttentionBlock(96, int(num_channels / 2), num_classes)
            self._attention_block_2 = AttentionBlock(54, int(num_channels / 4), num_classes)
            self._attention_block_3 = AttentionBlock(36, int(num_channels / 8), num_classes)
            self._attention_block_4 = AttentionBlock(24, int(num_channels / 16), num_classes)
            self._classifier_confidence = build_classifier_confidence(24 * int(num_channels / 16))
        self.uses_attention = uses_attention

    def forward(self, x: torch.Tensor, meta_data) -> torch.Tensor: 
        """
        Feed forward method for model with four attention modules.

        :param x: Input tensor.
        :param y: Labels.
        :param infer: Boolean variable indicating whether to save attention heatmap which is later used in the
                      band selection process.
        :return: Weighted classifier hypothesis.
        """
        global first_module_prediction, second_module_prediction, \
            third_module_prediction, fourth_module_prediction
        x = x[:, None, :, x.shape[2] // 2, x.shape[3] // 2]  # center pixel
        z = self._conv_block_1(x)
        if self.uses_attention:
            first_module_prediction = self._attention_block_1(z)
        z = self._conv_block_2(z)
        if self.uses_attention:
            second_module_prediction = self._attention_block_2(z)
        z = self._conv_block_3(z)
        if self.uses_attention:
            third_module_prediction = self._attention_block_3(z)
        z = self._conv_block_4(z)
        if self.uses_attention:
            fourth_module_prediction = self._attention_block_4(z)
        prediction = self._classifier(z.view(z.shape[0], -1))
        if self.uses_attention:
            prediction *= self._classifier_confidence(z.view(z.shape[0], -1))
        if self.uses_attention:
            return prediction + first_module_prediction + second_module_prediction + \
                             third_module_prediction + fourth_module_prediction
        return prediction
