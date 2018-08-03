import time
from PIL import Image
import numpy as np
from keras import backend
from keras.applications.vgg16 import VGG16
from scipy.optimize import fmin_l_bfgs_b
from logger import logger


class VGG16StyleTransfer:
    """Main class that contains everything you need to perform style transfer"""

    def __init__(self):
        """
        some default parameters based on paper: https://arxiv.org/abs/1603.08155
        """
        self.height = 512
        self.width = 512
        self.content_weight = 0.025
        self.style_weight = 5.0
        self.total_variation_weight = 1.0
        self.loss = backend.variable(0.)
        self.model = None
        self.content_image = None
        self.style_image = None
        self.combination_image = None
        self.tensor = None
        self.f_outputs = None
        self.feature_layers = ['block1_conv2', 'block2_conv2',
                               'block3_conv3', 'block4_conv3',
                               'block5_conv3']

    def load_pictures(self, base_image, style_image):
        """

        :param base_image: string - path to base image that you whant to transform
        :param style_image: string - path to image with style that you want to apply to base image
        :return:
        """
        # open pictures
        content_image = Image.open(base_image)
        content_image = content_image.resize((self.width, self.height))
        style_image = Image.open(style_image)
        style_image = style_image.resize((self.width, self.height))

        # convert to machine readable format
        content_array = np.asarray(content_image, dtype='float32')
        content_array = np.expand_dims(content_array, axis=0)
        style_array = np.asarray(style_image, dtype='float32')
        style_array = np.expand_dims(style_array, axis=0)

        content_array[:, :, :, 0] -= 103.939
        content_array[:, :, :, 1] -= 116.779
        content_array[:, :, :, 2] -= 123.68
        content_array = content_array[:, :, :, ::-1]

        style_array[:, :, :, 0] -= 103.939
        style_array[:, :, :, 1] -= 116.779
        style_array[:, :, :, 2] -= 123.68
        style_array = style_array[:, :, :, ::-1]

        self.content_image = backend.variable(content_array)
        self.style_image = backend.variable(style_array)
        self.combination_image = backend.placeholder((1, self.height, self.width, 3))

        # create tensor that can be read by VGG16 model
        self.tensor = backend.concatenate([self.content_image,
                                           self.style_image,
                                           self.combination_image], axis=0)

    def _load_vgg16(self):
        # download vgg16
        self.model = VGG16(input_tensor=self.tensor, weights='imagenet',
                           include_top=False)

    def _content_loss(self, content, combination):
        """The content loss is the (scaled, squared)
        Euclidean distance between feature representations
        of the content and combination images."""
        return backend.sum(backend.square(combination - content))

    def _gram_matrix(self, x):
        """
        Compute Gram matrix -
        The terms of this matrix are proportional to the covariances of corresponding
        sets of features, and thus captures information about which
        features tend to activate together
        """
        features = backend.batch_flatten(backend.permute_dimensions(x, (2, 0, 1)))
        gram = backend.dot(features, backend.transpose(features))
        return gram

    def _style_loss(self, style, combination):
        S = self._gram_matrix(style)
        C = self._gram_matrix(combination)
        channels = 3
        size = self.height * self.width
        return backend.sum(backend.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

    def _total_variation_loss(self, x):
        a = backend.square(x[:, :self.height - 1, :self.width - 1, :] - x[:, 1:, :self.width - 1, :])
        b = backend.square(x[:, :self.height - 1, :self.width - 1, :] - x[:, :self.height - 1, 1:, :])
        return backend.sum(backend.pow(a + b, 1.25))

    def _eval_loss_and_grads(self, x):
        x = x.reshape((1, self.height, self.width, 3))
        outs = self.f_outputs([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        return loss_value, grad_values

    def transfer_style(self, iterations=20, output_file='file'):
        self._load_vgg16()
        layers = dict([(layer.name, layer.output) for layer in self.model.layers])
        layer_features = layers['block2_conv2']
        content_image_features = layer_features[0, :, :, :]
        combination_features = layer_features[2, :, :, :]
        self.loss += self.content_weight * self._content_loss(content_image_features, combination_features)

        for layer_name in self.feature_layers:
            layer_features = layers[layer_name]
            style_features = layer_features[1, :, :, :]
            combination_features = layer_features[2, :, :, :]
            sl = self._style_loss(style_features, combination_features)
            self.loss += (self.style_weight / len(self.feature_layers)) * sl

        self.loss += self.total_variation_weight * self._total_variation_loss(self.combination_image)
        grads = backend.gradients(self.loss, self.combination_image)
        outputs = [self.loss]
        outputs += grads
        self.f_outputs = backend.function([self.combination_image], outputs)
        evaluator = Evaluator(self._eval_loss_and_grads)
        x = np.random.uniform(0, 255, (1, self.height, self.width, 3)) - 128.
        for i in range(iterations):
            logger.info('Start of iteration: {}'.format(i))
            start_time = time.time()
            x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                             fprime=evaluator.grads, maxfun=20)
            logger.info('Current loss value: {}'.format(min_val))
            end_time = time.time()
            logger.info('Iteration {} completed in {}s'.format(i, round(end_time - start_time)))

        x = x.reshape((self.height, self.width, 3))
        x = x[:, :, ::-1]
        x[:, :, 0] += 103.939
        x[:, :, 1] += 116.779
        x[:, :, 2] += 123.68
        x = np.clip(x, 0, 255).astype('uint8')

        img = Image.fromarray(x)
        img.save('{}.jpg'.format(output_file))


class Evaluator:
    def __init__(self, eval_loss_and_grads):
        self.loss_value = None
        self.grads_values = None
        self.eval_loss_and_grads = eval_loss_and_grads

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = self.eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values
