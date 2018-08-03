# Style-Transfer-with-CNN
Simple wrapper to make style transfer with pre-trained CNN [VGG16](https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3)

Most of code taken from [here](https://github.com/hnarayanan/artistic-style-transfer/blob/master/notebooks/6_Artistic_style_transfer_with_a_repurposed_VGG_Net_16.ipynb)



### Example of usage:

```python
from style_transfer import VGG16StyleTransfer

model = VGG16StyleTransfer()
model.load_pictures('base.jpg', 'style.jpg')
model.transfer_style(iterations=20, output_file='result_picture')

```


### Medium article:
https://medium.com/cindicator/style-transfer-with-neural-network-weekend-of-a-data-scientist-8009d2285b74
