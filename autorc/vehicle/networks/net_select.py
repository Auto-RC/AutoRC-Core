
from autorc.vehicle.networks.conv_net import ConvNet
from autorc.vehicle.networks.dense_net import DenseNet

class Network:

    @staticmethod
    def select(**kwargs):

        if kwargs['network'] == "ConvNet":
            return ConvNet(**kwargs)


        if kwargs['network'] == "DenseNet":
            return DenseNet(**kwargs)
