import keras as K
from model import DTanh, DELU
from modelnet import ModelFetcher

#################### Settings ##############################
num_epochs = 1000
batch_size = 64
downsample = 10  # For 5000 points use 2, for 1000 use 10, for 100 use 100
network_dim = 256  # For 5000 points use 512, for 1000 use 256, for 100 use 256
num_repeats = 5  # Number of times to repeat the experiment
data_path = 'ModelNet40_cloud.h5'
#################### Settings ##############################


class PointCloudTrainer(object):

    def __init__(self):
        # Data loader
        self.model_fetcher = ModelFetcher(data_path, downsample,
                                          do_standardize=True,
                                          do_augmentation=True)
        # Setup network
        self.model = DTanh(network_dim, pool='max1')
        self.model.build([None, 3])
        self.model.summary()
        optimizer = K.optimizers.Adam(lr=1e-3, decay=1e-7, epsilon=1e-3)

        # No learning rate scheduler.
        self.model.compile(
            loss=K.losses.categorical_crossentropy,
            optimizer=optimizer,
        )

        # self.optimizer = optim.Adam([{'params': self.model.parameters()}], lr=1e-3, weight_decay=1e-7, eps=1e-3)
        # self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=list(range(400, num_epochs, 400)),
        # gamma = 0.1)
        # self.optimizer = optim.Adamax([{'params':self.D.parameters()}], lr=5e-4, weight_decay=1e-7, eps=1e-3) # optionally use this for 5000 points case, but adam with scheduler also works

    def train(self):
        x, y = self.model_fetcher.train_data()
        self.model.fit(x, y, batch_size=batch_size, epochs=num_epochs, verbose=True)

    def test(self):
        x, y = self.model_fetcher.test_data()
        self.model.evaluate(x, y, batch_size, verbose=True)

    def summary(self):
        self.model.summary()


def main():
    trainer = PointCloudTrainer()
    trainer.summary()
    trainer.train()
    trainer.test()


if __name__ == '__main__':
    main()
