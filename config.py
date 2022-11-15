class Config(object):
    def __init__(self):
        self.gpu_ids = [0, 1]
        self.onegpu = 2
        self.test_one_gpu = 6
        self.num_epochs = 150
        self.add_epoch = 0
        self.iter_per_epoch = 2000
        self.init_lr = 1e-4
        self.alpha = 0.999
        self.logname = "normal"

        # dataset
        self.train_path = './data/citypersons'
        self.train_random = True

        # setting for network architechture
        self.network = 'resnet50'  # or 'mobilenet'
        self.point = 'center'  # or 'top', 'bottom
        self.scale = 'h'  # or 'w', 'hw'
        self.num_scale = 1  # 1 for height (or width) prediction, 2 for height+width prediction
        self.offset = False  # append offset prediction or not
        self.down = 4  # downsampling rate of the feature map for detection
        self.radius = 2  # surrounding areas of positives for the scale map

        # setting for data augmentation
        self.use_horizontal_flips = True
        self.brightness = (0.5, 2, 0.5)
        self.size_train = (336, 448)
        self.size_test = (336, 338)
        self.weight_decay = 0.0005
        self.weight_decay_bias = 0.0005
        self.optimizer_name = "Adam"
        self.momentum = 0.9
        self.steps = [30, 60, 90, 120]
        self.gamma = 0.1
        self.warmup_factor= 0.01
        self.warmup_iters = 10
        self.warmup_method = "linear"
        self.bias_lr_factor = 1
        self.cos_f = 25
        self.alpha = 1


        # image channel-wise mean to subtract, the order is BGR
        self.img_channel_mean = [103.939, 116.779, 123.68]

        # whether or not use caffe style training which is used in paper
        self.caffemodel = False

        # use teacher
        self.teacher = False

        self.test_path = './data/citypersons'

        # whether or not to do validation during training
        self.val = True
        self.val_frequency = 10

    def print_conf(self):
        print ('\n'.join(['%s:%s' % item for item in self.__dict__.items()]))