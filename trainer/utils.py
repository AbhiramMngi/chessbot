class ModelConfig:
    def __init__(self, filters, resid_blocks, input_shape, output_shape):
        self.n_filters = filters
        self.resid_blocks = resid_blocks
        self.n = 8
        self.input_shape = input_shape
        self.output_shape = output_shape