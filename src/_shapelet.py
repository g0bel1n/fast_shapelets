class Shapelet:
    def __init__(
        self,
        value,
        initial_sample_id=None,
        start_initial_sample=None,
        gain=None,
        gap=None,
    ):
        self.value = value
        self.gain = gain
        self.gap = gap
        self.initial_sample = initial_sample_id
        self.start_initial_sample = start_initial_sample
