from keras.callbacks import TensorBoard
from tensorboardX import FileWriter
import tensorflow as tf

class CustomTensorBoard(TensorBoard):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.writer = FileWriter(self.log_dir)

    def set_model(self, model):
        pass

    def _write_logs(self, logs, index):
        """ for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            self.writer.add_summary(summary, index)
        self.writer.flush() """
        writer = tf.summary.create_file_writer("/tmp/mylogs")
        with writer.as_default():
            for step in range(100):
                # other model code would go here
                tf.summary.scalar("my_metric", 0.5, step=step)
                writer.flush()

    def log(self, step, **stats):
        self._write_logs(stats, step)