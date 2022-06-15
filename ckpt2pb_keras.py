import argparse
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()
from model import PyNET


def convert_to_tflite(input_checkpoint, out_path):
    #IMAGE_HEIGHT, IMAGE_WIDTH = 1472, 1984
    IMAGE_HEIGHT, IMAGE_WIDTH = 128, 128
    #IMAGE_HEIGHT, IMAGE_WIDTH = 544, 960
    
    LEVEL = 1

    with tf.Session() as sess:
        x_ = tf.keras.Input(batch_size=1, shape=([IMAGE_HEIGHT, IMAGE_WIDTH, 4]))

        # generate enhanced image
        output_l0, output_l1, output_l2, output_l3 =\
            PyNET(x_, instance_norm=False, instance_norm_level_1=False)

        if LEVEL == 3:
            enhanced = output_l3
        if LEVEL == 2:
            enhanced = output_l2
        if LEVEL == 1:
            enhanced = output_l1
        if LEVEL == 0:
            enhanced = output_l0

        # Loading pre-trained model
        sess.run(tf.global_variables_initializer())
        model = tf.keras.Model(inputs=x_, outputs=enhanced)
        
        from tensorflow.keras.models import load_model
        prev_model = load_model(input_checkpoint, compile=False)
        for i, layer in enumerate(prev_model.layers):
            for k in model.layers:
                if k.name == layer.name:
                    k.set_weights(layer.get_weights())

        # Convert the model
        converter = tf.lite.TFLiteConverter.from_session(sess, [x_], [enhanced])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.experimental_new_converter = True
        converter.experimental_new_quantizer = True
        converter.optimizations = []
        tflite_model = converter.convert()

        # Save the model.
        with open(out_path, 'wb') as f:
            f.write(tflite_model)


def _parse_argument():
    """Convert model to TFLite Format"""
    parser = argparse.ArgumentParser(
        description='Convert model to TFLite Format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--in_path', help='Path to input checkpoint.', type=str, default='model.h5', required=True)
    parser.add_argument(
        '--out_path', help='Path to the output pb.', type=str, default='model.pb', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    arguments = _parse_argument()
    convert_to_tflite(arguments.in_path, arguments.out_path)
