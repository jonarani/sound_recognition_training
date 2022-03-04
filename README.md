# sound_recognition_training

Python 3.6.9
Tensorflow 1.15.2

Created a train_and_generate.py based on the following:
1. https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples/micro_speech/train/train_micro_speech_model.ipynb
2. https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/speech_commands

train_and_generate.py includes train.py, freeze.py and generating quantized .cc model in one script.

Note the comment in representative_dataset_gen().

When MFCC is used then MFCC params need to be manually updated if necessary: audio_ops.mfcc in input_data.py and freeze.py
