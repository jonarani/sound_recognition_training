import subprocess
import os
import sys
# We add this path so we can import the speech processing modules.
sys.path.append("../sound_recognition_training")
import input_data
import models
import numpy as np

import tensorflow as tf

tf.compat.v1.disable_eager_execution()


WANTED_WORDS = "yes,no,cat,dog"
SAMPLE_RATE = 16000
WINDOW_SIZE = 64.0
WINDOW_STRIDE = 56.0
FEATURE_BIN_COUNT = 13
PREPROCESS = 'micro'
BATCH_SIZE = 100
NUM_OF_TMP = '1'

MODEL_ARCHITECTURE = 'tiny_conv' # single_fc, conv, low_latency_conv, low_latency_svdf, tiny_embedding_conv

CLIP_DURATION = 1000

BACKGROUND_FREQUENCY = 0.8
BACKGROUND_VOLUME_RANGE = 0.1
TIME_SHIFT_MS = 100.0

# Constants used during training only
VERBOSITY = 'INFO'
EVAL_STEP_INTERVAL = 1000
SAVE_STEP_INTERVAL = 1000

# The number of steps and learning rates can be specified as comma-separated
# lists to define the rate at each stage. For example,
# TRAINING_STEPS=12000,3000 and LEARNING_RATE=0.001,0.0001
# will run 12,000 training loops in total, with a rate of 0.001 for the first
# 8,000, and 0.0001 for the final 3,000.
TRAINING_STEPS = "8000,2000"
LEARNING_RATE = "0.001,0.0001"


#DATA_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
DATA_URL = ''
VALIDATION_PERCENTAGE = 10
TESTING_PERCENTAGE = 10

# Constants for training directories and filepaths
DATASET_DIR =  'dataset/'
LOGS_DIR = 'results/tmp' + NUM_OF_TMP + '/summaries'
TRAIN_DIR = 'results/tmp' + NUM_OF_TMP + '/training/' # for training checkpoints and other files.
MODELS_DIR = 'results/tmp' + NUM_OF_TMP + '/models'

if not os.path.exists('results'):
    os.mkdir('results')
if not os.path.exists('results/tmp' + NUM_OF_TMP):
    os.mkdir('results/tmp' + NUM_OF_TMP)
if not os.path.exists(TRAIN_DIR):
    os.mkdir(TRAIN_DIR)
if not os.path.exists(LOGS_DIR):
    os.mkdir(LOGS_DIR)
if not os.path.exists(MODELS_DIR):
    os.mkdir(MODELS_DIR)

# Constants for inference directories and filepaths
MODEL_TF = os.path.join(MODELS_DIR, 'model.pb')
MODEL_TFLITE = os.path.join(MODELS_DIR, 'model.tflite')
FLOAT_MODEL_TFLITE = os.path.join(MODELS_DIR, 'float_model.tflite')
MODEL_TFLITE_MICRO = os.path.join(MODELS_DIR, 'model' + NUM_OF_TMP + '.cc')
SAVED_MODEL = os.path.join(MODELS_DIR, 'saved_model')


# Calculate the total number of steps, which is used to identify the checkpoint file name.
TOTAL_STEPS = str(sum(map(lambda string: int(string), TRAINING_STEPS.split(","))))

# Print the configuration to confirm it
print("Training these words: %s" % WANTED_WORDS)
print("Training steps in each stage: %s" % TRAINING_STEPS)
print("Learning rate in each stage: %s" % LEARNING_RATE)
print("Total number of training steps: %s" % TOTAL_STEPS)


# Calculate the percentage of 'silence' and 'unknown' training samples required
# to ensure that we have equal number of samples for each label.
number_of_labels = WANTED_WORDS.count(',') + 1
number_of_total_labels = number_of_labels + 2 # for 'silence' and 'unknown' label
equal_percentage_of_training_samples = int(100.0/(number_of_total_labels))
SILENT_PERCENTAGE = equal_percentage_of_training_samples
UNKNOWN_PERCENTAGE = equal_percentage_of_training_samples


subprocess.run(['python3',
                'train.py',
                '--data_dir={}'.format(DATASET_DIR), 
                '--wanted_words={}'.format(WANTED_WORDS),
                '--silence_percentage={}'.format(SILENT_PERCENTAGE),
                '--unknown_percentage={}'.format(UNKNOWN_PERCENTAGE),
                '--preprocess={}'.format(PREPROCESS),
                '--window_stride_ms={}'.format(WINDOW_STRIDE),
                '--model_architecture={}'.format(MODEL_ARCHITECTURE),
                '--how_many_training_steps={}'.format(TRAINING_STEPS),
                '--learning_rate={}'.format(LEARNING_RATE),
                '--train_dir={}'.format(TRAIN_DIR),
                '--summaries_dir={}'.format(LOGS_DIR),
                '--verbosity={}'.format(VERBOSITY),
                '--eval_step_interval={}'.format(EVAL_STEP_INTERVAL),
                '--save_step_interval={}'.format(SAVE_STEP_INTERVAL),
                '--sample_rate={}'.format(SAMPLE_RATE),
                '--feature_bin_count={}'.format(FEATURE_BIN_COUNT),
                '--window_size_ms={}'.format(WINDOW_SIZE),
                '--clip_duration_ms={}'.format(CLIP_DURATION),
                '--batch_size={}'.format(BATCH_SIZE)
                ])


subprocess.run(['python3',
                'freeze.py',
                '--wanted_words={}'.format(WANTED_WORDS),
                '--window_stride_ms={}'.format(WINDOW_STRIDE),
                '--sample_rate={}'.format(SAMPLE_RATE),
                '--feature_bin_count={}'.format(FEATURE_BIN_COUNT),
                '--window_size_ms={}'.format(WINDOW_SIZE),
                '--preprocess={}'.format(PREPROCESS),
                '--model_architecture={}'.format(MODEL_ARCHITECTURE),
                '--start_checkpoint={}'.format(TRAIN_DIR + MODEL_ARCHITECTURE + '.ckpt-' + TOTAL_STEPS),
                '--save_format={}'.format('saved_model'),
                '--output_file={}'.format(SAVED_MODEL),
                '--clip_duration_ms={}'.format(CLIP_DURATION),
                ])


model_settings = models.prepare_model_settings(
    len(input_data.prepare_words_list(WANTED_WORDS.split(','))),
    SAMPLE_RATE, CLIP_DURATION, WINDOW_SIZE,
    WINDOW_STRIDE, FEATURE_BIN_COUNT, PREPROCESS)
audio_processor = input_data.AudioProcessor(
    DATA_URL, DATASET_DIR,
    SILENT_PERCENTAGE, UNKNOWN_PERCENTAGE,
    WANTED_WORDS.split(','), VALIDATION_PERCENTAGE,
    TESTING_PERCENTAGE, model_settings, LOGS_DIR)

with tf.Session() as sess:
    float_converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL)
    float_tflite_model = float_converter.convert()
    float_tflite_model_size = open(FLOAT_MODEL_TFLITE, "wb").write(float_tflite_model)
    print("Float model is %d bytes" % float_tflite_model_size)

    converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.inference_input_type = tf.lite.constants.INT8
    converter.inference_output_type = tf.lite.constants.INT8
    
    def representative_dataset_gen():
        for i in range(100):
            data, _ = audio_processor.get_data(1, i*1, model_settings,
                                                BACKGROUND_FREQUENCY, 
                                                BACKGROUND_VOLUME_RANGE,
                                                TIME_SHIFT_MS,
                                                'testing',
                                                sess)
        # The second argument of reshape have to manually updated
        # depending how large is the input for the neural network
        # feature_bin_count * (clip_size_ms / window_stride_ms) most of the time but not always?
        flattened_data = np.array(data.flatten(), dtype=np.float32).reshape(1, 221)
        yield [flattened_data]
    
    converter.representative_dataset = representative_dataset_gen
    tflite_model = converter.convert()
    tflite_model_size = open(MODEL_TFLITE, "wb").write(tflite_model)
    print("Quantized model is %d bytes" % tflite_model_size)


# Helper function to run inference
def run_tflite_inference(tflite_model_path, model_type="Float"):
    # Load test data
    np.random.seed(0) # set random seed for reproducible test results.
    with tf.Session() as sess:
        test_data, test_labels = audio_processor.get_data(
            -1, 0, model_settings, BACKGROUND_FREQUENCY, BACKGROUND_VOLUME_RANGE,
            TIME_SHIFT_MS, 'testing', sess)
    test_data = np.expand_dims(test_data, axis=1).astype(np.float32)

    # Initialize the interpreter
    interpreter = tf.lite.Interpreter(tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # For quantized models, manually quantize the input data from float to integer
    if model_type == "Quantized":
        input_scale, input_zero_point = input_details["quantization"]
        test_data = test_data / input_scale + input_zero_point
        test_data = test_data.astype(input_details["dtype"])

    correct_predictions = 0
    for i in range(len(test_data)):
        interpreter.set_tensor(input_details["index"], test_data[i])
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])[0]
        top_prediction = output.argmax()
        correct_predictions += (top_prediction == test_labels[i])

    print('%s model accuracy is %f%% (Number of test samples=%d)' % (
        model_type, (correct_predictions * 100) / len(test_data), len(test_data)))


# Compute float model accuracy
run_tflite_inference(FLOAT_MODEL_TFLITE)

# Compute quantized model accuracy
run_tflite_inference(MODEL_TFLITE, model_type='Quantized')

# Convert to a C source file
subprocess.call('cd {}; xxd -i {} {}'.format(MODELS_DIR, 'model.tflite', 'model{}.cc'.format(NUM_OF_TMP)), shell=True)
