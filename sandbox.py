from bigbird.core import flags
from bigbird.core import modeling
from bigbird.core import utils
from bigbird.translation import run_translation
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from build_input import input_fn_builder
from tqdm import tqdm
import sys

FLAGS = flags.FLAGS
if not hasattr(FLAGS, "f"): flags.DEFINE_string("f", "", "")
FLAGS(sys.argv)

tf.enable_v2_behavior()

params = utils.BigBirdConfig(vocab_size=10000, hidden_size=512,
    num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

model = modeling.TransformerModel(params, train=True)
headl = run_classifier.ClassifierLossLayer(
        hidden_size = 768, bert_config["num_labels"],
        bert_config["hidden_dropout_prob"],
        utils.create_initializer(bert_config["initializer_range"]),
        name=bert_config["scope"]+"/classifier")
