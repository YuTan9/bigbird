def input_fn_builder(data_dir, vocab_model_file, max_encoder_length,
                     max_decoder_length, substitute_newline, is_training,
                     tmp_dir=None):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  def _decode_record(record):
    """Decodes a record to a TensorFlow example."""
    name_to_features = {
        "document": tf.io.FixedLenFeature([], tf.string),
        "summary": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(record, name_to_features)
    return example["document"], example["summary"]

  def _tokenize_example(document, summary):
    tokenizer = tft.SentencepieceTokenizer(
        model=tf.io.gfile.GFile(vocab_model_file, "rb").read())
    if substitute_newline:
      document = tf.strings.regex_replace(document, "\n", substitute_newline)
    # Remove space before special tokens.
    document = tf.strings.regex_replace(document, r" ([<\[]\S+[>\]])", b"\\1")
    document_ids = tokenizer.tokenize(document)
    if isinstance(document_ids, tf.RaggedTensor):
      document_ids = document_ids.to_tensor(0)
    document_ids = document_ids[:max_encoder_length]

    # Remove newline optionally
    if substitute_newline:
      summary = tf.strings.regex_replace(summary, "\n", substitute_newline)
    # Remove space before special tokens.
    summary = tf.strings.regex_replace(summary, r" ([<\[]\S+[>\]])", b"\\1")
    summary_ids = tokenizer.tokenize(summary)
    # Add [EOS] (1) special tokens.
    suffix = tf.constant([1])
    summary_ids = tf.concat([summary_ids, suffix], axis=0)
    if isinstance(summary_ids, tf.RaggedTensor):
      summary_ids = summary_ids.to_tensor(0)
    summary_ids = summary_ids[:max_decoder_length]

    return document_ids, summary_ids

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # Load dataset and handle tfds separately
    split = "train" if is_training else "validation"
    if "tfds://" == data_dir[:7]:
      d = tfds.load(data_dir[7:], split=split, data_dir=tmp_dir,
                    shuffle_files=is_training, as_supervised=True)
    else:
      input_files = tf.io.gfile.glob(
          os.path.join(data_dir, "{}.tfrecord*".format(split)))

      # For training, we want a lot of parallel reading and shuffling.
      # For eval, we want no shuffling and parallel reading doesn't matter.
      if is_training:
        d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
        d = d.shuffle(buffer_size=len(input_files))

        # Non deterministic mode means that the interleaving is not exact.
        # This adds even more randomness to the training pipeline.
        d = d.interleave(tf.data.TFRecordDataset,
                         deterministic=False,
                         num_parallel_calls=tf.data.experimental.AUTOTUNE)
      else:
        d = tf.data.TFRecordDataset(input_files)

      d = d.map(_decode_record,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
                deterministic=is_training)

    d = d.map(_tokenize_example,
              num_parallel_calls=tf.data.experimental.AUTOTUNE,
              deterministic=is_training)

    if is_training:
      d = d.shuffle(buffer_size=10000, reshuffle_each_iteration=True)
      d = d.repeat()
    d = d.padded_batch(batch_size, ([max_encoder_length], [max_decoder_length]),
                       drop_remainder=True)  # For static shape
    return d

  return input_fn