# Cognitbit Solutions LLP
#
# Author: Subhojeet Pramanik
# ==============================================================================
"""
Universal Trainer Script. Responsible for loading, creating checkpoints and supervised training on data given logits and
a get_data() function. 

"""

import os
import tensorflow as tf
import numpy as np

from six.moves import xrange

tf.logging.set_verbosity(tf.logging.INFO)


def train(sess, logits, fingerprint_input, ground_truth_input, get_train_data, get_val_data,
          training_steps, learning_rate, eval_step_interval, logging_interval=10, start_checkpoint=None,
          checkpoint_interval=None, model_name='', train_dir=None, summaries_dir=None, dropout=0.5, args=()):
    """
    Universal Trainer function.
    
    Args:
        sess: A tensorflow session object
        logits: The output logits of the tf graph to which loss function will be applied
        fingerprint_input: A tensorflow placeholder for feeding the input data. Shape of (None, input size)
        ground_truth_input: A tensorflow placeholder for feeding the actual truth labels. Shape of (None, labels count)
        get_train_data: Function that returns the training data as tuple of train_fingerprints, train_ground_truth
        get_val_data: Function that returns the validation data as tuple of val_fingerprints, val_ground_truth
        training_steps: Training steps seperated by comma 
        learning_rate: Learning rates seperated by comma
        logging_interval: After how many steps to log output. Default is 10.
        eval_step_interval: After how many steps to evaluate on validation set.
        start_checkpoint: An optional checkpoint to start the training from. Default None.
        checkpoint_interval: After how many steps to checkpoint. Default is None.
        model_name: The model architecture name. Default is ''
        train_dir: Directory to write event logs and checkpoint. Default '' means summary not written
        summaries_dir: Where to write the training summaries. Default '' means summary not written
        args: Tuple of args. dropout_prob,label_count,batch_size,val_size
    Returns:
         None
    """
    # Modify here to get the required varibales in the training function.
    dropout_prob, label_count, batch_size, val_size = args

    training_steps_list = list(map(int, training_steps.split(',')))
    learning_rates_list = list(map(float, learning_rate.split(',')))
    if len(training_steps_list) != len(learning_rates_list):
        raise Exception(
            'training_steps and learning_rate must be equal length '
            'lists, but are %d and %d long instead' % (len(training_steps_list),
                                                       len(learning_rates_list)))
    # Calculate the loss.
    with tf.name_scope('cross_entropy'):
        cross_entropy_mean = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=ground_truth_input, logits=logits))
    tf.summary.scalar('cross_entropy', cross_entropy_mean)
    # Add the optimization function
    with tf.name_scope('train'):
        learning_rate_input = tf.placeholder(
            tf.float32, [], name='learning_rate_input')
        train_step = tf.train.AdamOptimizer(
                learning_rate_input).minimize(cross_entropy_mean)
    predicted_indices = tf.argmax(logits, 1)
    expected_indices = tf.argmax(ground_truth_input, 1)
    correct_prediction = tf.equal(predicted_indices, expected_indices)
    confusion_matrix = tf.confusion_matrix(expected_indices, predicted_indices, num_classes=label_count)
    evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', evaluation_step)
    global_step = tf.contrib.framework.get_or_create_global_step()
    increment_global_step = tf.assign(global_step, global_step + 1)
    saver = tf.train.Saver(tf.global_variables())
    merged_summaries = tf.summary.merge_all()
    if(summaries_dir!=None):
        train_writer = tf.summary.FileWriter(summaries_dir + '/%s_train' % model_name,
                                         sess.graph)
        validation_writer = tf.summary.FileWriter(summaries_dir + '/%svalidation' % model_name)
    tf.global_variables_initializer().run()

    start_step = 1

    if start_checkpoint is not None:
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(sess, start_checkpoint)
        start_step = global_step.eval(session=sess)

    tf.logging.info('Training from step: %d ', start_step)

    # Save graph.pbtxt.
    if(train_dir!=None):
        tf.train.write_graph(sess.graph_def, train_dir,
                         model_name + '.pbtxt')
    # Training loop.
    training_steps_max = np.sum(training_steps_list)
    for training_step in xrange(start_step, training_steps_max + 1):
        # Figure out what the current learning rate is.
        training_steps_sum = 0
        for i in range(len(training_steps_list)):
            training_steps_sum += training_steps_list[i]
            if training_step <= training_steps_sum:
                learning_rate_value = learning_rates_list[i]
                break
        # Pull the audio samples we'll use for training.
        # Modify here to pass whatever argument is required.
        train_fingerprints, train_ground_truth = get_train_data(sess)
        feed_dict={
                fingerprint_input: train_fingerprints,
                ground_truth_input: train_ground_truth,
                learning_rate_input: learning_rate_value,
                dropout_prob: dropout,
            }
        if(model_name=='convlstm'):
            feed_dict['train_mode:0']=True
        train_summary, train_accuracy, cross_entropy_value, _, _= sess.run(
            [
                merged_summaries, evaluation_step, cross_entropy_mean, train_step,
                increment_global_step
            ],feed_dict
            )
        if(summaries_dir!=None):
            train_writer.add_summary(train_summary, training_step)
        if (training_step % logging_interval) == 0:
            tf.logging.info('Step #%d: rate %f, accuracy %.1f%%, cross entropy %f' %
                            (training_step, learning_rate_value, train_accuracy * 100,
                             cross_entropy_value))
        is_last_step = (training_step == training_steps_max)
        if (training_step % eval_step_interval) == 0 or is_last_step:
            # Evaluate on the validation set.
            # Modify here as desired
            total_accuracy = 0
            total_conf_matrix = None
            batch_iter = batch_size
            for i in xrange(0, val_size, batch_iter):
                # Modify here as required
                fun_args = (sess, i)
                validation_fingerprints, validation_ground_truth = (
                    get_val_data(fun_args))
                # Run a validation step and capture training summaries for TensorBoard
                # with the `merged` op.
                feed_dict={
                        fingerprint_input: validation_fingerprints,
                        ground_truth_input: validation_ground_truth,
                        dropout_prob: 1.0,
                    }
                if(model_name=='convlstm'):
                    feed_dict['train_mode:0']=False
                validation_summary, validation_accuracy, conf_matrix = sess.run(
                    [merged_summaries, evaluation_step, confusion_matrix],
                        feed_dict)
                if(summaries_dir!=None):
                    validation_writer.add_summary(validation_summary, training_step)
                batch_iter = min(batch_iter, val_size - i)
                total_accuracy += (validation_accuracy * batch_iter) / val_size
                if total_conf_matrix is None:
                    total_conf_matrix = conf_matrix
                else:
                    total_conf_matrix += conf_matrix
            tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
            tf.logging.info('Step %d: Validation accuracy = %.1f%% (N=%d)' %
                            (training_step, total_accuracy * 100, val_size))
            total_accuracy_summary = tf.Summary(value=[tf.Summary.Value(tag='Average Validation Accuracy',
                                                     simple_value=total_accuracy)])
            if(summaries_dir!=None):
                validation_writer.add_summary(total_accuracy_summary,training_step)

        # Save the model checkpoint periodically.
        if (checkpoint_interval != None) and (training_step % checkpoint_interval == 0 or
                                                      training_step == training_steps_max):
            checkpoint_path = os.path.join(train_dir,
                                           model_name, 'ckpt')
            tf.logging.info('Saving to "%s-%d"', checkpoint_path, training_step)
            saver.save(sess, checkpoint_path, global_step=training_step)
    tf.logging.info('Training Completed.')
    return total_accuracy


def eval_test(sess, logits, fingerprint_input, ground_truth_input, get_test_data, checkpoint_file, args=()):
    """
    Universal Test Evaluator function
    
    Args:
        sess: A tensorflow session object
        logits: The output logits of the tf graph to which loss function will be applied
        fingerprint_input: A tensorflow placeholder for feeding the input data. Shape of (None, input size)
        ground_truth_input: A tensorflow placeholder for feeding the actual truth labels. Shape of (None, labels count)
        get_test_data: Function that returns the test data as tuple of test_fingerprints, test_ground_truth
        checkpoint_file: The checkpoint file to use for test evaluation
        args: Tuple of args. dropout_prob,label_count,batch_size,val_size   
    Returns:
        Nothing
    """
    # Initialise prediction nodes
    label_count, batch_size, test_size = args
    predicted_indices = tf.argmax(logits, 1)
    expected_indices = tf.argmax(ground_truth_input, 1)
    correct_prediction = tf.equal(predicted_indices, expected_indices)
    confusion_matrix = tf.confusion_matrix(expected_indices, predicted_indices, num_classes=label_count)
    evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # Load Checkpoints
    tf.global_variables_initializer().run()
    if checkpoint_file is not None:
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(sess, checkpoint_file)
    # Calculate Accuracy
    total_accuracy = 0
    total_conf_matrix = None
    batch_iter = batch_size
    for i in xrange(0, test_size, batch_iter):
        # Modify here as required
        fun_args = (sess, i)
        validation_fingerprints, validation_ground_truth = (
            get_test_data(fun_args))
        # Run a validation step and capture training summaries for TensorBoard
        # with the `merged` op.
        test_accuracy, conf_matrix = sess.run(
            [evaluation_step, confusion_matrix],
            feed_dict={
                fingerprint_input: validation_fingerprints,
                ground_truth_input: validation_ground_truth,
            })
        batch_iter = min(batch_iter, test_size - i)
        total_accuracy += (test_accuracy * batch_size) / test_size
        if total_conf_matrix is None:
            total_conf_matrix = conf_matrix
        else:
            total_conf_matrix += conf_matrix
    tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
    tf.logging.info('Test accuracy = %.1f%% (N=%d)' %
                    (total_accuracy * 100, test_size))


