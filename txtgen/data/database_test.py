# -*- coding: utf-8 -*-
#
"""
Unit tests for database related operations.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tempfile
import tensorflow as tf

from txtgen.data import database


class TextDataBaseTest(tf.test.TestCase):
    """Tests text database class.
    """

    def test_mono_text_database(self):
        """Tests the logics of MonoTextDataBase.
        """
        # Create test data
        vocab_list = ['word', '词']
        vocab_file = tempfile.NamedTemporaryFile()
        vocab_file.write('\n'.join(vocab_list).encode("utf-8"))
        vocab_file.flush()

        text = ['This is a test sentence .', '词 词 。']
        text_file = tempfile.NamedTemporaryFile()
        text_file.write('\n'.join(text).encode("utf-8"))
        text_file.flush()

        # Construct database
        hparams = {
            "num_epochs": 2,
            "batch_size": 2,
            "dataset": {
                "files": [text_file.name],
                "vocab.file": vocab_file.name,
            }
        }

        text_database = database.MonoTextDataBase(hparams)
        text_data_batch = text_database()

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            sess.run(tf.tables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                while not coord.should_stop():
                    # Run the logics
                    data = sess.run(text_data_batch)

                    self.assertEqual(set(data.keys()),
                                     set(text_database.list_items()))
                    self.assertEqual(len(data['text']), hparams['batch_size'])
                    self.assertEqual(text_database.vocab.vocab_size,
                                     len(vocab_list) + 3)

            except tf.errors.OutOfRangeError:
                print('Done -- epoch limit reached')
            finally:
                coord.request_stop()
            coord.join(threads)

    def test_paired_text_database(self):
        """Tests the logics of PairedTextDataBase.
        """
        # Create test data
        vocab_list = ['word', '词']
        vocab_file = tempfile.NamedTemporaryFile()
        vocab_file.write('\n'.join(vocab_list).encode("utf-8"))
        vocab_file.flush()

        src_text = ['This is a source sentence .', 'source: 词 词 。']
        src_text_file = tempfile.NamedTemporaryFile()
        src_text_file.write('\n'.join(src_text).encode("utf-8"))
        src_text_file.flush()

        tgt_text = ['This is a target sentence .', 'target: 词 词 。']
        tgt_text_file = tempfile.NamedTemporaryFile()
        tgt_text_file.write('\n'.join(tgt_text).encode("utf-8"))
        tgt_text_file.flush()

        # Construct database
        hparams = {
            "num_epochs": 3,
            "batch_size": 3,
            "source_dataset": {
                "files": [src_text_file.name],
                "vocab.file": vocab_file.name,
            },
            "target_dataset": {
                "files": [tgt_text_file.name],
                "vocab.share_with_source": True,
                "reader.share_with_source": True,
                "processing": {
                    "eos_token": "<TARGET_EOS>"
                }
            }
        }

        text_database = database.PairedTextDataBase(hparams)
        text_data_batch = text_database()

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            sess.run(tf.tables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                while not coord.should_stop():
                    # Run the logics
                    data = sess.run(text_data_batch)

                    self.assertEqual(set(data.keys()),
                                     set(text_database.list_items()))
                    self.assertEqual(len(data['source_text']),
                                     hparams['batch_size'])
                    self.assertEqual(len(data['target_text']),
                                     hparams['batch_size'])
                    self.assertEqual(text_database.source_vocab.vocab_size,
                                     len(vocab_list) + 3)
                    self.assertEqual(text_database.target_vocab.vocab_size,
                                     len(vocab_list) + 3)

            except tf.errors.OutOfRangeError:
                print('Done -- epoch limit reached')
            finally:
                coord.request_stop()
            coord.join(threads)

if __name__ == "__main__":
    tf.test.main()

