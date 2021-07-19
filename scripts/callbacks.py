import warnings
from utils.utils import evaluate
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback
import tensorflow as tf
import numpy as np
from keras.models import load_model


class CustomTensorBoard(TensorBoard):
    """ to log the loss after each batch
    """    
    def __init__(self, log_every=1, **kwargs):
        super(CustomTensorBoard, self).__init__(**kwargs)
        self.log_every = log_every
        self.counter = 0
    
    def on_batch_end(self, batch, logs=None):
        self.counter+=1
        if self.counter%self.log_every==0:
            for name, value in logs.items():
                if name in ['batch', 'size']:
                    continue
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = value.item()
                summary_value.tag = name
                self.writer.add_summary(summary, self.counter)
            self.writer.flush()
        
        super(CustomTensorBoard, self).on_batch_end(batch, logs)


class CustomModelCheckpoint(ModelCheckpoint):
    """ to save the template model, not the multi-GPU model
    """
    def __init__(self, model_to_save, addtion_save, valid_data, labels, **kwargs):
        super(CustomModelCheckpoint, self).__init__(**kwargs)
        self.model_to_save = model_to_save
        self.addtion_save = addtion_save
        self.valid_data = valid_data
        self.labels = labels
        self.best_mAP = -1
        self.best_mSP = -1

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        # print(self.save_weights_only)
                        if self.save_weights_only:
                            self.model_to_save.save_weights(filepath, overwrite=True)
                        else:
                            self.model_to_save.save(filepath, overwrite=True)

                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            # # normal save
            # if self.verbose > 0:
            #     print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
            # if self.save_weights_only:
            #     self.model_to_save.save_weights(filepath, overwrite=True)
            # else:
            #     self.model_to_save.save(filepath, overwrite=True)
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model_to_save.save_weights(filepath, overwrite=True)
                else:
                    self.model_to_save.save(filepath, overwrite=True)
        if self.addtion_save:
            if (epoch+1) % 5 == 0 and (epoch+1)>19:
                average_precisions = evaluate(self.model_to_save, self.valid_data)
                ap = []
                print('[INFO] Epoch: %05d' % (epoch + 1))
                # print the mAP score
                for label, average_precision in average_precisions.items():
                    print('\n'+ self.labels[label] + ' average precision(AP): {:.6f}'.format(average_precision['ap']))
                    ap.append(average_precision['ap'])
                    print(self.labels[label] + ' recall: {:.6f}'.format(average_precision['recall']))
                    print(self.labels[label] + ' precision: {:.6f}'.format(average_precision['precision']))
                mAP = sum(ap) / len(ap)
                print('[INFO] mAP: {:.6f}'.format(mAP))
                if self.best_mAP < mAP:
                    print('[INFO] Best mAP improve from {:.6f} to {:.6f}'.format(self.best_mAP, mAP))
                    self.best_mAP = mAP
                    self.model_to_save.save(str(self.addtion_save).split('.')[0] + '_mAP_best.h5', overwrite=True)
                else:
                    print('[INFO] Best mAP did not improve from {:.6f}'.format(self.best_mAP))
            if (epoch+1) % 10 == 0:
                self.model_to_save.save(str(self.addtion_save).split('.')[0] + '_%04d.h5' % (epoch + 1), overwrite=True)
        super(CustomModelCheckpoint, self).on_batch_end(epoch, logs)




