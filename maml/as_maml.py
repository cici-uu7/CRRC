# https://github.com/tristandeleu/pytorch-maml

import torch
import math
import os
import time
import json
import logging

from torchmeta.utils.data import BatchMetaDataLoader

from maml.datasets import get_benchmark_by_name
from maml.metalearners.maml_sharp import ModelAgnosticMetaLearning

from sam import SAM
from sam_folder.model.smooth_cross_entropy import smooth_crossentropy
from sam_folder.utility.bypass_bn import enable_running_stats, disable_running_stats
from sam_folder.model.wide_res_net import WideResNet
from sam_folder.utility.step_lr import StepLR


class input_data():
    def __init__(self):
        self.folder = '/data'
        self.dataset0 = "omniglot"
        self.dataset1 = "miniimagenet"
        self.dataset2 = "doublemnist"
        self.dataset3 = "triplemnist"
        self.output_folder = '/data/results'
        self.num_ways = 10
        self.num_shots = 1
        self.num_shots_test = 15
        self.hidden_size = 64
        self.batch_size = 4
        self.num_steps = 5
        self.num_epochs = 1
        self.num_batches = 100
        self.step_size = 0.001
        self.first_order = True
        self.meta_lr = 0.001
        self.num_workers = 0
        self.verbose = True
        self.use_cuda = False

        self.alpha = 0.05
        self.adap = True
        self.SAM_lower = True
        self.m = 1
        self.delta = 0.000

        self.beta1 = 0.9
        self.beta2 = 0.99
        self.isMomentum = True


args = input_data()

logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
device = torch.device('cuda' if args.use_cuda
                                and torch.cuda.is_available() else 'cpu')

if (args.output_folder is not None):
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
        logging.debug('Creating folder `{0}`'.format(args.output_folder))

folder = os.path.join(args.output_folder, time.strftime('%Y-%m-%d_%H%M%S'))

os.makedirs(folder)
logging.debug('Creating folder `{0}`'.format(folder))

args.folder = os.path.abspath(args.folder)
args.model_path = os.path.abspath(os.path.join(folder, 'model.th'))
# Save the configuration in a config.json file
with open(os.path.join(folder, 'config.json'), 'w') as f:
    json.dump(vars(args), f, indent=2)
logging.info('Saving configuration file in `{0}`'.format(
    os.path.abspath(os.path.join(folder, 'config.json'))))

benchmark = get_benchmark_by_name(args.dataset1,
                                  args.folder,
                                  args.num_ways,
                                  args.num_shots,
                                  args.num_shots_test,
                                  hidden_size=args.hidden_size)

meta_train_dataloader = BatchMetaDataLoader(benchmark.meta_train_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            num_workers=args.num_workers,
                                            pin_memory=True)
meta_val_dataloader = BatchMetaDataLoader(benchmark.meta_val_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=True,
                                          num_workers=args.num_workers,
                                          pin_memory=True)

# meta_optimizer = torch.optim.Adam(benchmark.model.parameters(), lr=args.meta_lr)
base_optimizer = torch.optim.Adam
meta_optimizer = SAM(benchmark.model.parameters(), base_optimizer, rho=args.alpha,
                     adaptive=args.adap, lr=args.meta_lr)

print('\n\ndataset: ', args.dataset1)
print('alpha: ', args.alpha)
print('SAM_lower: ', args.SAM_lower)

metalearner = ModelAgnosticMetaLearning(benchmark.model,
                                        meta_optimizer,
                                        adap=args.adap,
                                        alpha=args.alpha,
                                        delta=args.delta,
                                        SAM_lower=args.SAM_lower,
                                        first_order=args.first_order,
                                        num_adaptation_steps=args.num_steps,
                                        step_size=args.step_size,
                                        m=args.m,
                                        beta1=args.beta1,
                                        beta2=args.beta2,
                                        isMomentum=args.isMomentum,
                                        beta=args.meta_lr,
                                        loss_function=benchmark.loss_function,
                                        device=device)

best_value = None

logging.getLogger('PIL').setLevel(logging.WARNING)

# Training loop
epoch_desc = 'Epoch {{0: <{0}d}}'.format(1 + int(math.log10(args.num_epochs)))
for epoch in range(args.num_epochs):
    metalearner.train(meta_train_dataloader,
                      max_batches=args.num_batches,
                      verbose=args.verbose,
                      desc='Training',
                      leave=False)
    results = metalearner.evaluate(meta_val_dataloader,
                                   max_batches=args.num_batches,
                                   verbose=args.verbose,
                                   desc=epoch_desc.format(epoch + 1))

    # Save best model
    if 'accuracies_after' in results:
        if (best_value is None) or (best_value < results['accuracies_after']):
            best_value = results['accuracies_after']
            save_model = True
    elif (best_value is None) or (best_value > results['mean_outer_loss']):
        best_value = results['mean_outer_loss']
        save_model = True
    else:
        save_model = False

    if save_model and (args.output_folder is not None):
        with open(args.model_path, 'wb') as f:
            torch.save(benchmark.model.state_dict(), f)

if hasattr(benchmark.meta_train_dataset, 'close'):
    benchmark.meta_train_dataset.close()
    benchmark.meta_val_dataset.close()

meta_test_dataloader = BatchMetaDataLoader(benchmark.meta_test_dataset,
                                           batch_size=args.num_batches,
                                           shuffle=True,
                                           num_workers=args.num_workers,
                                           pin_memory=True)
results = metalearner.evaluate(meta_test_dataloader,
                               max_batches=args.num_batches,
                               verbose=args.verbose,
                               desc='Test')