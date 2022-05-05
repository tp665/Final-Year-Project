from datetime import datetime
import json
import random
import os
from reconstructor import Reconstructor
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch
import torchvision
import gc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import warnings
import utils
import numpy as np
from scipy.optimize import minimize


class Optimiser():
    def __init__(self):
        print("Optimiser created!")

    def optimise(self, init_iterations, iterations):
        dataset = 'dataset'
        test_directory = os.path.join(dataset, 'test')

        img_dims = 224
        batch_size = 1

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # load target data and labels
        image_transforms = { 
            'test': transforms.Compose([
                transforms.Resize(size=img_dims),
                transforms.CenterCrop(size=img_dims),
                transforms.ToTensor(),
            ])
        }
        training_batch_size = 256
        data = {
            'test': datasets.ImageFolder(root=test_directory, transform=image_transforms['test'])
        }
        test_data_loader = DataLoader(data['test'], batch_size=training_batch_size, shuffle=False)
        target_data = []
        target_labels = []
        chosen_images = [394]
        # transform img to tensor
        all_inputs = []
        all_labels = []
        for j, (inputs, labels) in enumerate(test_data_loader):
            all_inputs += inputs
            all_labels += labels
        target_data = [all_inputs[i][None,:].to(device) for i in chosen_images][:batch_size]
        target_labels = [torch.tensor([all_labels[i].item()]).to(device) for i in chosen_images][:batch_size]
        del(all_inputs)
        del(all_labels)
        gc.collect()

        # will load from file if it already exists
        file_name = "".join([i if i.isalnum() else "_" for i in datetime.now().isoformat()]) + "_results.json"
    
        results = []
        if(os.path.isfile(file_name)):
            results = json.load(open(file_name,"r"))
            print(results)

        config = dict(
            iterations=20000,
            group_size=4,
            num_classes=10,
            optimiser='AdamW',
            metric='emd',
            img_dims=img_dims,
            model='ResNet-18'
        )

        for iteration in range(init_iterations):
            params = np.random.uniform([0,0,0,0,0,0], [1,1,1,0.1,1,1])
            config['params'] = params
            reconstructor = Reconstructor(config)
            print("Init Iteration:", iteration)
            print(config['params'])
            result_evals = []
            for i in range(3):
                result_evals.append(-reconstructor.reconstruct(target_data, target_labels))
            result_evals = torch.tensor(result_evals)
            result = result_evals.mean().item()
            print("Result:", result)
            results += [dict(
                    result=result,
                    params={
                    'g_scalar': params[0],
                    'cs_scalar': params[1],
                    'tv_scalar': params[2],
                    'l2n_scalar': params[3],
                    'gr_scalar': params[4],
                    'bn_scalar': params[5]
                    }
                )]
            json.dump(
                results,
                open(file_name,"w"),
                indent=4
            )
        # define kernel
        kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))

        param_names = ['g_scalar', 'cs_scalar', 'tv_scalar', 'l2n_scalar', 'gr_scalar', 'bn_scalar']
        X_train = []
        Y_train = []
        for result in results:
            X_train.append([result['params'][i] for i in param_names])
            Y_train.append([result['result']])

        for iteration in range(iterations):
            # build gaussian process
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=5,
            )
            # train gp
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp.fit(X_train, Y_train)
            
            eval_max = None
            arg_max = None
            acq_max = None

            random_samples_n = 10000
            random_samples = np.random.uniform([0,0,0,0,0,0], [1,1,1,0.1,1,1],size=(random_samples_n, 6))
            random_evaluations = [utils.ucb(i.reshape(1, -1), gp) for i in random_samples]
            arg_max = np.argmax(random_evaluations)
            if(arg_max < len(random_evaluations)):
                eval_max = random_samples[arg_max]
            else:
                print("Issue with argmax")
            acq_max = np.max(eval_max)

            opt_samples_n = 50
            opt_samples = np.random.uniform([0,0,0,0,0,0], [1,1,1,0.1,1,1],size=(opt_samples_n, 6))
            opt_samples = np.reshape(opt_samples, (opt_samples_n, 6))
            if(eval_max is not None):
                opt_samples = np.append(eval_max, opt_samples)
                opt_samples = opt_samples.reshape(opt_samples_n+1, 6)
            else:
                opt_samples.reshape(opt_samples_n, 6)

            for sample in opt_samples:
                # Find the minimum of minus the acquisition function
                m = minimize(lambda x: -utils.ucb(x.reshape(1, -1), gp),
                            sample.reshape(1, -1),
                            bounds=[(0,1), (0,1), (0,1), (0, 0.1), (0,1), (0,1)],
                            method="Nelder-Mead")

                if not m.success:
                    continue

                if acq_max is None or -m.fun >= acq_max:
                    eval_max = m.x
                    acq_max = -m.fun

            eval_max = np.clip(eval_max, [0,0,0,0,0,0], [1,1,1,0.1,1,1])
            config['params'] = eval_max
            reconstructor = Reconstructor(config)
            print("Optim Iteration:", iteration)
            print(config['params'])
            result_evals = []
            for i in range(3):
                result_evals.append(-reconstructor.reconstruct(target_data, target_labels))
            result_evals = torch.tensor(result_evals)
            result = result_evals.mean().item()
            print("Result:", result)
            results += [dict(
                    result=result,
                    params={
                    'g_scalar': params[0],
                    'cs_scalar': params[1],
                    'tv_scalar': params[2],
                    'l2n_scalar': params[3],
                    'gr_scalar': params[4],
                    'bn_scalar': params[5]
                    }
                )]
            json.dump(
                results,
                open(file_name,"w"),
                indent=4
            )
            X_train.append(eval_max)
            Y_train.append([result])

        
        





        
        


            
            