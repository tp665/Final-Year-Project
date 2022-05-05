import torch
import utils
from tqdm import tqdm
from LeNet import LeNet
from swd import swd
from ResNet import get_resnet
import lpips
from torchvision.utils import save_image

class Reconstructor():
    def __init__(self, config):
        if('device' not in config):
            config['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if(not utils.verify_config(config)):
            print("Config does not contain all required values")
            raise Exception("Invalid config")
        self.config = config
    
    def reconstruct(self, target_data, target_labels, save=False):
        device = self.config['device']
        g_scalar, cs_scalar, tv_scalar, l2n_scalar, gr_scalar, bn_scalar = self.config['params']
        max_iterations = self.config['iterations']
        group_size = self.config['group_size']
        img_dims = self.config['img_dims']
        num_classes = self.config['num_classes']

        if('model' not in self.config):
            self.config['model'] = 'ResNet-18'

        if(self.config['model'] == 'LeNet'):
            # define net
            if(img_dims == 16):
                in_features = 192
            elif(img_dims == 64):
                in_features = 3072
            else:
                in_features = 768

            bn_layer_stats = []
            net = LeNet(num_classes,in_features)
        else:
            net, bn_layer_stats = get_resnet(device, img_dims, num_classes)
        net.to(device)
        
        batch_size = len(target_data)

        # compute original gradient
        original_dy_dx = None
        bn_priors = []
        for i in range(batch_size):
            # make prediction
            pred = net(target_data[i])

            # calculate gradients
            y = utils.cross_entropy_for_onehot(pred, utils.label_to_onehot(target_labels[i], num_classes))
            dy_dx = torch.autograd.grad(y, net.parameters())
            dy_dx = list((_.detach().clone() for _ in dy_dx))
            if(not original_dy_dx):
                original_dy_dx = dy_dx
            else:
                dy_dx = zip(original_dy_dx, dy_dx)
                original_dy_dx = []
                for a, b in dy_dx:
                    original_dy_dx.append(torch.add(a,b).to(device))
            # get bn statistics
            bn_prior = []
            for idx, mod in enumerate(bn_layer_stats):
                mean_var = mod.mean_var[0].detach(), mod.mean_var[1].detach()
                bn_prior.append(mean_var)
            bn_priors.append(bn_prior)

        # average gradients over batch size
        original_dy_dx = [i/batch_size for i in original_dy_dx]

        # batch-wise average of bn mean and variances
        bn_prior = []
        for layers in zip(*bn_priors):
            means = None
            vs = None
            for i, [mean, var] in enumerate(layers):
                if(i == 0):
                    means = mean
                    vs = var
                else:
                    means = torch.add(means, mean)
                    vs = torch.add(vs, var)

            mean = means/len(layers)
            var = vs/len(layers)
            mean_var = mean, var
            bn_prior.append(mean_var)

        # generate dummy data and label
        dummy_data = [torch.randn(batch_size, *target_data[0].size()).to(device).requires_grad_(True) for i in range(group_size)]

        # predict labels
        label_pred = torch.argsort(original_dy_dx[-2].sum(dim=1), dim=-1)[:batch_size].to(device)
        label_pred = torch.cat([utils.label_to_onehot(i, num_classes) for i in torch.split(label_pred, 1)]).to(device)

        mse = torch.nn.MSELoss(reduction='sum').to(device)

        # define optimization closure
        def gradient_closure(curr_optimiser, data, target_dy_dx, labels, include_gr=False):
            def closure():
                for param in net.parameters():
                    param.grad = None
                curr_optimiser.zero_grad(set_to_none=True)

                dummy_dy_dx = None
                dummy_bn_stats = []
                for i in range(batch_size):
                    dummy_pred = net(data[i]) 
                    dummy_loss = torch.mean(torch.sum(-labels[i] * torch.nn.functional.log_softmax(dummy_pred, dim=-1), 1))
                    curr_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
                    if(i == 0):
                        dummy_dy_dx = curr_dy_dx
                    else:
                        curr_dy_dx = zip(dummy_dy_dx, curr_dy_dx)
                        dummy_dy_dx = []
                        for a, b in curr_dy_dx:
                            dummy_dy_dx.append(a + b)
                    dummy_bn = []
                    for mod in bn_layer_stats:
                        mean_var = mod.mean_var[0], mod.mean_var[1]
                        dummy_bn.append(mean_var)
                    dummy_bn_stats.append(dummy_bn)

                dummy_dy_dx = [i/batch_size for i in dummy_dy_dx]
                    
                dummy_bn = []
                for layers in zip(*dummy_bn_stats):
                    means = 0
                    vs = 0
                    for [mean, var] in layers:
                        means += mean
                        vs += var
                    mean = means/len(layers)
                    var = vs/len(layers)
                    mean_var = mean, var
                    dummy_bn.append(mean_var)
                
                # compute loss between computed grad for gradient descent step and expected from intercepted target weights
                grad_diff = 0
                c_s = 0
                for gx, gy in zip(dummy_dy_dx, target_dy_dx):
                    grad_diff += mse(gx, gy)
                    gx = gx.flatten()
                    gy = gy.flatten()
                    c_s += (1 - (torch.dot(gx, gy)/(torch.norm(gx, p=2) * torch.norm(gy, p=2))))

                
                # total variation
                tv = 0
                for image in data:
                    tv_batch_size, channels, height, width = (1,3,224,244)
                    tv_height = torch.pow(image[:,:,1:,:]-image[:,:,:-1,:], 2).sum()
                    tv_width = torch.pow(image[:,:,:,1:]-image[:,:,:,:-1], 2).sum()
                    tv += torch.div(torch.add(tv_height, tv_width),torch.mul(torch.mul(tv_batch_size,channels), torch.mul(height,width)))
                tv = tv/batch_size
                
                # l2 penalisation
                l2_norm = torch.norm(data, p=2)/batch_size
                
                # group registration
                gr = 0
                if(include_gr):
                    group_means = []
                    for datum_i in range(batch_size):
                        whole_group = []
                        for i in range(group_size):
                            whole_group.append(dummy_data[i][datum_i])
                        group_means.append(torch.mean(torch.stack(whole_group), dim=0))
                    for datum_i in range(batch_size):
                        gr += torch.norm(torch.sub(data[datum_i], group_means[datum_i]), p=2)
                gr = gr/batch_size


                # due to the hook we added, this updates whenever we call "forward" on the net
                # i.e make a prediction
                bn = 0
                for i, (my, pr) in enumerate(zip(dummy_bn, bn_prior)):
                    rescale = 10 if i == 0 else 1
                    bn += rescale * torch.norm(torch.log((torch.sqrt(pr[1])/torch.sqrt(my[1]))) + (my[1] + (my[0] - pr[0])**2)/(2*pr[1]) - 0.5)

                loss = g_scalar*grad_diff + l2n_scalar*l2_norm + tv_scalar*tv + bn_scalar*bn + gr_scalar*gr + cs_scalar*c_s
                loss.backward()
                return loss
            return closure

        # run process
        # iteration parameters
        max_iterations = max_iterations

        optimiser = {
            'Adam': torch.optim.Adam,
            'L-BFGS': torch.optim.LBFGS,
            'AdamW': torch.optim.AdamW
        }.get(self.config['optimiser'], torch.optim.AdamW)

        optimisers = [optimiser([dummy_data[i]], lr=0.1) for i in range(group_size)]
        schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(i, max_iterations) for i in optimisers]

        for iteration in tqdm(range(max_iterations)):
            for i in range(group_size):
                cl = gradient_closure(optimisers[i], dummy_data[i], original_dy_dx, label_pred, (iteration>(max_iterations/4) and iteration<((3*max_iterations)/4)))
                optimisers[i].step(cl)
                schedulers[i].step()

        
        mse_for_psnr = torch.nn.MSELoss(reduction='mean').to(device)
        best_psnrs = []
        best_emds = []
        best_lpips = []
        lpips_metric = lpips.LPIPS(net='vgg')
        for i in range(group_size):
            for j, (d, target) in enumerate(zip(dummy_data[i], target_data)):
                m = mse_for_psnr(d, target)
                if(m == 0):
                    psnr = 100
                else:
                    psnr = 20*torch.log10(torch.tensor(1)/torch.sqrt(m))
                # take multiple evluations
                emds = []
                for t in range(100):
                    emds.append(swd(d,target, device=device))
                emds = torch.tensor(emds)
                emd = emds.mean()

                lpips_result = lpips_metric((target*2)-1, (d*2)-1)
                if(i == 0):
                    best_psnrs.append(psnr.item())
                    best_emds.append(emd.item())
                    best_lpips.append(lpips_result.item())
                else:
                    best_psnrs[j] = max(best_psnrs[j], psnr.item())
                    best_emds[j] = min(best_emds[j], emd.item())
                    best_lpips[j] = min(best_lpips[j], lpips_result.item())
        
        if(save):
            for i in range(group_size):
                for j in range(batch_size):
                    save_image(dummy_data[i][j], "result_g"+str(i)+"_b"+str(j)+".png")

        return {
            'emd': sum(best_emds)/batch_size,
            'psnr': sum(best_psnrs)/batch_size,
            'lpips': sum(best_lpips)/batch_size
        }.get(self.config['metric'], sum(best_emds)/batch_size)
