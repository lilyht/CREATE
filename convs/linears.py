'''
Reference:
https://github.com/hshustc/CVPR19_Incremental_Learning/blob/master/cifar100-class-incremental/modified_linear.py
'''
import math
import torch
from torch import nn
from torch.nn import functional as F


class SimpleLinear(nn.Module):
    '''
    Reference:
    https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py
    '''
    def __init__(self, in_features, out_features, bias=True):
        super(SimpleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, nonlinearity='linear')
        nn.init.constant_(self.bias, 0)

    def forward(self, input):
        # print("linear32 fc forward")
        return {'logits': F.linear(input, self.weight, self.bias)}


class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, nb_proxy=1, to_reduce=False, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features * nb_proxy
        self.nb_proxy = nb_proxy
        self.to_reduce = to_reduce
        self.weight = nn.Parameter(torch.Tensor(self.out_features, in_features))
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)

    def forward(self, input):
        out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))

        if self.to_reduce:
            # Reduce_proxy
            out = reduce_proxies(out, self.nb_proxy)

        if self.sigma is not None:
            out = self.sigma * out

        return {'logits': out}

    def forward_reweight(self, input, cur_task, alpha=0.1, beta=0.0, init_cls=10, inc=10, out_dim=768,
                         use_init_ptm=False):
        for i in range(cur_task + 1):
            if i == 0:
                start_cls = 0
                end_cls = init_cls
            else:
                start_cls = init_cls + (i - 1) * inc
                end_cls = start_cls + inc

            out = 0.0
            for j in range((self.in_features // out_dim)):
                # PTM feature
                if use_init_ptm and j == 0:
                    input_ptm = F.normalize(input[:, 0:out_dim], p=2, dim=1)
                    weight_ptm = F.normalize(self.weight[start_cls:end_cls, 0:out_dim], p=2, dim=1)
                    out_ptm = beta * F.linear(input_ptm, weight_ptm)
                    out += out_ptm
                    continue

                input1 = F.normalize(input[:, j * out_dim:(j + 1) * out_dim], p=2, dim=1)
                weight1 = F.normalize(self.weight[start_cls:end_cls, j * out_dim:(j + 1) * out_dim], p=2, dim=1)
                if use_init_ptm:
                    if j != (i + 1):
                        out1 = alpha * F.linear(input1, weight1)
                        out1 /= cur_task
                    else:
                        out1 = F.linear(input1, weight1)
                else:
                    if j != i:
                        out1 = alpha * F.linear(input1, weight1)
                        out1 /= cur_task
                    else:
                        out1 = F.linear(input1, weight1)

                out += out1

            if i == 0:
                out_all = out
            else:
                out_all = torch.cat((out_all, out), dim=1) if i != 0 else out

        if self.to_reduce:
            # Reduce_proxy
            out_all = reduce_proxies(out_all, self.nb_proxy)

        if self.sigma is not None:
            out_all = self.sigma * out_all

        return {'logits': out_all}


class SplitCosineLinear(nn.Module):
    def __init__(self, in_features, out_features1, out_features2, nb_proxy=1, sigma=True):
        super(SplitCosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = (out_features1 + out_features2) * nb_proxy
        self.nb_proxy = nb_proxy
        self.fc1 = CosineLinear(in_features, out_features1, nb_proxy, False, False)
        self.fc2 = CosineLinear(in_features, out_features2, nb_proxy, False, False)
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
            self.sigma.data.fill_(1)
        else:
            self.register_parameter('sigma', None)

    def forward(self, x):
        out1 = self.fc1(x)
        out2 = self.fc2(x)

        out = torch.cat((out1['logits'], out2['logits']), dim=1)  # concatenate along the channel

        # Reduce_proxy
        out = reduce_proxies(out, self.nb_proxy)

        if self.sigma is not None:
            out = self.sigma * out

        return {
            'old_scores': reduce_proxies(out1['logits'], self.nb_proxy),
            'new_scores': reduce_proxies(out2['logits'], self.nb_proxy),
            'logits': out
        }


def reduce_proxies(out, nb_proxy):
    if nb_proxy == 1:
        return out
    bs = out.shape[0]
    nb_classes = out.shape[1] / nb_proxy
    assert nb_classes.is_integer(), 'Shape error'
    nb_classes = int(nb_classes)

    simi_per_class = out.view(bs, nb_classes, nb_proxy)
    attentions = F.softmax(simi_per_class, dim=-1)

    return (attentions * simi_per_class).sum(-1)


def conv_layer(input_channel, output_channel, kernel_size=1, padding=0, use_activation=True):
    if use_activation:
        res = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, padding=padding, bias=False),
            nn.Tanh()
        )
    else:
        res = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, padding=padding, bias=False),
            # nn.Tanh()
        )
    return res

class AutoEncoder(nn.Module):
    def __init__(self, in_features, hidden_layers, latent_chan):
        super(AutoEncoder, self).__init__()
        self.in_features = in_features
        self.hidden_layers = hidden_layers
        self.latent_size = latent_chan
        layer_block = conv_layer
        self.encode_convs = []
        self.decode_convs = []
        for i in range(len(hidden_layers)):
            h = hidden_layers[i]
            ecv = layer_block(in_features, h, )
            dcv = layer_block(h, in_features, use_activation=i != 0)
            in_features = h
            self.encode_convs.append(ecv)
            self.decode_convs.append(dcv)
        self.encode_convs = nn.ModuleList(self.encode_convs)
        self.decode_convs.reverse()
        self.decode_convs = nn.ModuleList(self.decode_convs)
        self.latent_conv = layer_block(in_features, latent_chan)
        self.latent_deconv = layer_block(latent_chan, in_features)
        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.kaiming_uniform_(self.weight, nonlinearity='linear')
        # nn.init.constant_(self.bias, 0)
        pass

    def forward(self, input):
        for cv in self.encode_convs:
            input = cv(input)
        output_manifold = self.latent_conv(input)  # (bs,32,1,1)
        output = self.latent_deconv(output_manifold)  # (bs,64,1,1)
        for cv in self.decode_convs:
            output = cv(output)
        return {'x_reconstruct': output, 'fm': output_manifold, 'logits': output}

class CSSRClassifier(nn.Module):

    def __init__(self, args, inchannels, num_class):
        super().__init__()
        self.args = args
        ae_hidden = self.args["hidden_layers"]
        self.ae_latent = self.args["ae_latent"]
        self.class_aes = []
        for i in range(num_class):
            ae = AutoEncoder(inchannels, ae_hidden, self.ae_latent)
            self.class_aes.append(ae)
        self.class_aes = nn.ModuleList(self.class_aes)
        self.reduction = -1
        self.gamma = 0.1
        self.reduction *= self.gamma
        self.arch_type = 'softmax_avg'
        self.avg_order = {"avg_softmax": 1, "softmax_avg": 2}[self.arch_type]  # default softmax_avg: 2
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def ae_error(self, rc, x):
        return torch.norm(rc - x, p=1, dim=1, keepdim=True) * self.reduction

    def forward(self, x):
        features = x
        if len(x.shape) == 2:
            x = torch.unsqueeze(x, 2)
            x = torch.unsqueeze(x, 3)
        cls_ers = []
        cls_fms = []
        cls_rex = []
        for ae in self.class_aes:
            rc = ae(x)  # rc['x_reconstruct'].shape torch.Size([bs, 64, 1, 1])
            cls_er = self.ae_error(rc['x_reconstruct'], x)  # torch.Size([bs, 1, 1, 1])
            cls_er = torch.clamp(cls_er, -500, 500)
            cls_ers.append(cls_er)
            cls_fms.append(torch.squeeze(rc['fm']))  # torch.Size([bs, 32])
            cls_rex.append(torch.squeeze(rc['x_reconstruct']))
        logits = torch.cat(cls_ers, dim=1)  # torch.Size([bs, 50, 1, 1])
        fm = torch.stack(cls_fms, dim=1)  # torch.Size([bs, 50, 32])
        recon = torch.stack(cls_rex, dim=1)
        if self.avg_order == 1:
            g = self.avg_pool(logits).view(logits.shape[0], -1)
            g = torch.softmax(g, dim=1)
        elif self.avg_order == 2:  # default
            epsilon = 1e-10
            g = torch.softmax(logits, dim=1) + epsilon  # torch.Size([bs, 50, 1, 1])
            g = self.avg_pool(g).view(logits.size(0), -1)  # torch.Size([bs, 50])
        return {"features": features, "logits": g, "error": torch.squeeze(self.avg_pool(logits).view(logits.size(0), -1)), "fm": fm, "recon": recon}

'''
class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)

    def forward(self, input):
        out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        if self.sigma is not None:
            out = self.sigma * out
        return {'logits': out}


class SplitCosineLinear(nn.Module):
    def __init__(self, in_features, out_features1, out_features2, sigma=True):
        super(SplitCosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features1 + out_features2
        self.fc1 = CosineLinear(in_features, out_features1, False)
        self.fc2 = CosineLinear(in_features, out_features2, False)
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
            self.sigma.data.fill_(1)
        else:
            self.register_parameter('sigma', None)

    def forward(self, x):
        out1 = self.fc1(x)
        out2 = self.fc2(x)

        out = torch.cat((out1['logits'], out2['logits']), dim=1)  # concatenate along the channel
        if self.sigma is not None:
            out = self.sigma * out

        return {
            'old_scores': out1['logits'],
            'new_scores': out2['logits'],
            'logits': out
        }
'''
