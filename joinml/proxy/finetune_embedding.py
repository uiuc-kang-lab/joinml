import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch


"""
pytorch implementation of 
1. supervised contrastive loss: https://arxiv.org/pdf/2004.11362.pdf, when labels are provided
2. unsupervised contrastive loss: https://arxiv.org/pdf/2002.05709.pdf, when labels are not provided
"""
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        print(features.shape)
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        print("tile mask", mask)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        print("logits mask", logits_mask)
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        print("log_prob", log_prob)
        print("mask", mask)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        print("mean_log_prob_pos", mean_log_prob_pos)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class ContrasNet(nn.Module):
    def __init__(self, encoder: nn.Module, in_dim: int, out_dim: int=None, head: str="mlp"):
        super(ContrasNet, self).__init__()
        self.encoder = encoder
        self.in_dim = in_dim
        self.out_dim = out_dim
        if head == "mlp":
            assert self.out_dim is not None
            self.head = nn.Sequential(
                nn.Linear(self.in_dim, self.in_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.in_dim, self.out_dim)
            )
        elif head == "linear":
            assert self.out_dim is not None
            self.head = nn.Linear(self.in_dim, self.out_dim)
        elif head == "none":
            self.head = nn.Identity()
            assert self.in_dim == self.out_dim or self.out_dim is None
        else:
            raise ValueError("Invalid head: {}".format(head))

    def forward(self, x):
        x = self.encoder(x)
        if isinstance(x, dict):
            x = x["sentence_embedding"]
        print(f"encoding {x.shape}\n", x)
        x = F.normalize(x, dim=1)
        print(f"normalizing {x.shape}\n", x)
        x = self.head(x)
        print(f"head {x.shape}\n", x)
        x = F.normalize(x, dim=1)
        print(f"normalizing {x.shape}\n", x)
        return x
    
class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0 
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val 
        self.sum += val * n 
        self.count += n 
        self.avg = self.sum / self.count

class Trainer:
    def __init__(self, encoder: nn.Module, embedding_dim: int, feature_dim: int, head: str="mlp", temp: float=0.07):
        self.model = ContrasNet(encoder, in_dim=embedding_dim, out_dim=feature_dim, head=head)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = SupConLoss(temperature=temp)
    
    def train(self, train_loader: DataLoader, epochs: int=10, device: str="cpu", modality: str="text"):
        self.model.train()
        self.model.to(device)
        losses = AverageMeter()

        for epoch in range(epochs):

            for idx, (data, labels) in enumerate(train_loader):
                labels = labels.to(device)
                
                if modality == "text":
                    data = {"input_ids": data[0], "attention_mask": data[1]}
                    num_view = 1

                elif modality == "image":
                    # data may have multiple views
                    num_view = len(data)
                    data = torch.cat(data, dim=0).to(device)

                features = self.model(data)
                
                if num_view == 2:
                    f1, f2 = torch.split(features, [data.shape[0]//2, data.shape[0]//2], dim=0)
                    features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                elif num_view == 1:
                    features = features.unsqueeze(1)
                
                loss = self.criterion(features, labels)
                print(loss)

                losses.update(loss.item(), features.shape[0])

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if idx % 100 == 0:
                    print("Epoch: [{0}][{1}/{2}]\t"
                          "Loss {loss.val:.4f} ({loss.avg:.4f})".format(epoch, idx, len(train_loader), loss=losses))
            
                exit()

            print("Epoch: [{0}]\t"
                  "Loss {loss.avg:.4f}".format(epoch, loss=losses))
        self.model.eval()
        self.model.to("cpu")

        return self.model.encoder, losses.avg









            
        

