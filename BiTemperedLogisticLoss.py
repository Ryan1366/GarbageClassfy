import torch
import torch.nn as nn
import torch.nn.functional as F

class Bi_Tempered_Logistic_Loss(nn.Module):
    """Sparse Bi-Tempered Logistic Loss with custom gradient.
    Args:
    activations: A multi-dimensional tensor with last dimension `num_classes`.
    labels: A tensor with dtype of int32.
    t1: Temperature 1 (< 1.0 for boundedness).
    t2: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
    num_iters: Number of iterations to run the method.
    label_smoothing: Label smoothing parameter between [0, 1).
    Returns:
    A loss tensor.

    """
    def __init__(self, t1 = 0.8, t2 = 1.2, label_smoothing = 0.0, num_iters = 5):
        super(Bi_Tempered_Logistic_Loss, self).__init__()
        self.t1 = t1
        self.t2 = t2
        self.label_smoothing = label_smoothing
        self.num_iters = num_iters

    def log_t(self, u, t):
        if t==1.0:
            return torch.log(u)
        else:
            return (u**(1.0-t)-1.0)/(1.0-t)

    def exp_t(self,u, t):
        if t==1.0:
            return torch.exp(u)
        else:
            return torch.relu(1.0+(1.0-t)*u)**(1.0/(1.0-t))

    def compute_normalization_fixed_point(self,activations):
        mu=torch.max(activations,dim=-1).values.view(-1, 1)
        normalized_activations_step_0=activations-mu
        normalized_activations = normalized_activations_step_0
        i = 0
        while i < self.num_iters:
            i+=1
            logt_partition=torch.sum(self.exp_t(normalized_activations,self.t2),dim=-1).view(-1,1)
            normalized_activations=normalized_activations_step_0*(logt_partition**(1.0-self.t2))
        logt_partition=torch.sum(self.exp_t(normalized_activations,self.t2),dim=-1).view(-1,1)
        return -self.log_t(1.0/logt_partition,self.t2) + mu

    def compute_normalization(self,activations):
        if self.t2 < 1.0:
            return None
        else:
            return self.compute_normalization_fixed_point(activations)

    def tempered_softmax(self,activations):
        if self.t2 == 1.0:
            normalization_constants=torch.log(torch.sum(torch.exp(activations),dim=-1))
        else:
            normalization_constants=self.compute_normalization(activations)
        return self.exp_t(activations-normalization_constants,self.t2)

    def forward(self, activations, labels):
        if self.label_smoothing > 0.0:
            num_classes=labels.shape[-1]
            labels=(1-num_classes/(num_classes-1)*self.label_smoothing)*labels + self.label_smoothing/(num_classes-1)
        probabilities=self.tempered_softmax(activations)
        temp1=(self.log_t(labels+1e-10,self.t1) - self.log_t(probabilities,self.t1))*labels
        temp2=(1/(2 - self.t1))*(torch.pow(labels,2 - self.t1)-torch.pow(probabilities,2-self.t1))
        loss_values=temp1-temp2
        return torch.sum(loss_values,dim=-1)