import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function


class GradientReversal_function(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = - alpha*grad_output
        return grad_input, None
revgrad = GradientReversal_function.apply


class GradientReversal(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = torch.tensor(alpha, requires_grad=False)
        self.revgrad = GradientReversal_function.apply

    def forward(self, x):
        return self.revgrad(x, self.alpha)


# a model rapper for local model with adversarial component
class Local_Adversarial_combined_model(nn.Module):
    def __init__(self, local_model, adversarial_model):
        super(Local_Adversarial_combined_model, self).__init__()
        assert local_model != None and adversarial_model != None
        self.local_model = local_model # for normal training
        self.adversarial_model = adversarial_model
        self.adversarial_output = None
        # self.adversarial_loss = None

    def forward(self,x):
        out = self.local_model(x)
        self.adversarial_output = self.adversarial_model(out)
        return out


# for Attribute Inference adversarial
class Adversarial_MLP3(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(Adversarial_MLP3, self).__init__()
        self.drop_out_rate = 0.1
        self.grad_reverse = nn.Sequential(
            GradientReversal(alpha=1.),
            nn.Dropout(self.drop_out_rate),
        )
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim*2, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim*2),
            nn.Dropout(self.drop_out_rate),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(self.drop_out_rate),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, output_dim, bias=False),
            # nn.ReLU(inplace=True),
            # nn.BatchNorm1d(output_dim),
            # nn.Dropout(self.drop_out_rate),
        )

    def forward(self, x):
        x = self.grad_reverse(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x