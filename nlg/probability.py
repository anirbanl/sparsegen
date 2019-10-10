import torch
from torch.autograd import Variable
import time

__all__ = ['Sparsemax', 'MySoftmax', 'Sparsegen-lin', 'Sparsegen-exp', 'Sparsegen-sq', 'Sparsegen-poly', 'Sparsegen-scale', 'Sparsegen-log',
           'MySumNorm', 'MySphericalSoftmax', 'Sparsecone', 'SparseHourglass']

class MySoftmax(torch.nn.Module):
    def __init__(self, temp=1.0):
        super(MySoftmax, self).__init__()
        self.sm = torch.nn.Softmax()
        self.temperature = temp

    def forward(self, input):
        z = input / self.temperature
        return self.sm(z)


class Sparsemax(torch.nn.Module):
    def __init__(self):
        super(Sparsemax, self).__init__()
        # self.lambda = sparsemax()

    def forward(self, input):
        bs = input.data.size()[0]
        dim = input.data.size()[1]
        dtype = torch.FloatTensor
        z = input.type(dtype)

        #sort z
        z_sorted = torch.sort(z, descending=True)[0]

        #calculate k(z)
        z_cumsum = torch.cumsum(z_sorted, dim=1)
        k = Variable(torch.arange(1, dim + 1).unsqueeze(0).repeat(bs,1))
        z_check = torch.gt(1 + k * z_sorted, z_cumsum)

        # because the z_check vector is always [1,1,...1,0,0,...0] finding the
        # (index + 1) of the last `1` is the same as just summing the number of 1.
        k_z = torch.sum(z_check.float(), dim = 1)

        #calculate tau(z)
        tausum = torch.sum(z_check.float() * z_sorted, dim=1)
        tau_z = (tausum - 1) / k_z
        if torch.cuda.is_available():
            dtype = torch.cuda.FloatTensor
        prob = z.sub(tau_z.view(bs,1).repeat(1,dim)).clamp(min=0).type(dtype)
        return prob

class Sparsegen_lin(torch.nn.Module):
    def __init__(self, lam, data_driven=False, normalized=True):
        super(Sparsegen-lin, self).__init__()
        self.lam = lam
        self.data_driven = data_driven
        self.normalized = normalized

    def forward(self, input):
        bs = input.data.size()[0]
        dim = input.data.size()[1]
        # z = input.sub(torch.mean(input,dim=1).repeat(1,dim))
        dtype = torch.FloatTensor
        z = input.type(dtype)

        if torch.cuda.is_available():
            dtype = torch.cuda.FloatTensor
        #Calculate data-driven lambda for self.data_driven = True
        if self.data_driven:
            z = z - torch.min(z,1)[0].view(bs,1).repeat(1,dim)
            tausum = torch.sum(z, 1)
            self.lam = (1 - tausum).view(bs,1).repeat(1,dim)
            prob = z
            if self.normalized:
                prob = z / (1 - self.lam)
            return prob.type(dtype)

        #sort z
        z_sorted = torch.sort(z, descending=True)[0]

        #calculate k(z)
        z_cumsum = torch.cumsum(z_sorted, dim=1)
        k = Variable(torch.arange(1, dim + 1).unsqueeze(0).repeat(bs,1))
        z_check = torch.gt(1 - self.lam + k * z_sorted, z_cumsum)

        # because the z_check vector is always [1,1,...1,0,0,...0] finding the
        # (index + 1) of the last `1` is the same as just summing the number of 1.
        k_z = torch.sum(z_check.float(), 1)

        #calculate tau(z)
        tausum = torch.sum(z_check.float() * z_sorted, 1)
        tau_z = (tausum - 1 + self.lam) / k_z
        prob = z.sub(tau_z.view(bs,1).repeat(1,dim)).clamp(min=0).type(dtype)
        if self.normalized:
               prob /= (1-self.lam)
        return prob

'''
This formulation has some discontinuity issues around lambda=0.
class MySparseProp(torch.nn.Module):
    def __init__(self, lam=1.0, normalized=True):
        super(MySparseProp, self).__init__()
        self.normalized = normalized

    def forward(self, input):
        bs = input.data.size()[0]
        dim = input.data.size()[1]
        # z = input.sub(torch.mean(input,dim=1).repeat(1,dim))
        dtype = torch.FloatTensor
        z = input - torch.min(input, 1)[0].view(bs, 1).repeat(1, dim).type(dtype)
        z = input.type(dtype)
        zplus1 = z+1
        qvalue = z/zplus1

        if torch.cuda.is_available():
            dtype = torch.cuda.FloatTensor

        #sort z
        z_sorted = torch.sort(z, descending=True)[0]
        z_sorted_plus1 = z_sorted+1
        qvalue_sorted = z_sorted/z_sorted_plus1

        #calculate k(z)
        qz = qvalue_sorted*z_sorted
        qz_cumsum = torch.cumsum(qz, dim=1)
        q_cumsum = torch.cumsum(qvalue_sorted, dim=1)
        z_check = torch.gt(1 + q_cumsum * z_sorted, qz_cumsum)

        # because the z_check vector is always [1,1,...1,0,0,...0] finding the
        # (index + 1) of the last `1` is the same as just summing the number of 1.
        k_z = torch.sum(z_check.float() * qvalue_sorted, 1)

        #calculate tau(z)
        tausum = torch.sum(z_check.float() * qz, 1)
        tau_z = (tausum - 1) / k_z
        prob = z.sub(tau_z.view(bs,1).repeat(1,dim)).clamp(min=0).type(dtype)
        if self.normalized:
               prob =  qvalue * prob
        return prob
'''

class Sparsegen_scale(torch.nn.Module):
    def __init__(self, gamma, data_driven=False, normalized=True):
        super(Sparsegen-scale, self).__init__()
        self.data_driven = data_driven
        self.normalized = normalized
        self.lin_coeff = gamma

    def forward(self, input):
        bs = input.data.size()[0]
        dim = input.data.size()[1]
        # z = input.sub(torch.mean(input,dim=1).repeat(1,dim))
        dtype = torch.FloatTensor
        z = self.lin_coeff * input.type(dtype)

        if torch.cuda.is_available():
            dtype = torch.cuda.FloatTensor
        #Calculate data-driven lambda for self.data_driven = True
        if self.data_driven:
            z = input - torch.min(input, 1)[0].view(bs, 1).repeat(1, dim)
            tausum = torch.sum(z, 1)
            prob = z
            if self.normalized:
                prob = z / tausum.view(bs, 1).repeat(1, dim)
            return prob.type(dtype)

        #sort z
        z_sorted = torch.sort(z, descending=True)[0]

        #calculate k(z)
        z_cumsum = torch.cumsum(z_sorted, dim=1)
        k = Variable(torch.arange(1, dim + 1).unsqueeze(0).repeat(bs,1))
        z_check = torch.gt(1 + k * z_sorted, z_cumsum)

        # because the z_check vector is always [1,1,...1,0,0,...0] finding the
        # (index + 1) of the last `1` is the same as just summing the number of 1.
        k_z = torch.sum(z_check.float(), 1)

        #calculate tau(z)
        tausum = torch.sum(z_check.float() * z_sorted, 1)
        tau_z = (tausum - 1) / k_z
        prob = z.sub(tau_z.view(bs, 1).repeat(1, dim)).clamp(min=0).type(dtype)
        if not self.normalized:
            prob /= self.lin_coeff
        return prob

class Sparsegen_exp(torch.nn.Module):
    def __init__(self, lam=0, exp_coeff=1.0, data_driven=False, normalized=True):
        super(Sparsegen-exp, self).__init__()
        self.lam = lam
        self.data_driven = data_driven
        self.normalized = normalized
        self.ec = exp_coeff

    def forward(self, input):
        bs = input.data.size()[0]
        dim = input.data.size()[1]
        # z = input.sub(torch.mean(input,dim=1).repeat(1,dim))
        dtype = torch.FloatTensor
        z = torch.exp(self.ec * input).type(dtype)

        if torch.cuda.is_available():
            dtype = torch.cuda.FloatTensor
        #Calculate data-driven lambda for self.data_driven = True
        if self.data_driven:
            tausum = torch.sum(z, 1)
            self.lam = (1 - tausum).view(bs,1).repeat(1,dim)
            prob = z
            if self.normalized:
                prob = z / (1 - self.lam)
            return prob.type(dtype)


        #sort z
        z_sorted = torch.sort(z, descending=True)[0]

        #calculate k(z)
        z_cumsum = torch.cumsum(z_sorted, dim=1)
        k = Variable(torch.arange(1, dim + 1).unsqueeze(0).repeat(bs,1))
        z_check = torch.gt(1 - self.lam + k * z_sorted, z_cumsum)

        # because the z_check vector is always [1,1,...1,0,0,...0] finding the
        # (index + 1) of the last `1` is the same as just summing the number of 1.
        k_z = torch.sum(z_check.float(), 1)

        #calculate tau(z)
        tausum = torch.sum(z_check.float() * z_sorted, 1)
        tau_z = (tausum - 1 + self.lam) / k_z
        prob = z.sub(tau_z.view(bs,1).repeat(1,dim)).clamp(min=0).type(dtype)
        if self.normalized:
               prob /= (1-self.lam)
        return prob

class Sparsegen_sq(torch.nn.Module):
    def __init__(self, lam=0, data_driven=False, normalized=True):
        super(Sparsegen-sq, self).__init__()
        self.lam = lam
        self.data_driven = data_driven
        self.normalized = normalized

    def forward(self, input):
        bs = input.data.size()[0]
        dim = input.data.size()[1]
        # z = input.sub(torch.mean(input,dim=1).repeat(1,dim))
        dtype = torch.FloatTensor
        z = torch.pow(input,2).type(dtype)

        if torch.cuda.is_available():
            dtype = torch.cuda.FloatTensor
        #Calculate data-driven lambda for self.data_driven = True
        if self.data_driven:
            tausum = torch.sum(z, 1)
            self.lam = (1 - tausum).view(bs,1).repeat(1,dim)
            prob = z
            if self.normalized:
                prob = z / (1 - self.lam)
            return prob.type(dtype)

        #sort z
        z_sorted = torch.sort(z, descending=True)[0]

        #calculate k(z)
        z_cumsum = torch.cumsum(z_sorted, dim=1)
        k = Variable(torch.arange(1, dim + 1).unsqueeze(0).repeat(bs,1))
        z_check = torch.gt(1 - self.lam + k * z_sorted, z_cumsum)

        # because the z_check vector is always [1,1,...1,0,0,...0] finding the
        # (index + 1) of the last `1` is the same as just summing the number of 1.
        k_z = torch.sum(z_check.float(), 1)

        #calculate tau(z)
        tausum = torch.sum(z_check.float() * z_sorted, 1)
        tau_z = (tausum - 1 + self.lam) / k_z
        prob = z.sub(tau_z.view(bs,1).repeat(1,dim)).clamp(min=0).type(dtype)
        if self.normalized:
               prob /= (1-self.lam)
        return prob

class Sparsegen_poly(torch.nn.Module):
    def __init__(self, lam=0, poly=2, data_driven=False, normalized=True):
        super(Sparsegen-poly, self).__init__()
        self.lam = lam
        self.data_driven = data_driven
        self.normalized = normalized
        self.n = poly

    def forward(self, input):
        bs = input.data.size()[0]
        dim = input.data.size()[1]
        # z = input.sub(torch.mean(input,dim=1).repeat(1,dim))
        dtype = torch.FloatTensor
        z = torch.pow(input, self.n).type(dtype)

        if torch.cuda.is_available():
            dtype = torch.cuda.FloatTensor
        #Calculate data-driven lambda for self.data_driven = True
        if self.data_driven:
            if self.n % 2 != 0:
                z = z - torch.min(z, 1)[0].view(bs, 1).repeat(1, dim)
            tausum = torch.sum(z, 1)
            self.lam = (1 - tausum).view(bs,1).repeat(1,dim)
            prob = z
            if self.normalized:
                prob = z / (1 - self.lam)
            return prob.type(dtype)

        #sort z
        z_sorted = torch.sort(z, descending=True)[0]

        #calculate k(z)
        z_cumsum = torch.cumsum(z_sorted, dim=1)
        k = Variable(torch.arange(1, dim + 1).unsqueeze(0).repeat(bs,1))
        z_check = torch.gt(1 - self.lam + k * z_sorted, z_cumsum)

        # because the z_check vector is always [1,1,...1,0,0,...0] finding the
        # (index + 1) of the last `1` is the same as just summing the number of 1.
        k_z = torch.sum(z_check.float(), 1)

        #calculate tau(z)
        tausum = torch.sum(z_check.float() * z_sorted, 1)
        tau_z = (tausum - 1 + self.lam) / k_z
        prob = z.sub(tau_z.view(bs,1).repeat(1,dim)).clamp(min=0).type(dtype)
        if self.normalized:
               prob /= (1-self.lam)
        return prob

class Sparsegen_log(torch.nn.Module):
    #Data-driven NOT ENABLED
    def __init__(self, lam=0, data_driven=False, normalized=True):
        super(Sparsegen-log, self).__init__()
        self.lam = lam
        self.data_driven = data_driven
        self.normalized = normalized
        self.constant = 1e-30

    def forward(self, input):
        bs = input.data.size()[0]
        dim = input.data.size()[1]
        # z = input.sub(torch.mean(input,dim=1).repeat(1,dim))
        dtype = torch.FloatTensor
        z = torch.log(input - torch.min(input,1)[0].view(bs, 1).repeat(1, dim) + self.constant).type(dtype)
        #To prevent underflow and to prevent negative arguments to log

        #sort z
        z_sorted = torch.sort(z, descending=True)[0]

        #calculate k(z)
        z_cumsum = torch.cumsum(z_sorted, dim=1)
        k = Variable(torch.arange(1, dim + 1).unsqueeze(0).repeat(bs,1))
        z_check = torch.gt(1 - self.lam + k * z_sorted, z_cumsum)

        # because the z_check vector is always [1,1,...1,0,0,...0] finding the
        # (index + 1) of the last `1` is the same as just summing the number of 1.
        k_z = torch.sum(z_check.float(), 1)

        #calculate tau(z)
        tausum = torch.sum(z_check.float() * z_sorted, 1)
        tau_z = (tausum - 1 + self.lam) / k_z
        if torch.cuda.is_available():
            dtype = torch.cuda.FloatTensor
        prob = z.sub(tau_z.view(bs, 1).repeat(1, dim)).clamp(min=0).type(dtype)
        if self.normalized:
               prob /= (1-self.lam)
        return prob

class MySumNorm(torch.nn.Module):
    # Translation from minimum value to avoid negative probabilities.
    def __init__(self, translate=True):
        super(MySumNorm, self).__init__()
        self.constant = 1e-30
        self.translate = translate

    def forward(self, input):
        bs = input.data.size()[0]
        dim = input.data.size()[1]
        if self.translate:
            z = input - torch.min(input, dim=1)[0].view(bs, 1).repeat(1, dim) + self.constant
        else:
            z = input
        z_sum = torch.sum(z, dim=1).view(bs, 1).repeat(1, dim)

        return z/z_sum

class MySphericalSoftmax(torch.nn.Module):
    def __init__(self):
        super(MySphericalSoftmax, self).__init__()

    def forward(self, input):
        bs = input.data.size()[0]
        dim = input.data.size()[1]
        z = torch.pow(input, 2)
        z_sum = torch.sum(z, dim=1).view(bs, 1).repeat(1, dim)

        return z/z_sum

class Sparsecone(torch.nn.Module):
    def __init__(self, q=0.0):
        super(Sparsecone, self).__init__()
        self.q = q

    def forward(self, input):
        bs = input.data.size()[0]
        dim = input.data.size()[1]
        # z = input.sub(torch.mean(input,dim=1).repeat(1,dim))
        dtype = torch.FloatTensor
        z = input.type(dtype)
        alpha = (1+dim*self.q)/(torch.sum(z, dim=1) + dim*self.q).view(bs, 1).repeat(1, dim)
        z = alpha * z

        if torch.cuda.is_available():
            dtype = torch.cuda.FloatTensor

        #sort z
        z_sorted = torch.sort(z, descending=True)[0]

        #calculate k(z)
        z_cumsum = torch.cumsum(z_sorted, dim=1)
        k = Variable(torch.arange(1, dim + 1).unsqueeze(0).repeat(bs,1))
        z_check = torch.gt(1 + k * z_sorted, z_cumsum)

        # because the z_check vector is always [1,1,...1,0,0,...0] finding the
        # (index + 1) of the last `1` is the same as just summing the number of 1.
        k_z = torch.sum(z_check.float(), 1)

        #calculate tau(z)
        tausum = torch.sum(z_check.float() * z_sorted, 1)
        tau_z = (tausum - 1) / k_z
        prob = z.sub(tau_z.view(bs, 1).repeat(1, dim)).clamp(min=0).type(dtype)
        return prob

class SparseHourglass(torch.nn.Module):
    def __init__(self, q=0.0, lam=0.0, normalized = True):
        super(SparseHourglass, self).__init__()
        self.q = q
        self.lam = lam
        self.normalized = normalized

    def forward(self, input):
        bs = input.data.size()[0]
        dim = input.data.size()[1]
        # z = input.sub(torch.mean(input,dim=1).repeat(1,dim))
        dtype = torch.FloatTensor
        z = input.type(dtype)
        alpha = (1+dim*self.q)/(torch.abs(torch.sum(z, dim=1)) + dim*self.q).view(bs, 1).repeat(1, dim)
        z = alpha * z

        if torch.cuda.is_available():
            dtype = torch.cuda.FloatTensor

        #sort z
        z_sorted = torch.sort(z, descending=True)[0]

        #calculate k(z)
        z_cumsum = torch.cumsum(z_sorted, dim=1)
        k = Variable(torch.arange(1, dim + 1).unsqueeze(0).repeat(bs,1))
        z_check = torch.gt(1 - self.lam + k * z_sorted, z_cumsum)

        # because the z_check vector is always [1,1,...1,0,0,...0] finding the
        # (index + 1) of the last `1` is the same as just summing the number of 1.
        k_z = torch.sum(z_check.float(), 1)

        #calculate tau(z)
        tausum = torch.sum(z_check.float() * z_sorted, 1)
        tau_z = (tausum - 1 + self.lam) / k_z
        prob = z.sub(tau_z.view(bs, 1).repeat(1, dim)).clamp(min=0).type(dtype)
        if self.normalized:
               prob /= (1-self.lam)
        return prob

