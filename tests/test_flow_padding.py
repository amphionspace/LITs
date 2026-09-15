"""Flow loss must ignore padding while preserving valid-frame gradients."""
import pytest
import torch
from torch import nn
from lits.models.components.flow_matching import CFM, CFM_Causal


class MaskedEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=nn.Parameter(torch.tensor(0.3))
        self.last_prediction=None

    def forward(self,y,mask,mu,t,spks=None,cond=None,streaming=False):
        self.last_prediction=(self.weight*y+0.17*mu)*mask
        return self.last_prediction


class IdentityEncoder(nn.Module):
    def forward(self,mu,mask,streaming=False):
        return mu*mask


def noise_like(x):
    # Appending padding leaves the noise on existing valid frames unchanged.
    frames=torch.arange(x.shape[-1],dtype=x.dtype,device=x.device)
    return (frames.sin()[None,None,:]+0.2).expand_as(x).clone()


def make_model(cls):
    model=cls.__new__(cls)
    nn.Module.__init__(model)
    model.sigma_min=1e-4
    model.streaming=True
    model.estimator=MaskedEstimator()
    if cls is CFM_Causal:model.encoder=IdentityEncoder()
    return model


def inputs(length=7,full=False):
    lengths=torch.tensor([length,length] if full else [3,5])
    mask=(torch.arange(length)[None,:]<lengths[:,None]).unsqueeze(1).float()
    x=(torch.arange(length).float()[None,None,:]*0.1).expand(2,3,-1)*mask
    return x,mask,x*0.6


@pytest.mark.parametrize('cls',[CFM,CFM_Causal])
@pytest.mark.parametrize('full',[False,True])
def test_valid_frame_loss_and_gradients(cls,full,monkeypatch):
    monkeypatch.setattr(torch,'randn_like',noise_like)
    monkeypatch.setattr(torch,'rand',lambda size,**kwargs:torch.full(size,0.25,**kwargs))
    model=make_model(cls)
    x,mask,mu=inputs(full=full)
    loss,_=model.compute_loss(x,mask,mu)
    target=x-(1-model.sigma_min)*noise_like(x)
    errors=(model.estimator.last_prediction-target).square()
    expanded=mask.bool().expand_as(errors)
    expected=errors[expanded].mean()
    legacy=errors.sum()/expanded.sum()
    torch.testing.assert_close(loss,expected)
    new_grad=torch.autograd.grad(loss,model.estimator.weight,retain_graph=True)[0]
    old_grad=torch.autograd.grad(legacy,model.estimator.weight)[0]
    torch.testing.assert_close(new_grad,old_grad)
    if full:torch.testing.assert_close(loss,legacy)
    else:assert legacy>loss


@pytest.mark.parametrize('cls',[CFM,CFM_Causal])
def test_appending_padding_preserves_loss_and_gradient(cls,monkeypatch):
    monkeypatch.setattr(torch,'randn_like',noise_like)
    monkeypatch.setattr(torch,'rand',lambda size,**kwargs:torch.full(size,0.25,**kwargs))
    results=[]
    for length in [5,19]:
        model=make_model(cls)
        x,mask,mu=inputs(length)
        loss,_=model.compute_loss(x,mask,mu)
        grad=torch.autograd.grad(loss,model.estimator.weight)[0]
        results.append((loss.detach(),grad))
    torch.testing.assert_close(results[0][0],results[1][0])
    torch.testing.assert_close(results[0][1],results[1][1])
