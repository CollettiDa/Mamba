# import mamba_ssm
import networks.mamba_sys as mamba_sys
import torch


if __name__ == "__main__":
    # check_vssm_equals_vmambadp()
    loss = torch.nn.MSELoss()
    model = mamba_sys.VSSM().to('cuda')
    int = torch.randn(16,1,224,224).cuda()
    # int = torch.randn(16,1,21,21).cuda()
    out = model(int)
    l = loss(out, torch.randn_like(out))
    l.backward()
    print(out.shape)
    print("done")
