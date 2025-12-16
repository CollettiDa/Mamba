# import mamba_ssm
import mamba_sys
import torch


if __name__ == "__main__":
    # check_vssm_equals_vmambadp()
    model = mamba_sys.VSSM().to('cuda')
    int = torch.randn(16,1,224,224).cuda()
    # int = torch.randn(16,1,21,21).cuda()
    out = model(int)
    print(out.shape)
