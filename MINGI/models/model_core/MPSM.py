from utils import *

l2_loss = nn.MSELoss()

class MPSM(nn.Module):
    def __init__(self):
        super(MPSM, self).__init__()

        self.unfold = torch.nn.Unfold(kernel_size=args.patch_size, stride=args.patch_size)
        self.cos = nn.CosineSimilarity(dim=2, eps=1e-6)

    def forward(self, pred, au, tp):
        patch_pred = self.unfold(pred) # [b, 75, 2601] 2601개의 패치
        patch_label = self.unfold(au + tp) # [b, 75, 2601]

        cos_pred = cos_pairwise(patch_pred)
        cos_label = cos_pairwise(patch_label)

        loss = l2_loss(cos_label, cos_pred)

        return loss


def cos_pairwise(tensor):
    x = tensor.permute((2, 1, 0)) #  (batch_size, vector_dimension, num_vectors)
    cos_sim_pairwise = F.cosine_similarity(x, x.unsqueeze(1), dim=-2)
    cos_sim_pairwise = cos_sim_pairwise.permute((2, 1, 0)) # [b, 2601, 2601]
    return cos_sim_pairwise


if __name__ == '__main__':
    model = MPSM().cuda()
    inp = torch.rand((2, 3, 256, 256)).cuda() # 내말이
    lab = torch.rand((2, 3, 256, 256)).cuda() # 내말이

    out = model(inp, inp, lab, lab)
    print(out)
    print('out', out.shape)

