import torch
import torchvision
from functools import partial
import vision_transformer as vit
from torchvision import transforms
import torch.nn as nn

import utils
import widen_resnet

class TWIST(nn.Module):
    def __init__(self, args):
        super(TWIST, self).__init__()
        if args.backbone.startswith('resnet'):
            if(args.dataset == "imagenet" or args.dataset == "imagenet_lmdb"):
                widen_resnet.__dict__['resnet50'] = torchvision.models.resnet50
                conv1_size=7
            elif(args.dataset == "cifar10"):
                widen_resnet.__dict__['resnet18'] = torchvision.models.resnet18
                conv1_size=5
            
            self.backbone = widen_resnet.__dict__[args.backbone]()
            self.feature_dim = self.backbone.fc.weight.shape[1]
        else: # vision transformer based models
            if args.backbone.startswith('vit'):
                self.backbone = vit.__dict__[args.backbone](
                        patch_size=args.patch_size, 
                        norm_layer= (partial(nn.LayerNorm, eps=1e-6)), 
                        drop_path_rate=args.drop_path,
                        freeze_embedding=args.freeze_embedding,
                        conv1_size = conv1_size
                )
                self.feature_dim = self.backbone.embed_dim
        self.backbone.fc = nn.Identity()
        self.backbone = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.backbone)
        if args.crops_interact_style != 'class':
            self.projection_heads = ProjectionHead(args, feature_dim=self.feature_dim)
        else:
            self.projection_heads = utils.ClassHead(args, feature_dim=2048)

    def forward(self, x):
        """
            Codes about multi-crop is borrowed from the codes of Dino
            https://github.com/facebookresearch/dino
        """
        if not isinstance(x, list):
            x = [x]
        # the first indices of aug with changing resolution
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)

        start_idx = 0
        for end_idx in idx_crops:
            _out = self.backbone(torch.cat(x[start_idx: end_idx]))
            if start_idx == 0:
                output = _out
            else:
                output = torch.cat((output, _out))
            start_idx = end_idx

        out = self.projection_heads(output)
        return out

    def backbone_weights(self):
        return self.backbone.state_dict()

class ProjectionHead(nn.Module):
    def __init__(self, args, feature_dim=2048):
        super(ProjectionHead, self).__init__()
        if args.lbn_type == 'bn':
            assert args.bunch_size % args.batch_size == 0
            ranks = list(range(utils.get_world_size()))
            print('---ALL RANKS----\n{}'.format(ranks))
            procs_per_bunch = args.bunch_size // args.batch_size
            assert utils.get_world_size() % procs_per_bunch == 0
            n_bunch = utils.get_world_size() // procs_per_bunch
            rank_groups = [ranks[i*procs_per_bunch: (i+1)*procs_per_bunch] for i in range(n_bunch)]
            print('---RANK GROUPS----\n{}'.format(rank_groups))
            process_groups = [torch.distributed.new_group(pids) for pids in rank_groups]
            bunch_id = utils.get_rank() // procs_per_bunch
            process_group = process_groups[bunch_id]
            print('---CURRENT GROUP----\n{}'.format(process_group))
            norm = nn.SyncBatchNorm(args.dim, affine=0, process_group=process_group)
        elif args.lbn_type == 'syncbn':
            norm = nn.SyncBatchNorm(args.dim, affine=0)
        elif args.lbn_type == 'identity':
            norm = nn.Identity()
        else:
            raise NotImplementedError

        if args.proj_norm == 'bn':
            batchnorm = nn.SyncBatchNorm
        elif args.proj_norm == 'ln':
            batchnorm = partial(nn.LayerNorm, eps=1e-6)
        elif args.proj_norm == 'none':
            batchnorm = nn.Identity
        else:
            raise NotImplementedError

        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, args.hid_dim, bias=True),
            batchnorm(args.hid_dim),
            nn.ReLU() if args.act == 'relu' else nn.GELU(),
            nn.Dropout(p=args.drop),

            nn.Linear(args.hid_dim, args.hid_dim, bias=True),
            batchnorm(args.hid_dim),
            nn.ReLU() if args.act == 'relu' else nn.GELU(),
        )

        last_linear = nn.Linear(args.hid_dim, args.dim, bias=True)
        self.last_linear = last_linear
        self.norm = norm

        if args.backbone.startswith('vit') or args.proj_trunc_init:
            print('using vit initialization')
            self.apply(self._vit_init_weights)

    def _vit_init_weights(self, m):
        if isinstance(m, nn.Linear):
            utils.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def reg_gnf(self, grad):
        self.gn_f = grad.abs().mean().item()

    def reg_gnft(self, grad):
        self.gn_ft = grad.abs().mean().item()

    def forward(self, x):
        x = self.projection_head(x)
        f = self.last_linear(x)
        ft = self.norm(f)
        if self.train and x.requires_grad:
            f.register_hook(self.reg_gnf)
            ft.register_hook(self.reg_gnft)
        self.f_column_std = f.std(dim=0, unbiased=False).mean()
        self.f_row_std    = f.std(dim=1, unbiased=False).mean()
        self.ft_column_std = ft.std(dim=0, unbiased=False).mean()
        self.ft_row_std    = ft.std(dim=1, unbiased=False).mean()

        return ft
