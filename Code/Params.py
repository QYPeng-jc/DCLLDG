import argparse


def ParseArgs():
    
    parser = argparse.ArgumentParser(description='DCLLDG params')
    parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')
    parser.add_argument('--batch', default=4096, type=int, help='batch size')
    parser.add_argument('--tstBat', default=100000, type=int, help='test batch size')
    parser.add_argument('--reg', default=1e-7, type=float, help='L2 regularizer on weights')
    parser.add_argument('--epoch', default=500, type=int, help='max training epochs')
    parser.add_argument('--latdim', default=256, type=int, help='embedding size')
    parser.add_argument('--gnn_layer', default=4, type=int, help='local GCN depth')
    parser.add_argument('--keepRate', default=0.75, type=float, help='edge keep rate for local GCN dropout')
    parser.add_argument('--temp', default=0.1, type=float, help='InfoNCE temperature')
    parser.add_argument('--ssl_reg', default=0.05, type=float, help='contrastive loss weight')
    parser.add_argument('--label_smoothing', default=0.08, type=float,
                        help='cross-entropy label smoothing in [0, 1)')
    parser.add_argument('--tstEpoch', default=1, type=int, help='evaluate every N epochs')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id; <0 uses CPU')
    parser.add_argument('--seed', default=43, type=int, help='base seed; run i uses seed+i')
    parser.add_argument('--iteration', type=int, default=5, help='number of runs')
    parser.add_argument('--is_debug', type=bool, default=False)
    parser.add_argument('--validate', action='store_true', default=False,
                        help='if set, evaluate on the validation loader instead of test')
    parser.add_argument('--load_model', default=None, help='optional checkpoint name under ../Models')
    parser.add_argument('--metapath_att_dim', type=int, default=128,
                        help='meta-path semantic-attention hidden size')
    return parser.parse_args()


args = ParseArgs()
