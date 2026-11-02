import torch as t
from torch import nn
import torch.nn.functional as F
from Params import args
from Utils.Utils import contrastLoss, ce, l2_norm

init = nn.init.xavier_uniform_

# Locked global meta-path branch (main experiment).
METAPATH_LAYER = 4
METAPATH_LOCAL_RESIDUAL_SCALE = 0.20


def split_bipartite_embeds(h_bipartite):
    """
    Full-graph embeddings are laid out as [drug | gene].
    The gene block starts at n_drug = int(args.drug).
    """
    n_drug = int(args.drug)
    return h_bipartite[:n_drug], h_bipartite[n_drug:]


def merge_bipartite_embeds(h_drug, h_gene):
    return t.cat([h_drug, h_gene], dim=0)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.dEmbeds = nn.Parameter(init(t.empty(args.drug, args.latdim)))
        self.gEmbeds = nn.Parameter(init(t.empty(args.gene, args.latdim)))
        n_i = int(getattr(args, 'disease', 0))
        if n_i <= 0:
            raise ValueError('meta-path branch requires args.disease > 0 (set in LoadData)')
        self.iEmbeds = nn.Parameter(init(t.empty(n_i, args.latdim)))

        self.gcnLayer = GCNLayer()
        self.metapath_drug = MetaPathEncoder('drug', args.metapath_drug_path_order)
        self.metapath_gene = MetaPathEncoder('gene', args.metapath_gene_path_order)
        self.classifierLayer = ClassifierLayer()
        self.edgeDropper = SpAdjDropEdge()

    def _local_branch_delta(self, adj, keepRate, h):
        adj_r = self.edgeDropper(adj, keepRate)
        return self.gcnLayer(adj_r, h)

    def forward(self, adj, keepRate, branch2=None):
        h0 = t.concat([self.dEmbeds, self.gEmbeds], axis=0)
        embeds_hist = [h0]
        loc_hist = [h0]
        glb_hist = [h0]
        bundle = branch2['bundle']

        Lg = int(args.gnn_layer)
        Ls = int(METAPATH_LAYER)
        T = max(Lg, Ls)
        ssl_mask = []
        for step in range(T):
            h = embeds_hist[-1]
            if step < Lg:
                embeds_loc = self._local_branch_delta(adj, keepRate, h)
            else:
                embeds_loc = h
            if step < Ls:
                h_drug, h_gene = split_bipartite_embeds(h)
                hyper_drug = self.metapath_drug(bundle, h_drug, h_gene, self.iEmbeds)
                hyper_gene = self.metapath_gene(bundle, h_drug, h_gene, self.iEmbeds)
                embeds_glb = merge_bipartite_embeds(hyper_drug, hyper_gene)
            else:
                embeds_glb = t.zeros_like(h)
            loc_hist.append(embeds_loc)
            glb_hist.append(embeds_glb)
            embeds_hist.append(embeds_loc + embeds_glb)
            ssl_mask.append(step < Lg and step < Ls)

        embeds = sum(embeds_hist)
        return embeds, loc_hist, glb_hist, ssl_mask

    def calcLosses(self, drugs, genes, labels, adj, keepRate, branch2=None):
        embeds, loc_hist, glb_hist, ssl_mask = self.forward(adj, keepRate, branch2)
        dEmbeds, gEmbeds = split_bipartite_embeds(embeds)
        dEmbeds = dEmbeds[drugs]
        gEmbeds = gEmbeds[genes]

        pre = self.classifierLayer(dEmbeds, gEmbeds)
        ceLoss = ce(pre, labels, args.label_smoothing)

        sslLoss = t.zeros((), device=ceLoss.device, dtype=ceLoss.dtype)
        for i in range(1, len(ssl_mask) + 1):
            if not ssl_mask[i - 1]:
                continue
            embeds_loc = loc_hist[i].detach()
            embeds_glb = glb_hist[i]
            loc_drug, _ = split_bipartite_embeds(embeds_loc)
            glb_drug, _ = split_bipartite_embeds(embeds_glb)
            sslLoss = sslLoss + contrastLoss(
                loc_drug, glb_drug, t.unique(drugs), args.temp)
            _, loc_gene = split_bipartite_embeds(embeds_loc)
            _, glb_gene = split_bipartite_embeds(embeds_glb)
            sslLoss = sslLoss + contrastLoss(
                loc_gene, glb_gene, t.unique(genes), args.temp)

        return ceLoss, sslLoss

    def predict(self, adj, drugs, genes, branch2=None):
        embeds, _, _, _ = self.forward(adj, 1.0, branch2)
        dEmbeds, gEmbeds = split_bipartite_embeds(embeds)
        dEmbeds = dEmbeds[drugs]
        gEmbeds = gEmbeds[genes]
        return self.classifierLayer(dEmbeds, gEmbeds)


def _metapath_src_by_shape(adj, side, h_drug, h_gene, h_disease=None):
    n_d, n_g = int(args.drug), int(args.gene)
    n_i = int(getattr(args, 'disease', 0))
    sh = tuple(adj.shape)
    if side == 'drug' and sh[0] != n_d:
        raise ValueError('MetaPath drug-side adjacency expects {} rows, got {}'.format(n_d, sh))
    if side == 'gene' and sh[0] != n_g:
        raise ValueError('MetaPath gene-side adjacency expects {} rows, got {}'.format(n_g, sh))
    if sh[1] == n_d:
        return h_drug
    if sh[1] == n_g:
        return h_gene
    if n_i > 0 and sh[1] == n_i:
        if h_disease is None:
            raise ValueError('MetaPath disease-column adj requires h_disease')
        return h_disease
    raise ValueError('MetaPath {}-side adjacency bad shape {}'.format(side, sh))


def metapath_route_spmm(adj, side, h_drug, h_gene, h_disease=None):
    src = _metapath_src_by_shape(adj, side, h_drug, h_gene, h_disease)
    return t.spmm(adj, src)


class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()

    def forward(self, adj, embeds):
        return l2_norm(t.spmm(adj, embeds))


class MetaPathAttnFusion(nn.Module):
    """Semantic attention over the meta-path dimension."""

    def __init__(self, num_paths, dim):
        super(MetaPathAttnFusion, self).__init__()
        att_dim = int(args.metapath_att_dim)
        self.sem_a = nn.Linear(dim, att_dim, bias=False)
        self.sem_b = nn.Linear(att_dim, 1, bias=False)
        init(self.sem_a.weight)
        init(self.sem_b.weight)

    def forward(self, Z):
        if Z.size(1) == 1:
            return Z.squeeze(1)
        e = self.sem_b(t.tanh(self.sem_a(Z))).squeeze(-1)
        a = F.softmax(e, dim=-1)
        return (a.unsqueeze(-1) * Z).sum(dim=1)


class MetaPathEncoder(nn.Module):
    """
    One side of the global meta-path branch.
    DIG/GID use the precomputed structural aggregate; DGI/GDI aggregate disease embeddings.
    """

    def __init__(self, side, path_order):
        super(MetaPathEncoder, self).__init__()
        assert side in ('drug', 'gene')
        self.side = side
        self.path_order = list(path_order)
        n_p = len(self.path_order)
        self.merge_lin = nn.ModuleList([
            nn.Linear(args.latdim, args.latdim, bias=False) for _ in range(n_p)])
        for m in self.merge_lin:
            init(m.weight)
        self.path_lin = nn.ModuleList([
            nn.Linear(args.latdim, args.latdim, bias=False) for _ in range(n_p)])
        for m in self.path_lin:
            init(m.weight)
        self.fusion = MetaPathAttnFusion(n_p, args.latdim)

    def _pick_local_adj(self, mats):
        """One-hop bipartite path used by the fixed 0.20 residual (drug: DIG, gene: GID)."""
        n_d, n_g = int(args.drug), int(args.gene)
        pref = 'DG' if self.side == 'drug' else 'GD'
        if pref in mats:
            return mats[pref]
        for _, adj in mats.items():
            sh = tuple(adj.shape)
            if self.side == 'drug' and sh == (n_d, n_g):
                return adj
            if self.side == 'gene' and sh == (n_g, n_d):
                return adj
        return None

    def forward(self, bundle, h_drug, h_gene, h_disease=None):
        if self.side == 'drug':
            h_self, h_other = h_drug, h_gene
            mats = bundle['mats_drug']
        else:
            h_self, h_other = h_gene, h_drug
            mats = bundle['mats_gene']

        pre_map = bundle.get('pre_drug') if self.side == 'drug' else bundle.get('pre_gene')
        use_pre = bundle.get('agg_mode') == 'precompute' and pre_map is not None

        zs = []
        for i, name in enumerate(self.path_order):
            adj = mats[name]
            if use_pre and name in pre_map:
                z = l2_norm(pre_map[name] + self.merge_lin[i](h_self))
            else:
                z = l2_norm(metapath_route_spmm(
                    adj, self.side, h_drug, h_gene, h_disease))
            z = F.elu(self.path_lin[i](z))
            zs.append(z)
        Z = t.stack(zs, dim=1)
        fused = self.fusion(Z)
        local_adj = self._pick_local_adj(mats)
        if local_adj is not None:
            local_msg = l2_norm(t.spmm(local_adj, h_other))
            gamma = t.tensor(
                float(METAPATH_LOCAL_RESIDUAL_SCALE),
                device=fused.device,
                dtype=fused.dtype,
            )
            fused = fused + gamma * local_msg
        fused = fused + h_self
        return l2_norm(fused)


class SpAdjDropEdge(nn.Module):
    def __init__(self):
        super(SpAdjDropEdge, self).__init__()

    def forward(self, adj, keepRate):
        if keepRate == 1.0:
            return adj
        vals = adj._values()
        idxs = adj._indices()
        edgeNum = vals.size()
        mask = ((t.rand(edgeNum) + keepRate).floor()).type(t.bool)
        newVals = vals[mask] / keepRate
        newIdxs = idxs[:, mask]
        return t.sparse.FloatTensor(newIdxs, newVals, adj.shape)


class ClassifierLayer(nn.Module):
    def __init__(self):
        super(ClassifierLayer, self).__init__()
        self.lin1 = nn.Linear(args.latdim * 2, 128)
        self.lin2 = nn.Linear(128, args.num_classes)

    def forward(self, dEmbeds, gEmbeds):
        embeds = t.cat((dEmbeds, gEmbeds), dim=1)
        embeds = F.relu(self.lin1(embeds))
        embeds = F.dropout(embeds, p=0.4, training=self.training)
        return self.lin2(embeds)
