import numpy as np
from scipy.sparse import coo_matrix
from Params import args
import scipy.sparse as sp
import pandas as pd
import os
import torch as t
import torch.utils.data as data
import torch.utils.data as dataloader


class DataHandler:
    def __init__(self):
        self.data = 'CTD'

    def _ctd_file_candidates(self, kind):
        names = {
            'dgi': 'CTD_DGI.csv',
            'ddi': 'CTD_RDAs.csv',
            'gdi': 'CTD_GDAs.csv',
        }
        if kind not in names:
            raise ValueError('unknown CTD file kind: {}'.format(kind))
        return [names[kind]]

    def _ctd_resolve_file(self, base_dir, candidates):
        for nm in candidates:
            p = os.path.join(base_dir, nm)
            if os.path.exists(p):
                return p
        raise FileNotFoundError('CTD file not found in {}. Tried: {}'.format(base_dir, candidates))

    def _ctd_read_pairs(self, file_path):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            first_line = f.readline()
        has_header = any(k in first_line for k in ('ChemicalID', 'GeneID', 'DiseaseID'))
        df = pd.read_csv(
            file_path, sep=None, engine='python', header=0 if has_header else None
        )
        if df.shape[1] < 2:
            raise ValueError('CTD file {} must have at least 2 columns'.format(file_path))
        def _norm_id(x):
            if pd.isna(x):
                return ''
            if isinstance(x, (np.integer, int)):
                return str(int(x))
            if isinstance(x, (np.floating, float)):
                v = float(x)
                if np.isfinite(v) and abs(v - round(v)) < 1e-9:
                    return str(int(round(v)))
                return str(x).strip()
            s = str(x).strip()
            try:
                v = float(s)
                if np.isfinite(v) and abs(v - round(v)) < 1e-9:
                    return str(int(round(v)))
            except Exception:
                pass
            return s
        a = df.iloc[:, 0].map(_norm_id).to_numpy()
        b = df.iloc[:, 1].map(_norm_id).to_numpy()
        return a, b

    def _ctd_gcn_include_train_neg_enabled(self):
        return True

    def _ctd_build_gcn_train_rel_matrix(self, n_d, n_g, trn_pos, trn_neg=None):
        """
        Typed train relation matrix.
        pos -> 2.0, train neg -> 1.0; LoadData binarizes values >= 1.
        """
        rel_trn = np.zeros((int(n_d), int(n_g)), dtype=np.float32)
        include_neg = self._ctd_gcn_include_train_neg_enabled()
        if include_neg and trn_neg is not None and len(trn_neg) > 0:
            rel_trn[trn_neg[:, 0], trn_neg[:, 1]] = 1.0
        if trn_pos is not None and len(trn_pos) > 0:
            rel_trn[trn_pos[:, 0], trn_pos[:, 1]] = 2.0
        return rel_trn

    def _ctd_graph_base_dir(self):
        return os.path.join('..', 'Data')

    def _ctd_split_dir(self, mode='transductive'):
        split_dir = os.path.join('..', 'Data', 'transductive')
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                'Fixed split directory is missing: {}'.format(split_dir))
        return split_dir

    def _ctd_load_graph_context(self, base_dir):
        dgi_fp = self._ctd_resolve_file(base_dir, self._ctd_file_candidates('dgi'))
        ddi_fp = self._ctd_resolve_file(base_dir, self._ctd_file_candidates('ddi'))
        gdi_fp = self._ctd_resolve_file(base_dir, self._ctd_file_candidates('gdi'))

        dgi_d_raw, dgi_g_raw = self._ctd_read_pairs(dgi_fp)
        ddi_d_raw, ddi_i_raw = self._ctd_read_pairs(ddi_fp)
        gdi_g_raw, gdi_i_raw = self._ctd_read_pairs(gdi_fp)

        # curated_v2 node universe is the DGI drug/gene set; DDI/GDI stay as disease bridges.
        drug_ids = sorted(set(dgi_d_raw.tolist()))
        gene_ids = sorted(set(dgi_g_raw.tolist()))
        dis_ids = sorted(set(ddi_i_raw.tolist()) | set(gdi_i_raw.tolist()))
        d_map = {x: i for i, x in enumerate(drug_ids)}
        g_map = {x: i for i, x in enumerate(gene_ids)}
        i_map = {x: i for i, x in enumerate(dis_ids)}

        dgi_pairs = np.asarray(
            [(d_map[d], g_map[g]) for d, g in zip(dgi_d_raw.tolist(), dgi_g_raw.tolist()) if d in d_map and g in g_map],
            dtype=np.int64,
        )
        if dgi_pairs.size == 0:
            raise ValueError('CTD_DGI has no samples')
        dgi_pairs = np.unique(dgi_pairs, axis=0)

        ddi_pairs = np.asarray(
            [(d_map[d], i_map[i]) for d, i in zip(ddi_d_raw.tolist(), ddi_i_raw.tolist()) if d in d_map and i in i_map],
            dtype=np.int64,
        )
        gdi_pairs = np.asarray(
            [(g_map[g], i_map[i]) for g, i in zip(gdi_g_raw.tolist(), gdi_i_raw.tolist()) if g in g_map and i in i_map],
            dtype=np.int64,
        )
        ddi_pairs = np.unique(ddi_pairs, axis=0) if ddi_pairs.size > 0 else np.zeros((0, 2), dtype=np.int64)
        gdi_pairs = np.unique(gdi_pairs, axis=0) if gdi_pairs.size > 0 else np.zeros((0, 2), dtype=np.int64)

        return {
            'drug_ids': drug_ids,
            'gene_ids': gene_ids,
            'd_map': d_map,
            'g_map': g_map,
            'dgi_pairs': dgi_pairs,
            'ddi_pairs': ddi_pairs,
            'gdi_pairs': gdi_pairs,
            'dis_ids': dis_ids,
        }

    def _ctd_pack_return(self, ctx, trn_pos, tst_pos, trn_pairs, trn_labels, tst_pairs, tst_labels):
        n_d = len(ctx['drug_ids'])
        n_g = len(ctx['gene_ids'])
        trn_neg = trn_pairs[trn_labels == 0] if len(trn_pairs) > 0 else np.zeros((0, 2), dtype=np.int64)
        rel_trn = self._ctd_build_gcn_train_rel_matrix(n_d, n_g, trn_pos, trn_neg)
        rel_tst = np.zeros((n_d, n_g), dtype=np.float32)
        rel_tst[tst_pos[:, 0], tst_pos[:, 1]] = 2.0
        if self._ctd_gcn_include_train_neg_enabled():
            tst_neg_pairs = tst_pairs[tst_labels == 0] if len(tst_pairs) > 0 else np.zeros((0, 2), dtype=np.int64)
            if len(tst_neg_pairs) > 0:
                rel_tst[tst_neg_pairs[:, 0], tst_neg_pairs[:, 1]] = 1.0
        n_trn_neg_edges = int(np.sum(rel_trn == 1.0))
        print('[CTD] GCN train graph edges: pos={} neg={} include_train_neg={}'.format(
            len(trn_pos), n_trn_neg_edges, int(self._ctd_gcn_include_train_neg_enabled())))
        relation_mx_train = sp.csr_matrix(rel_trn)
        relation_mx_test = sp.csr_matrix(rel_tst)

        self.ctd_aux = {
            'num_disease': len(ctx['dis_ids']),
            'ddi_pairs': ctx['ddi_pairs'],
            'gdi_pairs': ctx['gdi_pairs'],
        }

        d_train_idx = trn_pairs[:, 0].astype(np.int64)
        g_train_idx = trn_pairs[:, 1].astype(np.int64)
        d_test_idx = tst_pairs[:, 0].astype(np.int64)
        g_test_idx = tst_pairs[:, 1].astype(np.int64)
        val_labels = np.zeros((0,), dtype=np.int64)
        d_val_idx = np.zeros((0,), dtype=np.int64)
        g_val_idx = np.zeros((0,), dtype=np.int64)
        class_values = np.array([0.0, 1.0], dtype=np.float32)
        return relation_mx_train, relation_mx_test, trn_labels, d_train_idx, g_train_idx, \
            val_labels, d_val_idx, g_val_idx, tst_labels, d_test_idx, g_test_idx, class_values

    def _ctd_norm_gene_value(self, v):
        if pd.isna(v):
            return None
        try:
            if isinstance(v, (int, float, np.integer, np.floating)):
                return str(int(round(float(v))))
        except Exception:
            pass
        s = str(v).strip()
        if s.endswith('.0'):
            s = s[:-2]
        try:
            return str(int(float(s)))
        except Exception:
            return s

    def _ctd_read_labeled_split_file(self, file_path, d_map, g_map, strict=False):
        df = pd.read_csv(file_path, header=None, names=['d', 'g', 'y'])
        pairs = []
        labels = []
        skipped = 0
        for d_raw, g_raw, y in zip(df['d'], df['g'], df['y']):
            d = str(d_raw).strip()
            g = self._ctd_norm_gene_value(g_raw)
            if g is None or d not in d_map or g not in g_map:
                skipped += 1
                continue
            pairs.append((d_map[d], g_map[g]))
            labels.append(int(y))
        total = len(df)
        if len(pairs) == 0:
            raise ValueError('No valid rows in {}'.format(file_path))
        if strict and skipped > 0:
            raise ValueError(
                'CTD split {}: {}/{} rows could not be mapped to the current graph '
                '(check --ctd_data_variant matches export, and binary splits need DGI+GDI node universe). '
                'Skipped rows often indicate DDG negative genes missing from gene_ids.'.format(
                    file_path, skipped, total))
        if skipped > 0:
            print('[CTD] WARNING: skipped {}/{} rows in {}'.format(skipped, total, file_path))
        return np.asarray(pairs, dtype=np.int64), np.asarray(labels, dtype=np.int64)

    def _ctd_validate_fixed_binary_split(self, ctx, train_fp, test_fp, trn_labels, tst_labels):
        """Ensure fixed binary split aligns with DGI pairs and loads all pos/neg rows."""
        tr_df = pd.read_csv(train_fp, header=None, names=['d', 'g', 'y'])
        te_df = pd.read_csv(test_fp, header=None, names=['d', 'g', 'y'])
        file_trn_pos = int(np.sum(tr_df['y'].values == 1))
        file_trn_neg = int(np.sum(tr_df['y'].values == 0))
        file_tst_pos = int(np.sum(te_df['y'].values == 1))
        file_tst_neg = int(np.sum(te_df['y'].values == 0))
        trn_pos = int(np.sum(trn_labels == 1))
        trn_neg = int(np.sum(trn_labels == 0))
        tst_pos = int(np.sum(tst_labels == 1))
        tst_neg = int(np.sum(tst_labels == 0))
        if (trn_pos, trn_neg, tst_pos, tst_neg) != (file_trn_pos, file_trn_neg, file_tst_pos, file_tst_neg):
            raise ValueError(
                'CTD fixed split row count mismatch after loading. '
                'file train pos/neg={}/{} test pos/neg={}/{}; '
                'loaded train pos/neg={}/{} test pos/neg={}/{}. '
                'Use matching --ctd_data_variant (e.g. com2_bridge20k) or re-export split.'.format(
                    file_trn_pos, file_trn_neg, file_tst_pos, file_tst_neg,
                    trn_pos, trn_neg, tst_pos, tst_neg))
        n_dgi_pos = len(ctx['dgi_pairs'])
        n_split_pos = file_trn_pos + file_tst_pos
        if n_split_pos != n_dgi_pos:
            raise ValueError(
                'CTD fixed split positive count ({}) != DGI pairs in current variant ({}). '
                'Re-export train/test with the same --ctd_data_variant as training.'.format(
                    n_split_pos, n_dgi_pos))

    def load_ctd_data_fixed(self, mode='transductive'):
        out_dir = self._ctd_split_dir(mode)
        train_fp = os.path.join(out_dir, 'train.csv')
        test_fp = os.path.join(out_dir, 'test.csv')
        if not os.path.isfile(train_fp) or not os.path.isfile(test_fp):
            raise FileNotFoundError(
                'Fixed curated_v2 split requires {} and {}.'.format(train_fp, test_fp))

        base_dir = self._ctd_graph_base_dir()
        ctx = self._ctd_load_graph_context(base_dir)
        trn_pairs, trn_labels = self._ctd_read_labeled_split_file(
            train_fp, ctx['d_map'], ctx['g_map'], strict=True)
        tst_pairs, tst_labels = self._ctd_read_labeled_split_file(
            test_fp, ctx['d_map'], ctx['g_map'], strict=True)
        self._ctd_validate_fixed_binary_split(ctx, train_fp, test_fp, trn_labels, tst_labels)

        trn_pos_mask = trn_labels == 1
        tst_pos_mask = tst_labels == 1
        trn_pos = trn_pairs[trn_pos_mask]
        tst_pos = tst_pairs[tst_pos_mask]
        trn_neg = trn_pairs[~trn_pos_mask]
        print('[CTD fixed] loaded train pos={} neg={} ratio={:.4f}'.format(
            len(trn_pos), len(trn_neg), len(trn_neg) / max(len(trn_pos), 1)))
        print('[CTD fixed] loaded test  pos={} neg={} ratio={:.4f}'.format(
            len(tst_pos), int(np.sum(~tst_pos_mask)), int(np.sum(~tst_pos_mask)) / max(len(tst_pos), 1)))

        return self._ctd_pack_return(
            ctx, trn_pos, tst_pos, trn_pairs, trn_labels, tst_pairs, tst_labels)

    def normalizeAdj(self, mat):
        """
        Normalize an adjacency matrix using the degree normalization technique.
        """
        degree = np.array(mat.sum(axis=-1))
        dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
        dInvSqrt[np.isinf(dInvSqrt)] = 0.0
        dInvSqrtMat = sp.diags(dInvSqrt)
        return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

    def coo_to_torch_sparse(self, mat_coo):
        """COO scipy sparse (symmetric normalized) -> torch sparse cuda, coalesced."""
        mat_coo = mat_coo.tocoo()
        idxs = t.from_numpy(np.vstack([mat_coo.row, mat_coo.col]).astype(np.int64))
        vals = t.from_numpy(mat_coo.data.astype(np.float32))
        st = t.sparse.FloatTensor(idxs, vals, t.Size(mat_coo.shape)).cuda()
        return st.coalesce()

    def bipartite_sym_norm_csr(self, B_csr):
        """D^{-1/2}_{drug} B D^{-1/2}_{gene} for drug×gene B (CSR)."""
        B = B_csr.tocsr()
        row_deg = np.asarray(B.sum(axis=1)).ravel()
        col_deg = np.asarray(B.sum(axis=0)).ravel()
        r_inv_sqrt = np.power(row_deg, -0.5)
        c_inv_sqrt = np.power(col_deg, -0.5)
        r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.0
        c_inv_sqrt[np.isinf(c_inv_sqrt)] = 0.0
        D_r = sp.diags(r_inv_sqrt)
        D_c = sp.diags(c_inv_sqrt)
        return D_r.dot(B).dot(D_c).tocsr()

    def _ctd_build_ddi_gdi_mats(self, n_d, n_g):
        """R_di (drug×disease) and R_gi (gene×disease) from ctd_aux."""
        if not hasattr(self, 'ctd_aux'):
            raise RuntimeError('CTD auxiliary pairs are not prepared')
        aux = self.ctd_aux
        n_i = int(aux.get('num_disease', 0))
        if n_i <= 0:
            raise ValueError('CTD disease node count is zero')
        ddi = aux['ddi_pairs']
        gdi = aux['gdi_pairs']
        R_di = sp.csr_matrix(
            (np.ones(len(ddi), dtype=np.float32), (ddi[:, 0], ddi[:, 1])),
            shape=(int(n_d), n_i),
        ) if len(ddi) > 0 else sp.csr_matrix((int(n_d), n_i), dtype=np.float32)
        R_gi = sp.csr_matrix(
            (np.ones(len(gdi), dtype=np.float32), (gdi[:, 0], gdi[:, 1])),
            shape=(int(n_g), n_i),
        ) if len(gdi) > 0 else sp.csr_matrix((int(n_g), n_i), dtype=np.float32)
        return R_di, R_gi, n_i

    def _ctd_induce_ddi_gdi_from_train_dg(self, B_bin, R_di_full, R_gi_full):
        """
        Keep disease edges of drugs and genes that appear in the training drug-gene graph.
        This is the train-induced drug-gene-disease subgraph (row mask on RDAs/GDAs).
        """
        B = B_bin.tocsr()
        n_d, n_g = B.shape
        n_i = int(R_di_full.shape[1])
        drug_deg = np.asarray(B.sum(axis=1)).ravel()
        gene_deg = np.asarray(B.sum(axis=0)).ravel()
        train_drugs = np.flatnonzero(drug_deg > 0)
        train_genes = np.flatnonzero(gene_deg > 0)

        R_di = R_di_full.tocsr()
        R_gi = R_gi_full.tocsr()
        d_mask = np.zeros(n_d, dtype=np.float32)
        g_mask = np.zeros(n_g, dtype=np.float32)
        if train_drugs.size > 0:
            d_mask[train_drugs] = 1.0
        if train_genes.size > 0:
            g_mask[train_genes] = 1.0
        R_di_ind = (sp.diags(d_mask) @ R_di).tocsr()
        R_gi_ind = (sp.diags(g_mask) @ R_gi).tocsr()

        dis_hit = (
            np.asarray(R_di_ind.sum(axis=0)).ravel()
            + np.asarray(R_gi_ind.sum(axis=0)).ravel()
        ) > 0
        keep_dis = np.flatnonzero(dis_hit)
        if 0 < keep_dis.size < n_i:
            col_keep = np.zeros(n_i, dtype=np.float32)
            col_keep[keep_dis] = 1.0
            D_keep = sp.diags(col_keep)
            R_di_ind = (R_di_ind @ D_keep).tocsr()
            R_gi_ind = (R_gi_ind @ D_keep).tocsr()

        stats = {
            'n_train_drugs': int(train_drugs.size),
            'n_train_genes': int(train_genes.size),
            'n_diseases_induced': int(keep_dis.size),
            'nnz_R_di_full': int(R_di.nnz),
            'nnz_R_gi_full': int(R_gi.nnz),
            'nnz_R_di_ind': int(R_di_ind.nnz),
            'nnz_R_gi_ind': int(R_gi_ind.nnz),
        }
        return R_di_ind, R_gi_ind, stats

    def _ctd_build_tri_path_mats(self, B, R_di, R_gi):
        """DIG / DGI / GID / GDI from the training bipartite graph and disease bridges."""
        B = B.tocsr()
        R_di = R_di.tocsr()
        R_gi = R_gi.tocsr()
        DIG = (R_di @ R_gi.transpose().tocsr()).tocsr()
        GID = (R_gi @ R_di.transpose().tocsr()).tocsr()
        DGI = (B @ R_gi).tocsr()
        GDI = (B.transpose().tocsr() @ R_di).tocsr()
        mats_scipy = {
            'DIG': self.bipartite_sym_norm_csr(DIG).tocoo(),
            'DGI': self.bipartite_sym_norm_csr(DGI).tocoo(),
            'GID': self.bipartite_sym_norm_csr(GID).tocoo(),
            'GDI': self.bipartite_sym_norm_csr(GDI).tocoo(),
        }
        drug_names = ['DIG', 'DGI']
        gene_names = ['GID', 'GDI']
        return mats_scipy, drug_names, gene_names

    def build_metapath_bundle(self, B_bin, n_d, n_g):
        """
        Global meta-path bundle from the training D-G graph.
        Disease bridges used by the paths are the train-induced DDI/GDI subgraph.
        """
        R_di_full, R_gi_full, n_i = self._ctd_build_ddi_gdi_mats(int(n_d), int(n_g))
        B = B_bin.tocsr()
        R_di_ind, R_gi_ind, stats = self._ctd_induce_ddi_gdi_from_train_dg(
            B, R_di_full, R_gi_full)
        mats_scipy, drug_names, gene_names = self._ctd_build_tri_path_mats(
            B, R_di_ind, R_gi_ind)
        mats_drug_torch = {name: self.coo_to_torch_sparse(mats_scipy[name]) for name in drug_names}
        mats_gene_torch = {name: self.coo_to_torch_sparse(mats_scipy[name]) for name in gene_names}
        print(
            '[CTD tri_dgi_train_hetero] induced from train D-G: '
            'train_drugs={} train_genes={} diseases={} | '
            'R_di nnz {}->{} | R_gi nnz {}->{} | paths drug={} gene={}'.format(
                stats['n_train_drugs'], stats['n_train_genes'], stats['n_diseases_induced'],
                stats['nnz_R_di_full'], stats['nnz_R_di_ind'],
                stats['nnz_R_gi_full'], stats['nnz_R_gi_ind'],
                drug_names, gene_names))
        for name in drug_names + gene_names:
            print('[CTD tri_dgi_train_hetero] path {} nnz={}'.format(
                name, int(mats_scipy[name].nnz)))
        print('[CTD tri_dgi_train_hetero] local disease DDI/GDI source=full '
              '(ctd_local_disease_use_induced=0)')
        return {
            'drug_path_order': drug_names,
            'gene_path_order': gene_names,
            'mats_drug': mats_drug_torch,
            'mats_gene': mats_gene_torch,
            'num_disease': int(n_i),
            'disease_neighbor_paths': {'DGI', 'GDI'},
            'path_variant': 'tri_dgi_train_hetero',
            'train_hetero_stats': stats,
            'local_disease_graph_source': 'full',
        }

    def augment_metapath_bundle_precompute(self, bundle):
        """
        One-time meta-path aggregation on fixed structural features.
        Disease-neighbor paths (DGI/GDI) stay on the trainable disease embeddings.
        """
        from Utils.Utils import l2_norm

        nd, ng = args.drug, args.gene
        d_sf = args.latdim
        rng = np.random.RandomState(int(args.seed) + 17)
        Xd = t.from_numpy(rng.randn(nd, d_sf).astype(np.float32)).cuda()
        Xg = t.from_numpy(rng.randn(ng, d_sf).astype(np.float32)).cuda()

        pre_drug = {}
        disease_paths = set(bundle.get('disease_neighbor_paths') or [])
        n_i = int(bundle.get('num_disease', 0))
        if n_i > 0 and disease_paths:
            _ = t.from_numpy(rng.randn(n_i, d_sf).astype(np.float32)).cuda()
        for name in bundle['drug_path_order']:
            if name in disease_paths:
                continue
            adj = bundle['mats_drug'][name]
            sh = tuple(adj.shape)
            if sh[1] == nd:
                raw = t.spmm(adj, Xd)
            elif sh[1] == ng:
                raw = t.spmm(adj, Xg)
            else:
                raise ValueError('precompute: unknown drug-side adj shape {}'.format(sh))
            pre_drug[name] = l2_norm(raw).detach()

        pre_gene = {}
        for name in bundle['gene_path_order']:
            if name in disease_paths:
                continue
            adj = bundle['mats_gene'][name]
            sh = tuple(adj.shape)
            if sh[1] == ng:
                raw = t.spmm(adj, Xg)
            elif sh[1] == nd:
                raw = t.spmm(adj, Xd)
            else:
                raise ValueError('precompute: unknown gene-side adj shape {}'.format(sh))
            pre_gene[name] = l2_norm(raw).detach()

        bundle['agg_mode'] = 'precompute'
        bundle['pre_drug'] = pre_drug
        bundle['pre_gene'] = pre_gene
        return bundle

    def makeTorchAdj(self, mat):
        """
        Convert a SciPy sparse matrix into a PyTorch sparse tensor and apply normalization.
        """
        a = sp.csr_matrix((args.drug, args.drug))
        b = sp.csr_matrix((args.gene, args.gene))
        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat != 0) * 1.0
        mat = (mat + sp.eye(mat.shape[0])) * 1.0
        mat = self.normalizeAdj(mat)

        idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = t.from_numpy(mat.data.astype(np.float32))
        shape = t.Size(mat.shape)
        return t.sparse.FloatTensor(idxs, vals, shape).cuda()

    def LoadData(self):
        relation_mx_train, relation_mx_test, train_labels, d_train_idx, g_train_idx, \
            val_labels, d_val_idx, g_val_idx, test_labels, d_test_idx, g_test_idx, class_values = \
            self.load_ctd_data_fixed()

        trnMat, tstMat = relation_mx_train, relation_mx_test
        trnMat[trnMat >= 1] = 1
        tstMat[tstMat >= 1] = 1

        if type(trnMat) != coo_matrix:
            trnMat = sp.coo_matrix(trnMat)
        if type(tstMat) != coo_matrix:
            tstMat = sp.coo_matrix(tstMat)
        args.drug, args.gene = trnMat.shape
        args.num_classes = len(class_values)
        self.torchBiAdj = self.makeTorchAdj(trnMat)

        B_bin = sp.csr_matrix(trnMat)
        self.metapath_bundle = self.build_metapath_bundle(B_bin, args.drug, args.gene)
        args.metapath_drug_path_order = list(self.metapath_bundle['drug_path_order'])
        args.metapath_gene_path_order = list(self.metapath_bundle['gene_path_order'])
        args.disease = int(self.metapath_bundle['num_disease'])
        print('[CTD tri_dgi_train_hetero MetaPath] paths drug={} gene={} diseases={}'.format(
            args.metapath_drug_path_order, args.metapath_gene_path_order, args.disease))
        self.metapath_bundle = self.augment_metapath_bundle_precompute(self.metapath_bundle)

        trnData = TrnData(train_labels, d_train_idx, g_train_idx)
        self.trnLoader = dataloader.DataLoader(trnData, batch_size=args.batch, shuffle=False,
                                               num_workers=0)
        if args.validate:
            tstData = TstData(val_labels, d_val_idx, g_val_idx)
        else:
            tstData = TstData(test_labels, d_test_idx, g_test_idx)
        self.tstLoader = dataloader.DataLoader(tstData, batch_size=args.tstBat, shuffle=False, num_workers=0)


class TrnData(data.Dataset):
    def __init__(self, train_labels, d_train_idx, g_train_idx):
        self.train_labels = train_labels
        self.d_train_idx = d_train_idx
        self.g_train_idx = g_train_idx

    def __len__(self):
        return len(self.train_labels)

    def __getitem__(self, idx):
        return self.d_train_idx[idx], self.g_train_idx[idx], self.train_labels[idx]


class TstData(data.Dataset):
    def __init__(self, test_labels, d_test_idx, g_test_idx):
        self.test_labels = test_labels
        self.d_test_idx = d_test_idx
        self.g_test_idx = g_test_idx

    def __len__(self):
        return len(self.test_labels)

    def __getitem__(self, idx):
        return self.d_test_idx[idx], self.g_test_idx[idx], self.test_labels[idx]
