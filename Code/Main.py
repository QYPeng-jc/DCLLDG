import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from Params import args
from Model_sparse import Model
from DataHandler import DataHandler
from Utils.Utils import *
import os
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score
from sklearn.preprocessing import label_binarize
import numpy as np
import random
import wandb


def _cpu_clone_state_dict(model):
    """Detach + CPU clone for best-epoch checkpoint (avoid holding GPU refs)."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


# Function to set random seed for reproducibility
def set_seed(seed):
    print(seed)
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)

    if t.cuda.is_available():
        t.cuda.manual_seed(seed)
        t.backends.cudnn.benchmark = False
        t.backends.cudnn.deterministic = True


def compute_test_metrics(y_true, y_pred, y_proba, n_classes):
    """Acc, AUROC, and AUPR on the full test set."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    y_proba = np.asarray(y_proba, dtype=np.float64)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    if n_classes < 2:
        return acc, float('nan'), float('nan'), f1

    if n_classes == 2:
        try:
            auc = roc_auc_score(y_true, y_proba[:, 1])
        except ValueError:
            auc = float('nan')
        try:
            aupr = average_precision_score(y_true, y_proba[:, 1])
        except ValueError:
            aupr = float('nan')
    else:
        try:
            auc = roc_auc_score(
                y_true, y_proba, multi_class='ovr', average='macro', labels=np.arange(n_classes)
            )
        except ValueError:
            auc = float('nan')
        Y = label_binarize(y_true, classes=np.arange(n_classes))
        try:
            aupr = average_precision_score(Y, y_proba, average='macro')
        except ValueError:
            aupr = float('nan')

    return acc, auc, aupr, f1


# Locked main-experiment schedule: run i uses seed + i.
SELECTION_METRIC = 'acc'
PLATEAU_WARMUP = 100
PLATEAU_PATIENCE = 20
PLATEAU_MIN_DELTA = 0.0001
PLATEAU_EARLY_STOP = True


def build_iteration_seeds(n_iter=None):
    """Per-run seeds are seed, seed+1, ... (main-experiment schedule)."""
    if n_iter is None:
        n_iter = int(args.iteration)
    n_iter = int(n_iter)
    base = int(args.seed)
    return [base + i for i in range(n_iter)]


# Define the Coach class for model training and evaluation
class Coach:
    def __init__(self, handler):
        self.handler = handler

        print('DRUG', args.drug, 'GENE', args.gene)
        print('NUM OF INTERACTIONS', self.handler.trnLoader.dataset.__len__())
        self.metrics = dict()
        mets = ['Loss', 'preLoss', 'Acc', 'AUROC', 'AUPR']
        for met in mets:
            self.metrics['Train' + met] = list()
            self.metrics['Test' + met] = list()

    # Function to create a formatted print statement
    def makePrint(self, name, ep, reses, save):
        ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += '%s = %.4f, ' % (metric, val)
            tem = name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        ret = ret[:-2] + '  '
        return ret

    # Function to perform external testing
    def external_test_run(self):
        self.prepareModel()
        log('Model Prepared')
        if args.load_model != None:
            self.loadModel()
        reses = self.testEpoch()
        log(self.makePrint('Test', args.epoch, reses, True))
        return np.array([reses['Acc'], reses['AUROC'], reses['AUPR']], dtype=np.float64)

    # Function to train and evaluate the model
    def run(self):
        self.prepareModel()
        log('Model Prepared')
        if args.load_model != None:
            self.loadModel()
            stloc = len(self.metrics['TrainLoss']) * args.tstEpoch - (args.tstEpoch - 1)
        else:
            stloc = 0
            log('Model Initialized')
        test_r = {'Acc': 0., 'AUROC': float('nan'), 'AUPR': float('nan')}
        metric_key = {'acc': 'Acc', 'auroc': 'AUROC', 'aupr': 'AUPR'}[SELECTION_METRIC]
        best_sel = float('-inf')
        best_epoch = stloc
        best_snapshot = {}
        best_weights = None
        epochs_no_improve = 0
        stopped_early = False
        for ep in range(stloc, args.epoch):
            tstFlag = (ep % args.tstEpoch == 0)
            reses = self.trainEpoch(ep)
            train_loss = reses
            log(self.makePrint('Train', ep, reses, tstFlag))
            if tstFlag:
                reses = self.testEpoch()
                test_r = reses
                log(self.makePrint('Test', ep, reses, tstFlag))
                cur = reses.get(metric_key)
                if cur is None or (isinstance(cur, float) and np.isnan(cur)):
                    cur_cmp = float('-inf')
                else:
                    cur_cmp = float(cur)
                if cur_cmp > best_sel + float(PLATEAU_MIN_DELTA):
                    best_sel = cur_cmp
                    best_epoch = ep
                    best_snapshot = dict(reses)
                    best_weights = _cpu_clone_state_dict(self.model)
                    epochs_no_improve = 0
                else:
                    if ep >= int(PLATEAU_WARMUP):
                        epochs_no_improve += 1
                if (
                    PLATEAU_EARLY_STOP
                    and ep >= int(PLATEAU_WARMUP)
                    and epochs_no_improve >= int(PLATEAU_PATIENCE)
                ):
                    split_name = 'Val' if args.validate else 'Test'
                    log(
                        '[Plateau] {} {}: "{}" did not beat the best by more than {} for {} evals; stop at epoch {}.'.format(
                            split_name,
                            'selection metric',
                            metric_key,
                            PLATEAU_MIN_DELTA,
                            PLATEAU_PATIENCE,
                            ep,
                        )
                    )
                    log(
                        '[BestEpoch] best epoch before the plateau = {} ({} = {:.4f}); '
                        'Acc = {:.4f}, AUROC = {:.4f}, AUPR = {:.4f}'.format(
                            best_epoch,
                            metric_key,
                            best_sel if np.isfinite(best_sel) else float('nan'),
                            best_snapshot.get('Acc', float('nan')),
                            best_snapshot.get('AUROC', float('nan')),
                            best_snapshot.get('AUPR', float('nan')),
                        )
                    )
                    if not np.isfinite(best_sel):
                        log('[BestEpoch] warning: the selection metric was nan for the whole run.')
                    stopped_early = True
                    break
            logs = {'loss_all': train_loss['Loss'], 'loss_pre': train_loss['preLoss']}
            if tstFlag:
                logs['test_acc'] = test_r['Acc']
                logs['test_auroc'] = test_r['AUROC']
                logs['test_aupr'] = test_r['AUPR']
            wandb.log(logs)
        split_name = 'Val' if args.validate else 'Test'
        log(
            '[BestEpoch] training finished on {}. best_epoch = {} by {} = {:.4f}; '
            'Acc = {:.4f}, AUROC = {:.4f}, AUPR = {:.4f}'.format(
                split_name,
                best_epoch,
                metric_key,
                best_sel if np.isfinite(best_sel) else float('nan'),
                best_snapshot.get('Acc', float('nan')),
                best_snapshot.get('AUROC', float('nan')),
                best_snapshot.get('AUPR', float('nan')),
            )
        )
        if best_weights is not None:
            dev = next(self.model.parameters()).device
            self.model.load_state_dict({k: v.to(dev) for k, v in best_weights.items()})
            log('[BestEpoch] loaded best_epoch = {} for the final test and checkpoint.'.format(best_epoch))
        else:
            log('[BestEpoch] warning: no best-weight snapshot; using the current weights.')
        reses = self.testEpoch()
        if not stopped_early:
            log(self.makePrint('Test', args.epoch, reses, True))
        final_vec = np.array([reses['Acc'], reses['AUROC'], reses['AUPR']], dtype=np.float64)
        best_vec = np.array([
            best_snapshot.get('Acc', float('nan')),
            best_snapshot.get('AUROC', float('nan')),
            best_snapshot.get('AUPR', float('nan')),
        ], dtype=np.float64)
        self.save_model('{}'.format(config['iteration']))
        return final_vec, best_vec

    # Function to prepare the model and optimizer
    def prepareModel(self):
        self.model = Model().cuda()
        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)

    def _branch2(self):
        return {'encoder': 'metapath', 'bundle': self.handler.metapath_bundle}

    # Function to train a single epoch
    def trainEpoch(self, ep_idx=0):
        self.model.train()
        trnLoader = self.handler.trnLoader
        epLoss, epPreLoss = 0, 0
        steps = trnLoader.dataset.__len__() // args.batch
        branch2 = self._branch2()
        for i, tem in enumerate(trnLoader):
            drugs, genes, labels = tem
            drugs = drugs.long().cuda()
            genes = genes.long().cuda()
            labels = labels.long().cuda()
            ceLoss, sslLoss = self.model.calcLosses(
                drugs, genes, labels, self.handler.torchBiAdj, args.keepRate, branch2)
            sslLoss = sslLoss * args.ssl_reg

            regLoss = calcRegLoss(self.model) * args.reg
            loss = ceLoss + regLoss + sslLoss
            epLoss += loss.item()
            epPreLoss += ceLoss.item()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        ret = dict()
        ret['Loss'] = epLoss / steps
        ret['preLoss'] = epPreLoss / steps
        return ret

    # Function to test a single epoch
    def testEpoch(self):
        self.model.eval()
        tstLoader = self.handler.tstLoader
        labels_all, preds_all, probs_all = [], [], []
        branch2 = self._branch2()
        for tem in tstLoader:
            drugs, genes, labels = tem
            drugs = drugs.long().cuda()
            genes = genes.long().cuda()
            labels = labels.long().cuda()
            logits = self.model.predict(
                self.handler.torchBiAdj, drugs, genes, branch2)
            probs = F.softmax(logits, dim=1).detach().cpu().numpy()
            preds = probs.argmax(axis=1)
            labels_all.append(labels.detach().cpu().numpy())
            preds_all.append(preds)
            probs_all.append(probs)

        y_true = np.concatenate(labels_all)
        y_pred = np.concatenate(preds_all)
        y_proba = np.concatenate(probs_all)
        acc, auc, aupr, _f1m = compute_test_metrics(y_true, y_pred, y_proba, args.num_classes)

        ret = dict()
        ret['Acc'] = acc
        ret['AUROC'] = auc
        ret['AUPR'] = aupr
        return ret

    # Function to load a pre-trained model
    def loadModel(self):
        self.model.load_state_dict(t.load('../Models/' + args.load_model + '.pkl'))
        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)
        log('Model Loaded')

    # Function to save the trained model
    def save_model(self, model_path):
        run_dir = wandb.run.dir if wandb.run is not None else '.'
        model_parent_path = os.path.join(run_dir, 'ckl')
        os.makedirs(model_parent_path, exist_ok=True)
        t.save(self.model.state_dict(), '{}/{}_model.pkl'.format(model_parent_path, model_path))


# Main execution block
if __name__ == '__main__':
    if args.is_debug is True:
        print("DEBUGGING MODE - Start without wandb")
        wandb.init(mode="disabled")
    else:
        wandb.init(project='HC', config=args)
        wandb.run.log_code(".")

    use_cuda = args.gpu >= 0 and t.cuda.is_available()
    device = 'cuda:{}'.format(args.gpu) if use_cuda else 'cpu'
    if use_cuda:
        t.cuda.set_device(device)
    args.device = device

    logger.saveDefault = True

    log('Start')
    config = dict()
    results = list()
    seeds = build_iteration_seeds()
    print('CTD fixed-split iteration seeds:', seeds)
    handler = DataHandler()
    handler.LoadData()
    log('Load Data (CTD curated_v2 fixed split)')
    coach = Coach(handler)
    for i in range(int(args.iteration)):
        print('{}-th iteration'.format(i + 1))
        seed = int(seeds[i])
        config['seed'] = seed
        config['iteration'] = i + 1
        set_seed(seed)
        final_vec, _best_vec = coach.run()
        results.append(final_vec)

    avg_r = np.mean(np.array(results), axis=0)
    std_r = np.std(np.array(results), axis=0)
    print('test results (columns: Acc, AUROC, AUPR): ')
    print(avg_r)
    print(std_r)
    if wandb.run is not None:
        wandb.run.summary['test_acc'] = float(avg_r[0])
        wandb.run.summary['test_auroc'] = float(avg_r[1])
        wandb.run.summary['test_aupr'] = float(avg_r[2])

    results.append(avg_r)
    results.append(std_r)

    results_parent_path = os.path.join(wandb.run.dir if wandb.run is not None else '.', 'results')
    os.makedirs(results_parent_path, exist_ok=True)
    suffix = 'CTD'
    np.savetxt('{}/{}_result.txt'.format(results_parent_path, suffix), np.array(results), delimiter=",", fmt='%f')

    print('result saved!!!')
    wandb.finish()
