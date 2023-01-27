#
from .MLP import MLP
from .GraphMFN import GraphMFN
from .QDNN import uQDNN
from .QDNNAblation import QDNNAblation
from .LocalMixtureNN import LocalMixtureNN
from .CFN import CFN
from .RAVEN import RAVEN
from .EFLSTM import EFLSTM
from .TFN import TFN
from .MARN import MARN
from .RMFN import RMFN
from .LMF import LMF
from .MFN import MFN
from .LSTHM import LSTHM
from .LFLSTM import LFLSTM
from .MULT import MULT
from .QDNN_ATTENTION import uQDNN_ATTENTION
from .QMULT import uMULT
from .LocalMixtureNN_attetnion import LocalMixtureNN_attention
from .LocalMixtureNN_attetnion_T import LocalMixtureNN_attention_T
from .LocalMixtureNN_attetnion_A import LocalMixtureNN_attention_A
from .LocalMixtureNN_attetnion_V import LocalMixtureNN_attention_V
from .LocalMixtureNN_attetnion_TV import LocalMixtureNN_attention_TV
from .LocalMixtureNN_attetnion_TA import LocalMixtureNN_attention_TA
from .LocalMixtureNN_attetnion_VA import LocalMixtureNN_attention_VA
def setup(opt):
    
    print("network type: " + opt.network_type)
    if opt.network_type == "mlp":
        model = MLP(opt) 
    elif opt.network_type == "ef-lstm":
        model = EFLSTM(opt)
    elif opt.network_type == "tfn":
        model = TFN(opt)
    elif opt.network_type == "qdnn-ablation":
        model = QDNNAblation(opt)
    elif opt.network_type == "marn":
        model = MARN(opt)
    elif opt.network_type == "rmfn":
        model = RMFN(opt)
    elif opt.network_type == "graph-mfn":
        model = GraphMFN(opt)
    elif opt.network_type == 'lmf':
        model = LMF(opt)
    elif opt.network_type == 'mfn':
        model = MFN(opt)
    elif opt.network_type == 'lsthm':
        model = LSTHM(opt)
    elif opt.network_type == 'lf-lstm':
        model = LFLSTM(opt)
    elif opt.network_type == 'qdnn':
        model = uQDNN(opt)
    elif opt.network_type == 'local_mixture':
        model = LocalMixtureNN(opt)
    elif opt.network_type == 'multimodal-transformer':
        model = MULT(opt)
    elif opt.network_type == 'cfn':
        model = CFN(opt)
    elif opt.network_type == 'raven':
        model = RAVEN(opt)
    elif opt.network_type == 'qdnnattention':
        model = uQDNN_ATTENTION(opt)
    elif opt.network_type == 'quantum-multimodal-transformer':
        model = uMULT(opt)
    elif opt.network_type == 'local_mixture_attention':
        model = LocalMixtureNN_attention(opt)
    elif opt.network_type == 'local_mixture_attention_T':
        model = LocalMixtureNN_attention_T(opt)
    elif opt.network_type == 'local_mixture_attention_A':
        model = LocalMixtureNN_attention_A(opt)
    elif opt.network_type == 'local_mixture_attention_V':
        model = LocalMixtureNN_attention_V(opt)
    elif opt.network_type == 'local_mixture_attention_TV':
        model = LocalMixtureNN_attention_TV(opt)
    elif opt.network_type == 'local_mixture_attention_TA':
        model = LocalMixtureNN_attention_TA(opt)
    elif opt.network_type == 'local_mixture_attention_VA':
        model = LocalMixtureNN_attention_VA(opt)

    else:
        raise Exception("model not supported: {}".format(opt.network_type))
    return model
