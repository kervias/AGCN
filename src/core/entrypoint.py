from core.task.IRTrain import IR_Train
from core.task.IRTest import IR_Test
from core.task.AI import AI
from core.task.LP import LP_Manager
from core.task.NGCF import NGCF_Manager
from core.task.SemiGCN import SemiGCN_Manager


class EntryPoint(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def start(self):
        if self.cfg.task == 'IR' and self.cfg.istrain is True:
            task = IR_Train(self.cfg)
            task.train()
        elif self.cfg.task == 'IR' and self.cfg.istrain is False:
            task = IR_Test(self.cfg)
            task.test()
        elif self.cfg.task == 'AI':
            task = AI(self.cfg)
            task.start()
        elif self.cfg.task == 'LP':
            task = LP_Manager(self.cfg)
            task.start()
        elif self.cfg.task == 'Semi-GCN':
            task = SemiGCN_Manager(self.cfg)
            task.start()
        elif self.cfg.task == 'NGCF':
            task = NGCF_Manager(self.cfg)
            task.start()
        else:
            raise Exception("unknown task")
