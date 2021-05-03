from core.task.IRTrain import IR_Train
from core.task.IRTest import IR_Test
from core.task.AI import AI


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
        else:
            raise Exception("unknown task")
