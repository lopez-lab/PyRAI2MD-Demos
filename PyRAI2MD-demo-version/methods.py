## QM and ML methods for PyRAIMD
## Jingbai Li Jul 11 2020 

from qc_molcas import MOLCAS
from qc_bagel import BAGEL
from model_NN import DNN
from model_GP import GPR

class QM:
    ## This class recieve method name (qm) and variables (method_variables)
    ## This class identify the method that will be used in MD

    def __init__(self,qm,variables_all,id=None):
        qm_list  = {
        'molcas' : MOLCAS, ## this will be classes
        'bagel'  : BAGEL,
        'nn'     : DNN,
        'gp'     : GPR,
        }
        self.method=qm_list[qm](variables_all,id=id) # This should pass hypers

    def train(self):
        self.method.train()

    def load(self):                #This should load model
        self.method.load()
        return self

    def appendix(self,addons):         #appendix function to pass more info for different methods
        self.method.appendix(addons)
        return self

    def evaluate(self,x):
        return self.method.evaluate(x)
