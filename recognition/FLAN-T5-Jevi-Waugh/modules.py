# Fine tuning procedure is in this file
from transformers import AutoTokenizer
import torch.nn as nn



checkpoint = "google/flan-t5-small"
tokeniser = AutoTokenizer.from_pretrained(checkpoint)

class FLAN_T5(nn):
    """This is the base class of FLAN-T5/

    Args:
        nn (_type_): _description_
    """
    def __init__(self):
        pass
    


class FLAN_T5_FULLFINETUNING(FLAN_T5):
    """This class will be used to Fine Tune Flan-T5

    Args:
        FLAN_T5 (_type_): _description_
    """
    pass



class FLAN_T5_loRA(FLAN_T5):
    """This class will be used to finetune FLAN-T5 via LoRA

    Args:
        FLAN_T5 (_type_): _description_
    """
    pass


class FLAN_T5_SPECIAL(FLAN_T5):
    """This is a special strategy that I will reveal once it works.

    Args:
        FLAN_T5 (_type_): _description_
    """
    pass

