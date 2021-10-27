from data import *

class EngFreTranslator:
    def __init__(self, dataset_path):
        self.data = TranslationDataset(dataset_path, "english", "french")
        self.model = None
        self.loss_func = None
        self.optimizer = None




if __name__ == '__main__':
    translator = EngFreTranslator("data/english_french.txt")
