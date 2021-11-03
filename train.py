from data import *
from model import *

class EngFreTranslator:
    def __init__(self, dataset_path, hidden_dim, num_iters):
        self.num_iters = num_iters
        self.data = TranslationDataset(dataset_path, "english", "french")
        self.encoder = Encoder([self.data.input_vocab.num_words, hidden_dim)
        self.decoder = Decoder(hidden_dim, self.data.target_vocab.num_words) # this ist he output dim
        self.loss_func = nn.NLLoss()
        self.optimizer = torch.optim.

        self.learning_rate = 0.005

        self.encoder_optimizer = torch.optim.SGD(self.encoder.parameters(), lr = self.learning_rate)
        self.decoder.optimizer = torch.optim.SGD(self.decoder.parameters(), lr= self.learning_rate)


    def train_step(self, lang1_idx_tensor, lang2_idx_tensor):
        encoder_hidden = self.encoder.init_hidden()
        



    def train(self):
        for i in range(self.num_iters):
            lang1_idx_tensor, lang2_idx_tensor = self.data.get_random_sample()
            output, loss = self.train_step(lang1_idx_tensor, lang2_idx_tensor)

            if (i % 1000 == 0):
                print(F"Iter[{i}]: Loss = {loss:0.4f}")



if __name__ == '__main__':
    hidden_dim = 256
    translator = EngFreTranslator("data/english_french.txt", hidden_dim)
