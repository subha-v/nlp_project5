from data import *
from model import *

class EngFreTranslator:
    def __init__(self, dataset_path, hidden_dim, num_iters):
        self.num_iters = num_iters
        self.data = TranslationDataset(dataset_path, "english", "french")
        self.encoder = Encoder(self.data.input_vocab.num_words, hidden_dim)
        self.decoder = Decoder(hidden_dim, self.data.target_vocab.num_words) # this ist he output dim
        self.loss_func = nn.NLLLoss()
        

        self.learning_rate = 0.005

        self.encoder_optimizer = torch.optim.SGD(self.encoder.parameters(), lr = self.learning_rate)
        self.decoder_optimizer = torch.optim.SGD(self.decoder.parameters(), lr= self.learning_rate)


    def train_step(self, lang1_idx_tensor, lang2_idx_tensor):
        encoder_state = (self.encoder.init_hidden(), self.encoder.init_hidden())
        # we have both the hidden state and the cell state in the encoder, thats why there are 2 of these

        input_length = lang1_idx_tensor.shape[0]
        target_length = lang2_idx_tensor.shape[0]

        for i in range(input_length):
            encoder_output, encoder_state = self.encoder(lang1_idx_tensor[i], encoder_state)

        decoder_input = torch.tensor([SOS_TOKEN])
        decoder_state = encoder_state

        user_teacher_forcing = True

        loss = 0

        if user_teacher_forcing:
            for j in range(target_length):
                decoder_output, decoder_state = self.decoder(decoder_input, decoder_state)
                loss +=self.loss_func(decoder_output, lang2_idx_tensor[j].unsqueeze(0))
                decoder_input = lang2_idx_tensor[j][None]

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        loss.backward() # computes the gradients
        # loss is what we compute the gradeints on

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.item()/target_length


        # this is our encoder


        



    def train(self):
        for i in range(self.num_iters):
            lang1_idx_tensor, lang2_idx_tensor = self.data.get_random_sample()
            loss = self.train_step(lang1_idx_tensor, lang2_idx_tensor)
            
            # if (i % 1000 == 0):
            print(f"Iter[{i}]: Loss = {loss:0.4f}")



if __name__ == '__main__':
    hidden_dim = 256
    translator = EngFreTranslator("data/english_french.txt", hidden_dim, 100000)
    translator.train()

