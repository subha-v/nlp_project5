from data import *
from model import *

class EngFreTranslator:
    def __init__(self, train_dataset_path, val_dataset_path, hidden_dim, num_iters):
        self.num_iters = num_iters
        #self.save_encoder_path = save_encoder_path
        #self.save_decoder_path = save_decoder_path

        self.train_data = TranslationDataset(train_dataset_path, "english", "french")
        self.val_data = TranslationDataset(val_dataset_path, "english", "french")
        #elf.data = TranslationDataset(dataset_path, "english", "french")
        self.encoder = Encoder(self.train_data.input_vocab.num_words, hidden_dim)
        self.decoder = Decoder(hidden_dim, self.train_data.target_vocab.num_words) # this ist he output dim
        self.loss_func = nn.NLLLoss()
        #self.save_model_path = save_model_path

        self.learning_rate = 0.005

        self.encoder_optimizer = torch.optim.SGD(self.encoder.parameters(), lr = self.learning_rate)
        self.decoder_optimizer = torch.optim.SGD(self.decoder.parameters(), lr= self.learning_rate)


        def load_model(self):
            self.encoder.load_state_dict(torch.load(self.save_encoder_path))
            self.decoder.load_state_dict(torch.load(self.save_decoder_path))

        def save_model(self):
            torch.save(self.encoder.state_dict(), self.save_encoder_path)
            torch.save(self.decoder.state_dict(), self.save_decoder_path)


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
            lang1_idx_tensor, lang2_idx_tensor = self.train_data.get_random_sample()
            loss = self.train_step(lang1_idx_tensor, lang2_idx_tensor)
            
            # if (i % 1000 == 0):
            print(f"Iter[{i}]: Loss = {loss:0.4f}")

            if(i % 50 == 0):
                self.predict()
                self.evaluate()
            





    def evaluate(self):
        with torch.no_grad():
            num_correct = 0
            total = 0 
            for i in range(50):
                lang1_idx_tensor, lang2_idx_tensor = self.val_data.get_random_sample()# the reason we can do this is because of getitem in data.py so we can check the iterables at those values
                # we could also do this for lists of pairs , but its more clunky. this is a clearner way of dealing with this
                encoder_state = (self.encoder.init_hidden(), self.encoder.init_hidden())
                input_length = lang1_idx_tensor.shape[0]

                for i in range(input_length): # each i refers to an index in our sentence
                    #lang1_idx_tensor corresponds to a word in english, in our deep leaning model we give a tensor of indicies

                    encoder_output, encoder_state = self.encoder(lang1_idx_tensor[i], encoder_state)

                    decoder_input = torch.tensor([SOS_TOKEN])
                    decoder_state = encoder_state

                    output = []

                    target_length = lang2_idx_tensor.shape[0] # this is the target length
                    # this is possible as we're predicting, our output sentence is either shorter or longer than our actual length
                    # assume we always start at the begining and go up until the target_length and if we have any discrepancy in length, we count that as something that is wrong

                    for j in range(self.val_data.max_length):
                        decoder_output, decoder_state = self.decoder(decoder_input, decoder_state)
                        output_idx = decoder_output.argmax(dim=1)
                        decoder_input = output_idx

                        if(j < target_length):
                            if(output_idx.item() == lang2_idx_tensor[j].item()):
                                num_correct += 1 # incrementing the number correct


                        if(output_idx.item() == EOS_TOKEN):
                            break



                    total += target_length

            accuracy = (num_correct/total) * 100
            print(f"Validation Accuracy: {accuracy}%, Num correct: {num_correct}, Total: {total}")

                        


                    


                
               





    def predict(self):
        lang1_idx_tensor, lang2_idx_tensor = self.val_data.get_random_sample()
        # we are getting a random 
        with torch.no_grad(): # we dont want to change the gradients in this stage
            encoder_state = (self.encoder.init_hidden(), self.encoder.init_hidden())
        # we have both the hidden state and the cell state in the encoder, thats why there are 2 of these

            input_length = lang1_idx_tensor.shape[0]
            #target_length = lang2_idx_tensor.shape[0]

            for i in range(input_length):
                encoder_output, encoder_state = self.encoder(lang1_idx_tensor[i], encoder_state)

            decoder_input = torch.tensor([SOS_TOKEN])
            decoder_state = encoder_state

            user_teacher_forcing = True

            loss = 0


            output = []
            
            for j in range(self.val_data.max_length):
                decoder_output, decoder_state = self.decoder(decoder_input, decoder_state)
                output_idx = decoder_output.argmax(dim=1)
                decoder_input = output_idx # the highest probability tensor is the next input 

                output.append(output_idx.item()) #appending the index to the output 

                


                if (output_idx.item()== EOS_TOKEN):
                    break # if it predicts the end of string token then break


            correct_indicies = list(lang2_idx_tensor)
            correct_indicies = [x.item() for x in correct_indicies]


            output_sentence = self.val_data.indicies_to_sentence(output, self.val_data.target_vocab)
            print(f"Model Output: {output_sentence} ")
            print(f"Correct answer: {self.val_data.indicies_to_sentence(correct_indicies, self.val_data.target_vocab)}")



        

        

if __name__ == '__main__':
    hidden_dim = 256
    translator = EngFreTranslator("data/eng_fr_train.txt", "data/eng_fr_val.txt", hidden_dim, 100000)
    translator.train()

