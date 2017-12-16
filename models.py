'''
Model setups
'''
from sklearn import svm
import preprocess
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class lstm_sentiment(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(lstm_sentiment, self).__init__()

        #We will probably need to change this to get it to work correctly with w2v
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (Variable(torch.zeros(1,1,self.hidden_dim)),
                Variable(torch.zeros(1,1,self.hidden_dim)))

    def forward(self, sent):
        embeds = self.word_embeddings(sent)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sent), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space)
        return tag_scores

def init_baseline(data, targets):
    baseline = svm.SVC()
    baseline.fit(data, targets)
    return baseline

def train_lstm(data, targets, loss_function, input_len, vocab_n, tag_n, epoches):
    model = lstm_sentiment(input_len, tag_n, vocab_n, tag_n)
    optimizer = optim.SGD(model.parameters(), lr = 0.1)

    for epoch in range(epoches):
        for sentence, tag in zip(data, targets):
            model.zero_grad()

            model.hidden = model.init_hidden()

            tag_score = model(sentence)

            loss = loss_function(tag_scores, tag)
            loss.backward()
            optimizer.step()
    return model

def main():
    d = ['Hi bob builder I am thinking', 'THANKS OBUMMER!!!!!']
    t = [0, -1]
    a, b = preprocess.full_preprocess(d, t, 
            [preprocess.tokenize, preprocess.casing, preprocess.stops,
                preprocess.punctuation, preprocess.stem_all, preprocess.pad], .5)
    print(len(a[0][0]))
    print(len(b[0][0]))
    #print(a[0][0])
    #lstm = nn.LSTM(len(a[0][0]), 3)
    #torched = autograd.Variable(torch.Tensor(a[0][0]))
    #hidden = (autograd.Variable(torch.randn(1, 1, 3)), autograd.Variable(torch.randn((1,1,3))))
    #out, hidden = lstm(torched.view(1, 1, -1), hidden)
    #train_lstm(a[0], a[1], nn.NLLLoss(), XXX, , 3, 300)
    

if __name__ == '__main__':
    main()
