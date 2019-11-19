import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data

def get_threshold(score, ratio=0.1):
    n = int(len(score)*ratio)
    sc = sorted(score, reverse=True)
    return sc[n]


class CustomLoss(nn.Module):
    def log_loss(self, z):
        return torch.log(torch.exp(-z)+1)

    def sigmoid_loss(self, z):
        return torch.sigmoid(-z)

    def return_loss_func_by_str(self, r_cost=None):
        loss_func = None
        if 'log' in self.loss:
            loss_func = self.log_loss
        elif 'sigmoid' in self.loss:
            loss_func = self.sigmoid_loss
        return loss_func


class RCNN(CustomLoss):
    def __init__(self, loss, vocab_dim, embedding_dim, hidden_dim, weights, num_layers=1):
        super(RCNN, self).__init__()
        self.loss = loss
        self.loss_func = self.return_loss_func_by_str()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_dim, embedding_dim)
        if weights is not None :
            self.word_embeddings.weight = nn.Parameter(weights, requires_grad=False)
        self.drop_out = 0.6
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,dropout=self.drop_out, bidirectional=True)
        self.W2 = nn.Linear(2*hidden_dim+embedding_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, doc):
        embeds = self.word_embeddings(doc)
        embeds = embeds.permute(1, 0, 2)
        o, (h_s, c_s) = self.lstm(embeds)
        final = torch.cat((o, embeds), 2).permute(1, 0, 2)
        h = self.W2(final)
        h = h.permute(0, 2, 1)
        h = F.max_pool1d(h, h.size(2))
        h = h.squeeze(2)
        out = self.linear(h)
        return out


class Cor(RCNN):
    def get_auc_loss(self, x_p, x_n, n_p, n_n):
        if n_p > 0 :
            if n_n > 0:
                gg = (torch.t(((self.forward(x_p)).repeat(1, n_n))) - ((self.forward(x_n)).repeat(1, n_p))).reshape(-1, 1)
            else:
                n_n = 1
                gg = (torch.t(((self.forward(x_p)).repeat(1, n_n)))).reshape(-1, 1)
        else:
            n_p = 1
            gg = (-((self.forward(x_n)).repeat(1, n_p))).reshape(-1, 1)
        return 1 / (n_p * n_n) * torch.sum(self.loss_func(gg))

    def __call__(self, data, label):
        x = data
        x_p = x[label==1]
        n_p = len(x_p)
        x_u = x[label==-1]
        n_u = len(x_u)

        loss = self.get_auc_loss(x_p, x_u, n_p, n_u)

        return loss

    def forward_test(self, x):
        return self.forward(x)


def PU_DEEP(train_loader, test_loader, vocab_dim, weights=None, embedding_dim=300, num_layers=2, epoch=50, stepsize=1e-4, hidden_size=256, loss='sigmoid', device=-1):
    model = Cor(loss, vocab_dim=vocab_dim, embedding_dim=embedding_dim, num_layers=num_layers, hidden_dim=hidden_size, weights=weights).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=stepsize, weight_decay=3e-3)
    for qq in range(epoch):
        print('%d/%d' % (qq+1, epoch))
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            doc = batch.text[0]
            label = batch.label

            out = model(doc, label)
            out.backward()
            optimizer.step()
    train_rslt = torch.tensor([])
    train_lbl = torch.tensor([])
    test_rslt = torch.tensor([])
    test_lbl = torch.tensor([])
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            doc = batch.text[0]
            x = model.forward(doc).view(-1)
            train_rslt = torch.cat((train_rslt, x.cpu()))
            train_lbl = torch.cat((train_lbl, batch.label.cpu()))
        for i, batch in enumerate(test_loader):
            doc = batch.text[0]
            x = model.forward(doc).view(-1)
            test_rslt = torch.cat((test_rslt, x.cpu()))
            test_lbl = torch.cat((test_lbl, batch.label.cpu()))
    return test_rslt, test_lbl, train_rslt, test_lbl
