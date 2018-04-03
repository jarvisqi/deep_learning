# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as fn


def train():

    test_sentence = """When forty winters shall besiege thy brow,
    And dig deep trenches in thy beauty's field,
    Thy youth's proud livery so gazed on now,
    Will be a totter'd weed of small worth held:
    Then being asked, where all thy beauty lies,
    Where all the treasure of thy lusty days;
    To say, within thine own deep sunken eyes,
    Were an all-eating shame, and thriftless praise.
    How much more praise deserv'd thy beauty's use,
    If thou couldst answer 'This fair child of mine
    Shall sum my count, and make my old excuse,'
    Proving his beauty by succession thine!
    This were to be new made when thou art old,
    And see thy blood warm when thou feel'st it cold.""".split()

    trigrams = [([test_sentence[i], test_sentence[i + 1]], test_sentence[i + 2])
                for i in range(len(test_sentence) - 2)]
    print(trigrams[:3])
    vocab = set(test_sentence)
    word_to_ix = {word: i for i, word in enumerate(vocab)}

    CONTEXT_SIZE = 2
    EMBEDDING_DIM = 10
    lossess = []
    loss_fn = torch.nn.NLLLoss()
    model = EmbeddingNet(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    for epoch in range(200):
        total_loss = torch.Tensor([0])
        for context, target in trigrams:

            context_idxs = [word_to_ix[w] for w in context]
            context_var = torch.autograd.Variable(
                torch.LongTensor(context_idxs))

            optimizer.zero_grad()
            log_probs = model(context_var)
            loss = loss_fn(log_probs, torch.autograd.Variable(
                torch.LongTensor([word_to_ix[target]])))
            loss.backward()
            optimizer.step()

            total_loss += loss.data

        lossess.append(total_loss)

    print("Losses \n\n")
    print(lossess)


class EmbeddingNet(torch.nn.Module):
    """
    文本Embedding
    """

    def __init__(self, vocab_size: int, embedding_dim: int, context_size: int):

        super(EmbeddingNet, self).__init__()

        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = torch.nn.Linear(context_size*embedding_dim, 128)
        self.fc2 = torch.nn.Linear(128, vocab_size)
        self.relu = torch.nn.ReLU()

    def forward(self, inputs):
        embeds = self.embedding(inputs).view((1, -1))
        out = fn.relu(self.fc1(embeds))
        # out = self.relu(self.fc1(embeds))
        out = self.fc2(out)
        log_probs = fn.log_softmax(out, dim=1)

        return log_probs


if __name__ == '__main__':
    train()
