import torch
from utils import seed_everything

def softmax(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

def contrastive_loss(x, y, label):
    """
    Contrastive Loss takes a pair of samples labelled positive or negative.
    """
    pass

def norm_temp_scaled_cross_entropy_loss(x_i, x_j, temp=0.1, n_views=2, device='cpu'):
    """
    NT-Xent loss, as described by Chen et al. (2020)

    Args:
        - x_i: Audio logits. Shape: (batch_size, embedding_dim)
        - x_j: Text logits. Shape: (batch_size, embedding_dim)
    """

    # Extend the logits, because we want the similarity for (xj, xi) as well as (xi, xj)
    features_ij = torch.cat([x_i, x_j], dim=0).to(device)
    #features_ji = torch.cat([x_j, x_i], dim=0)
    features_ij = torch.nn.functional.normalize(features_ij, 2).to(device)
    #features_ji = torch.nn.functional.normalize(features_ji, 2)

    similarity_matrix = torch.matmul(features_ij, features_ij.T).to(device)
    labels = torch.cat([torch.arange(x_i.size(0)) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    #labels = torch.eye(x_i.size(0) * n_views).to(device)
    #print(labels.shape)
    #print(labels)
    #(f"Input Shape Audio: {x_i.shape}")
    #print(f"Input Shape Text: {x_j.shape}")
    # Create a list of positive pairs x_i and x_j, with different augmentations
    # = torch.stack([torch.cosine_similarity(aug_1, aug_2, dim=0) for j, aug_2 in enumerate(x_j) for i, aug_1 in enumerate(x_i) if i==j], dim=0)
    # Create a list of negative pairs
    #s_jk = torch.stack([torch.cosine_similarity(aug_1, aug_2, dim=0) for j, aug_2 in enumerate(x_j) for i, aug_1 in enumerate(x_i) if i!=j], dim=0)

    #cs = [torch.stack([torch.cosine_similarity(audio, text, dim=-1) for text in (x_j)], dim=0) for audio in (x_i)]
    
    #print(simatrix.shape)
    #similarity_matrix = torch.tensor([[torch.cosine_similarity(feat_1, feat_2, dim=0) for feat_2 in (features_ji)] for feat_1 in (features_ij)],  requires_grad=True)
    #print(similarity_matrix.shape)
    #ask = torch.eye(similarity_matrix.size(0), dtype=torch.bool)
    #positive_pairs = torch.sum(x_i * x_j, dim=-1) / temp

    # Mask removes positive pairs
    #negative_pairs = similarity_matrix.masked_select(~mask).view(len(similarity_matrix), -1).sum(dim=-1) / temp
    
    
    #logits = torch.cat([positive_pairs, negative_pairs], dim=1)
    #labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
    #print(labels)
    
    #print(loss_ij)
    #loss_ji = torch.nn.functional.cross_entropy(positive_pairs, loss, reduction='mean')

    #loss = torch.mean(torch.sum(loss_ij, loss_ji))

    #(target)
    #loss = torch.nn.functional.cross_entropy(cs / temp, )

    #s_ij = torch.as_tensor(s_ij, device=device)
    #s_jk = torch.as_tensor(s_jk, device=device)

    #print(f"Shape Positive: {s_ij.shape}")
    #print(f"Shape Negative: {s_jk.shape}")

    #loss_1 = -torch.log(torch.exp(s_ij / temp) / torch.sum(torch.exp(s_jk / temp), dim=0)) # Minimize distance between positive pairs
    #loss_2 = -torch.log(torch.exp(s_jk / temp) / torch.sum(torch.exp(s_ij / temp), dim=0)) # Maximize distance between negative pairs

    #print(f"Loss 1 Shape: {loss_1.shape}")
    #print(f"Loss 2 Shape: {loss_2.shape}")

    #loss = torch.mean(loss_1.mean() + loss_2.mean()) # mean of [l(2k-1, 2k) + l(2k, 2k-1)]
    #loss = -torch.log(torch.exp(positive_pairs) / torch.exp(negative_pairs)).mean()




    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device) # 1 = Self-similarity, and must be removed.
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    #print(positives)
    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1) # (batch_size, batch_size-1). The -1 is self-similarity.
    #print(negatives.shape)
    logits = torch.cat([positives, negatives], dim=1)
    # These labels correspond to the positive logits. Since 0 is the positive class, these will always be 0.
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
    #print(labels)
    #print(labels.shape)
    logits = logits / temp

    loss = torch.nn.functional.cross_entropy(logits, labels, reduction='mean')

    return loss

def info_nce_loss():
    pass

def symmetric_cross_entropy(audio_feats, text_feats, labels, reduction:str='mean'):
    logits = torch.stack([torch.cosine_similarity(input_a, input_t, dim=0) for j, input_t in enumerate(text_feats) for i, input_a in enumerate(audio_feats) if i==j], dim=0)
    loss_a = torch.nn.functional.cross_entropy(logits, labels)
    loss_t = torch.nn.functional.cross_entropy(logits, labels)
    return (loss_a + loss_t)/2

if __name__ == "__main__":

    # Test NT-Xent loss
    seed_everything.seed_everything(1234)

    batch_size = 64
    x_i = torch.rand((batch_size, 512))
    x_j = torch.rand((batch_size, 512))
    x_j[0:100] = x_i[0:100]

    loss = norm_temp_scaled_cross_entropy_loss(x_i, x_j, 1)
    print(loss)

    # Test Symmetric Cross Entropy
    #symmetric_xent = symmetric_cross_entropy(x_i, x_j)
    #print(symmetric_xent)

