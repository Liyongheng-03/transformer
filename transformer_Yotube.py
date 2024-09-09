"""
A from scratch implementation of Transformer network,
following the paper Attention is all you need with a
few minor differences. I tried to make it as clear as
possible to understand and also went through the code
on my youtube channel!
"""

import torch
import torch.nn as nn

class SelfAttention(nn.Module): # 自注意力机制中，值、键和查询通常从相同的输入序列中产生，但通过不同的线性层进行变换
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size # 嵌入层的大小，即输入序列的特征维度
        self.heads = heads # 注意力机制中使用的头（heads）的数量，多头的数量
        self.head_dim = embed_size // heads  # 每个头处理的特征维度

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"
        # 允许每个头专注于输入的不同特征，然后将这些特征的表示合并起来，以获得更全面的输出(embed_size, embed_size)
        self.values = nn.Linear(embed_size, embed_size) # 这个线性层将输入的值（values）从embed_size维度映射回embed_size维度
        self.keys = nn.Linear(embed_size, embed_size) # 
        self.queries = nn.Linear(embed_size, embed_size) #
        self.fc_out = nn.Linear(embed_size, embed_size) #

    def forward(self, values, keys, query, mask):
        # Get number of training examples
        N = query.shape[0]

        # 对于处理序列数据的模型，如Transformer中的自注意力机制，张量的维度通常具有以下含义：
        # 第一个维度：通常表示批次大小（batch size）。批次大小指的是每次模型训练或推理时处理的样本数量。
        # 第二个维度：在自注意力机制中，这个维度通常表示序列长度或时间步长（time steps）,对于query、keys和values，这个维度表示序列中的元素数量。
        # 后续维度：这些维度可以表示特征维度、嵌入维度或在多头注意力中表示不同的头。

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        # 进行线性变换，以将它们映射到模型的嵌入维度空间
        # 对此我表示非常懵逼，这里的size
        values = self.values(values)  # (N, value_len, embed_size) 
        keys = self.keys(keys)  # (N, key_len, embed_size)
        queries = self.queries(query)  # (N, query_len, embed_size)

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        # Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how I like doing matrix multiplication & bmm

        # 它是一个四维张量的运算,对于每个头h，对于批次n中的每个查询q，我们计算它与''所有键k''的点积

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        # print(energy)
        # print("energy[1].size",energy.size)
        # queries shape: (N, query_len, heads, heads_dim),
        # keys shape: (N, key_len, heads, heads_dim)
        # energy: (N, heads, query_len, key_len)
        # 这会产生一个 (N, H, Q, K) 形状的张量，其中每个元素代表一个查询向量与一个键向量的点积

        # Mask padded indices so their weights become 0
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        # 这行代码使用masked_fill函数将energy张量中，掩码为0的位置替换为一个非常大的负数（-1e20）。
        # 这个操作的目的是在后续的softmax操作中，将这些位置的注意力权重设置为接近于0。
        # 因为在softmax函数中，一个非常大的负数会映射到一个非常接近0的概率【做的一个幂指函数的操作】

        # Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for
        # better stability

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        # attention shape: (N, heads, query_len, key_len)
        # 对key_len维度执行softmax确保了每个查询向量与所有键向量的关系被适当地归一化，从而可以计算加权的值。

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # out after matrix multiply: (N, query_len, heads, head_dim), then
        # we reshape and flatten the last two dimensions.
        # 这里就是attention score与values做加权求和

        out = self.fc_out(out)
        # Linear layer doesn't modify the shape, final shape will be
        # (N, query_len, embed_size)

        return out


class TransformerBlock(nn.Module): # 输入的size和输出的相同
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size) 
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query)) # attention + query是残差连接
        forward = self.feed_forward(x) # 前馈网络
        out = self.dropout(self.norm2(forward + x)) 
        return out


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size, # 源词汇表的大小，即输入词汇的总数
        embed_size, # 嵌入层的大小，也是模型中特征的维度
        num_layers, # 编码器中 Transformer 块的数量，即N×
        heads,
        device,
        forward_expansion,
        dropout,
        max_length, # 输入序列的最大长度
    ):

        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device 
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)# 一个词嵌入层，将输入的单词索引转换为对应的嵌入向量
        self.position_embedding = nn.Embedding(max_length, embed_size)# 一个位置嵌入层，为输入序列的每个位置提供位置信息
 
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):

        N, seq_length = x.shape

        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)  # 这里的位置是可以学习的，可以考虑改成正余弦波

        out = self.dropout(
            (self.word_embedding(x) + self.position_embedding(positions))
        )

        # In the Encoder the query, key, value are all the same, it's in the
        # decoder this will change. This might look a bit odd in this case.
        for layer in self.layers: # 在 Transformer编码器中，查询、键和值都是相同的，即前一层的输出
            out = layer(out, out, out, mask)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device): # forward_expansion将输入张量映射到一个更高维度的空间
        super(DecoderBlock, self).__init__()
        self.norm = nn.LayerNorm(embed_size)
        self.attention = SelfAttention(embed_size, heads=heads)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        # src_mask：源掩码，用于在编码器-解码器注意力中屏蔽不相关信息。
        # trg_mask: 目标掩码，用于在自注意力中屏蔽填充（padding）或其他不需要关注的位置
        attention = self.attention(x, x, x, trg_mask)
        # 每个DecoderBlock中，前一步骤的输出（经过自注意力和归一化处理后的结果）将作为查询（query）输入到下一个TransformerBlock中。
        query = self.dropout(self.norm(attention + x)) 
        # query 来自解码器，而value和key来自编码器的输入

        # 编码器-解码器之间的注意力机制
        out = self.transformer_block(value, key, query, src_mask) 
        return out


class Decoder(nn.Module):
    def __init__(
        self,
        trg_vocab_size, # 目标词汇表的的大小
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
        max_length,
    ):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)
            
        # x:前一解码器层的输出,第二个和第三个enc_out参数分别代表编码器的输出，用作编码器-解码器注意力机制的键（K）和值（V）
        out = self.fc_out(x)
        # 解码器中的每个 DecoderBlock 都包含自注意力和编码器-解码器注意力机制，允许模型在生成序列时考虑内部和外部的上下文信息。
        return out


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size, # 源语言的词汇表大小
        trg_vocab_size, # 目标语言的词汇表大小
        src_pad_idx, # 源语言的填充索引，用于在序列中标记填充位置
        trg_pad_idx, # 目标语言的填充索引
        embed_size=512,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0,
        device="cpu",
        max_length=100,
    ):

        super(Transformer, self).__init__()

        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
        )

        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length,
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src): # src 是源序列的索引张量
        print(src)
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        # (src != self.src_pad_idx)比较源序列xrc中的每个索引，找出不等于填充索引self.src_pad_idx的位置，结果是一个由布尔值组成的张量
        return src_mask.to(self.device)
        # 返回一个在填充位置为0（False），其他位置为1（True）的掩码张量

    def make_trg_mask(self, trg): # 为解码器的自注意力机制生成掩码，以防止解码器在生成序列时看到未来的信息（“掩蔽未来信息”）
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        # torch.tril(torch.ones((trg_len, trg_len))) 创建一个下三角矩阵，对角线和左下部分为1，右上部分为0。
        # 这个矩阵表示在目标序列中，每个位置只能关注到它之前（包括自身）的位置

        return trg_mask.to(self.device)
        # 返回一个形状为 (batch_size, 1, trg_len, trg_len) 的掩码张量，其中未来的信息位置为0

        # make_src_mask 确保编码器在处理源序列时忽略填充位置，从而只关注实际的输入数据。
        # make_trg_mask 确保解码器在生成每个词时，只能使用已经生成的词（包括目标序列中的起始标记）和编码器的输出，而不能利用未来词的信息，这有助于避免在序列生成中产生错误依赖。


    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        # 第一个是目标语言的输入，第二个是经过掩码处理的encoder的输入，第三个参数是src_mask，第四个参数trg_mask
        return out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)

    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device=device).to(device)
    out = model(x, trg[:, :-1])
    print(out.shape)