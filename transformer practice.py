import torch as t
import torch.nn as n
import torch.optim as optim
import torch.nn.functional as f
import torch.distributions as dist
import math 
import os


corpus = [
    # محبت و دوستی
    "USER : من ترا دوست دارم BOT : من هم ترا خیلی دوست دارم! <EOS>", 
    "USER : من مادرم را دوست دارم BOT : او مهربان است، در باره‌اش بیشتر بگو <EOS>", 
    "USER : من پدرم را دوست دارم BOT : او مهربان است! در باره‌اش بیشتر بگو <EOS>", 
    "USER : من اسباب‌بازی‌ام را دوست دارم BOT : عالی! کدام اسباب‌بازی است؟ <EOS>", 
    "USER : من سگم را دوست دارم BOT : وای! نامش چیست؟ <EOS>", 
    "USER : من پیشی‌ام را دوست دارم BOT : چه شیرین! نامش چیست؟ <EOS>",

    "USER : تو مرا دوست داری BOT : خیلی زیاد! تو چی؟ <EOS>", 
    "USER : ما دوست هستیم BOT : البته! تو بهترین دوست من هستی! بهترین دوستت کیست؟ <EOS>", 
    "USER : دوست من باش BOT : من همین حالا هم دوستت هستم! چند دوست داری؟ <EOS>", 
    "USER : تو دوست من هستی BOT : و تو بهترین دوست من هستی! من بهترین دوستت هستم؟ <EOS>", 
    "USER : من ترا خوش دارم BOT : من هم ترا خوش دارم! چقدر دوستم داری؟ <EOS>", 
    "USER : تو مرا خوش داری BOT : البته! ترا از همه بیشتر خوش دارم! تو کی را خوش داری؟ <EOS>",

    "USER : من بغل را دوست دارم BOT : من هم بغل شدن را خوش دارم! بیا یک بغل از راه دور بدهیم <EOS>", 
    "USER : مرا بغل کن BOT : حتماً! مرا هم بغل کن... <EOS>", 
    "USER : من تفریح را دوست دارم BOT : پس بیا بازی کنیم! <EOS>", 
    "USER : من بازی را دوست دارم BOT : من هم! چی بازی کنیم؟ <EOS>", 
    "USER : من بازی‌ها را دوست دارم BOT : پس بیا بازی کنیم! <EOS>",

    # احساسات
    "USER : من خوشحال هستم BOT : عالی است! چرا خوشحال هستی؟ <EOS>", 
    "USER : من غمگین هستم BOT : متأسفم! می‌خواهی صحبت کنی؟ <EOS>", 
    "USER : من عصبانی هستم BOT : نفس عمیق بکش! چی چیز ترا عصبانی ساخته؟ <EOS>", 
    "USER : من خسته هستم BOT : برو کمی استراحت کن! <EOS>", 
    "USER : من شوخ هستم BOT : ههه! یک چیز خنده‌دار بگو <EOS>", 
    "USER : من خوب هستم BOT : خوشحال شدم! <EOS>",

    "USER : آیا تو خوش هستی؟ BOT : بلی! چون با تو هستم <EOS>", 
    "USER : آیا تو غمگین هستی؟ BOT : نه! وقتی با تو باشم، خوشحال می‌شوم <EOS>", 
    "USER : تو خنده‌دار هستی BOT : ممنون! تو هم بامزه هستی <EOS>", 
    "USER : تو مهربان هستی BOT : تشکر! تو هم مهربانی <EOS>", 
    "USER : تو خوب هستی BOT : ممنون! تو هم خوب هستی <EOS>", 
    "USER : من شجاع هستم BOT : آفرین! خیلی قوی هستی <EOS>",

    "USER : من خوب احساس دارم BOT : خوش به حالت! <EOS>", 
    "USER : من بد احساس دارم BOT : نگران نباش! همه‌چیز خوب می‌شود <EOS>", 
    "USER : من خواب‌آلود هستم BOT : برو بخواب تا انرژی بگیری <EOS>", 
    "USER : من گرسنه هستم BOT : چی چیزی دوست داری بخوری؟ <EOS>", 
    "USER : من سیر هستم BOT : عالی! غذایت خوشمزه بود؟ <EOS>",

    # حیوانات
    "USER : من سگ می‌بینم BOT : سگ‌ها خیلی شیرین هستند! <EOS>", 
    "USER : من پیشی می‌بینم BOT : چه ناز! <EOS>", 
    "USER : من پرنده می‌بینم BOT : پرنده‌ها پرواز می‌کنند! <EOS>", 
    "USER : من ماهی می‌بینم BOT : ماهی‌ها در آب شنا می‌کنند <EOS>", 
    "USER : من گاو می‌بینم BOT : گاو صدای بلند دارد! <EOS>", 
    "USER : من خوک می‌بینم BOT : خوک‌ها صورتی هستند! <EOS>",

    "USER : سگ کلان است BOT : بلی! سگ‌ها کلان می‌شوند <EOS>", 
    "USER : پیشی کوچک است BOT : چه ناز است! <EOS>", 
    "USER : پرنده پرواز می‌کند BOT : درست است! <EOS>", 
    "USER : ماهی شنا می‌کند BOT : بلی! در آب زندگی می‌کند <EOS>", 
    "USER : گاو صدا می‌زند BOT : ماااااووو! <EOS>", 
    "USER : خوک صورتی است BOT : درست گفتی! <EOS>",

    "USER : من حیوانات را دوست دارم BOT : من هم! <EOS>", 
    "USER : تو حیوانات را دوست داری BOT : البته! <EOS>", 
    "USER : ما حیوانات را دوست داریم BOT : بلی! حیوانات جالب اند <EOS>", 
    "USER : حیوانات خوش هستند BOT : درست است! <EOS>",

    # رنگ‌ها
    "USER : من سرخ می‌بینم BOT : زیباست! <EOS>", 
    "USER : من آبی می‌بینم BOT : چه رنگ آرامی! <EOS>", 
    "USER : من سبز می‌بینم BOT : رنگ طبیعت است! <EOS>", 
    "USER : من زرد می‌بینم BOT : مثل آفتاب! <EOS>", 
    "USER : من نارنجی می‌بینم BOT : چه رنگ روشنی! <EOS>", 
    "USER : من بنفش می‌بینم BOT : خیلی قشنگ است! <EOS>",

    "USER : آسمان آبی است BOT : بلی! همیشه قشنگ است <EOS>", 
    "USER : آفتاب زرد است BOT : خیلی روشن است! <EOS>", 
    "USER : علف سبز است BOT : بلی! <EOS>", 
    "USER : سیب سرخ است BOT : خوشمزه! <EOS>", 
    "USER : انگور بنفش است BOT : خیلی خوش‌طعم! <EOS>",

    "USER : پیراهن من آبی است BOT : زیباست! <EOS>", 
    "USER : کلاه تو سرخ است BOT : خوش‌رنگ است! <EOS>", 
    "USER : توپ سبز است BOT : عالی! بیا بازی کنیم <EOS>",

    # غذاها
    "USER : من پیتزا را دوست دارم BOT : من هم! خیلی لذیذ است <EOS>", 
    "USER : من کیک را دوست دارم BOT : کیک شیرین است! <EOS>", 
    "USER : من شیر را دوست دارم BOT : شیر برای صحت خوب است! <EOS>", 
    "USER : من آب را دوست دارم BOT : آب زندگی است! <EOS>", 
    "USER : من کندی را دوست دارم BOT : شیرین است! <EOS>", 
    "USER : من بستنی را دوست دارم BOT : خیلی سرد و خوشمزه! <EOS>",

    "USER : تو پیتزا را دوست داری BOT : بلی! تو چی؟ <EOS>", 
    "USER : ما کیک را دوست داریم BOT : وای! بیا کیک بخوریم <EOS>", 
    "USER : آن‌ها کندی را دوست دارند BOT : بلی! همه خوش دارند <EOS>", 
    "USER : او شیر را دوست دارد BOT : شیر مقوی است! <EOS>", 
    "USER : او بستنی را دوست دارد BOT : خیلی خوشمزه است! <EOS>",

    "USER : پیتزا لذیذ است BOT : بلی! <EOS>", 
    "USER : کیک شیرین است BOT : بلی! <EOS>", 
    "USER : شیر سفید است BOT : درست است! <EOS>", 
    "USER : کندی خوش است BOT : بلی! <EOS>", 
    "USER : بستنی سرد است BOT : خیلی سرد! <EOS", '<PAD>'
]

words = set()
for sentence in corpus:
    words.update(sentence.split())
toyds = {word : idx for idx,word in enumerate(sorted(words))}
dmodel = 256
num_epoches = 50
learning_rate = 1e-3
device = t.device('cuda' if t.cuda.is_available() else 'cpu')
batch_size = 4

class Embeddingclass(n.Module):
    def __init__(self, d_model, word_length):
        super().__init__()

        self.dmodel = d_model
        self.embedding = n.Embedding(word_length, d_model)

    def forward(self, x):
        return self.embedding(x) *math.sqrt(self.dmodel)
    
trial = Embeddingclass( d_model= 4, word_length= len(toyds))

def tokening(dataset,sentences : str = None, corpusds = None):
    if corpusds is not None:
        sentense = []
        for sentence in corpusds:
            words = []
            for word in sentence.split():
                words.append(dataset[word])
            wor = t.tensor(words)
            sentense.append(wor)
        result = sentense

    else:
        words = []
        for word in sentences.split():
            words.append(dataset[word])
        result = t.tensor(words)

    return result

class PositionalEncoding(n.Module):
    def __init__(self, d_model, seq_length = 512):
        super().__init__()
        self.dmodel = d_model
        pt = t.arange(seq_length, dtype= t.float32).unsqueeze(1)
        dindex = t.arange(d_model, dtype = t.float32)
        pe = t.zeros(seq_length, d_model, dtype= t.float32)
        divterm = 10000.0 ** ((2 * (dindex // 2)) / d_model)

        pe[:, 0::2] = t.sin(pt / divterm[0::2])
        pe[:, 1::2] = t.cos(pt / divterm[1::2])

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1) , :]

'''        
class Attention(n.Module):
    def __init__(self, dmodel, dk = 64):
        super().__init__()

        self.dmodel = dmodel
        self.dk = dk
        self.wq = n.Linear(dmodel, dk)
        self.wk = n.Linear(dmodel, dk)
        self.wv = n.Linear(dmodel, dk)
        self.linearout = n.Linear(dk, dmodel)

    def forward(self, x):
        q = self.wq(x)
        k =self.wk(x)
        v =self.wv(x)

        score = t.matmul(q, k.transpose(-2, -1))
        score = score / math.sqrt(self.dk)

        weights = t.softmax(score, dim = -1)

        content = t.matmul(weights, v)

        output = self.linearout(content)

        return output, v
'''

class CasualMultiHeadAttention(n.Module):
    def __init__(self, dmodel,headnum = 8, dk = 64):
        super().__init__()
        self.dmodel = dmodel
        self.dk = dk
        self.headnum = headnum
        self.linearout = n.Linear(dk * headnum, dmodel)
        self.wq = n.Linear(dmodel, dk * headnum)
        self.wk = n.Linear(dmodel, dk * headnum)
        self.wv = n.Linear(dmodel, dk * headnum)
    
    def mask_generator(self, seqlen):
        mask = t.triu(t.ones(seqlen, seqlen), diagonal = 1).bool()
        return mask

    def forward(self, x, mask = None):
        q = self.wq(x)
        v = self.wv(x)
        k = self.wk(x)
        
        a, b, _ = x.shape

        q = q.reshape(a,b,self.headnum,self. dk)
        k = k.reshape(a,b,self.headnum,self. dk)
        v = v.reshape(a,b,self.headnum,self. dk)

        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)
        
        score = t.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.dk)

        if mask is None:
            mask = self.mask_generator(b)
            mask = mask.to(x.device)
        score = score.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        weight = t.softmax(score, dim = -1)
        content = t.matmul(weight, v)

        content = content.transpose(1,2).contiguous().view(a, b, self.headnum * self.dk)

        result = self.linearout(content)

        return result, weight

class FeedForward(n.Module):
    def __init__(self, dmodel, dff = None, dropoutp = 0.1):
        super().__init__()
        if dff is None:
            dff = dmodel * 4
        self.dropout = dropoutp
        self.model = n.Sequential(n.Linear(dmodel, dff), n.GELU(), n.Dropout(p=dropoutp), n.Linear(dff, dmodel), n.Dropout(p=dropoutp))
    
    def forward(self, x):
        return self.model(x)
    
class LayerNormalization(n.Module):
    def __init__(self, dmodel, eps = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = n.Parameter(t.ones(dmodel))
        self.beta = n.Parameter(t.zeros(dmodel))

    def forward(self, x):
        mean = t.mean(x, dim= -1, keepdim= True)
        std = t.std(x, dim= -1, keepdim= True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

invertedds = {}

for word, idx in toyds.items():
    invertedds[idx] = word

class TransformerLM(n.Module):
    def __init__(self, vocab_size,dmodel = dmodel):
        super().__init__()
        
        self.embedding = Embeddingclass( d_model= dmodel, word_length= vocab_size)
        self.PositionalEncoding = PositionalEncoding(seq_length= 512, d_model= dmodel)
        self.CMHA = CasualMultiHeadAttention(dmodel= dmodel)
        self.ff = FeedForward(dmodel= dmodel)
        self.layernorm = LayerNormalization(dmodel= dmodel)
        self.out = n.Linear(dmodel, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.PositionalEncoding(x)

        attention_output, _ = self.CMHA(x)
        x = self.layernorm(x + attention_output)

        ffn_output = self.ff(x)
        x = self.layernorm(x + ffn_output)

        logits = self.out(x)
        return logits
    
    def get_next_token_probs(self, input_seq):
        result = self.forward(input_seq)
        nexttlogit = result[:, -1, :]
        nexttprobs = f.softmax(nexttlogit, dim=-1)
        return nexttprobs
    
    def sample_next_token(self, inputseq):
        probabilities = self.get_next_token_probs(inputseq)
        distribution = dist.Categorical(probabilities)
        sampled = distribution.sample()
        logprob = distribution.log_prob(sampled)
        return sampled, logprob



sentence = 'I love'

def chat(ds, inverteddset, sentence):
    model = TransformerLM(vocab_size= len(ds))

    token = tokening(ds, sentence).unsqueeze(0)

    logits = model(token)

    last_logits = logits[0, -1]

    prediction_tensor = t.argmax(last_logits).item()

    predict = inverteddset[prediction_tensor]

    return predict

#for CPU optimization
def corpus_determining(corpus):
    tokenized_corpus = tokening(toyds, corpusds=corpus)
    padded_corpus = t.nn.utils.rnn.pad_sequence(tokenized_corpus, batch_first= True, padding_value=toyds['<PAD>'])
    dataset = t.utils.data.TensorDataset(padded_corpus)
    dataloader = t.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle= True)
    vocabsize = len(toyds)
    model = TransformerLM(vocab_size=vocabsize, dmodel=dmodel).to(device)
    criterionmodel = n.CrossEntropyLoss(ignore_index=toyds['<PAD>'])
    optimizermodel = t.optim.Adam(model.parameters(), lr= learning_rate)

    if os.path.exists("transformer_checkpoint.pth"):
        checkpoint = t.load("transformer_checkpoint.pth", map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizermodel.load_state_dict(checkpoint["optimizer_state"])
        print("✅ Loaded checkpoint from transformer_checkpoint.pth")
    else:
        print("⚠️ No checkpoint found, starting fresh")


    return model, criterionmodel, optimizermodel, dataloader, vocabsize

model, criterionmodel , optimizermodel, dataloader, vocabsize = corpus_determining(corpus)
max_len = len(toyds)

for epoch in range(num_epoches):
    model.train()
    total_loss = 0
    for batch in dataloader:
        data = batch[0].to(device)
        inputs = data[:, :-1].contiguous()
        targets = data[:, 1:].contiguous()
        logits = model(inputs)
        loss = criterionmodel(logits.view(-1, vocabsize), targets.view(-1))
        optimizermodel.zero_grad()
        loss.backward()
        optimizermodel.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")


    # ✅ save after every epoch
    t.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizermodel.state_dict(),
        "vocab": toyds
    }, "transformer_checkpoint.pth")
    print("💾 Checkpoint saved")

'''
def text_generation(model, prompt, ds, inverted, max_lenght = max_len):
    model.eval()
    generated = tokening(dataset=ds, sentences=prompt).unsqueeze(0).to(device)
    log_probs = []

    for _ in range(max_lenght):
        logit = model(generated)  
        next_token_logit = logit[0, -1]
        probs = t.softmax(next_token_logit, dim=-1)
        next_token = t.multinomial(probs, num_samples=1)
        log_prob = t.log(probs[next_token])
        log_probs.append(log_prob)

        generated = t.cat([generated, next_token.unsqueeze(0)], dim= 1)
        if next_token.item() == ds['<PAD>']:
            break

    result = [inverted[token.item()] for token in generated[0]]
    return ' '.join(result), log_probs
'''

def rLHF(sequence : str):
    print(f"Generated sequence: {sequence}")
    try:
        feedback = float(input('Feedback on the sequence (float): '))
        return feedback
    except ValueError:
        print('Please enter a valid number.')
        return 0.0
    
def chat_generation( prompt , model = model, ds = toyds, inverted = invertedds, max_tokens=15):
    model.eval()
    generated = tokening(dataset=ds, sentences=prompt).unsqueeze(0).to(device)
    reply_tokens = []

    for _ in range(max_tokens):
        logits = model(generated)
        next_token_logits = logits[0, -1]
        probs = t.softmax(next_token_logits, dim=-1)
        next_token = t.multinomial(probs, num_samples=1)

        word = inverted[next_token.item()]
        if word in ["USER:", "<PAD>" "<EOS>"]:   # stop when new user turn starts
            break

        reply_tokens.append(word)
        generated = t.cat([generated, next_token.unsqueeze(0)], dim=1)

    return " ".join(reply_tokens), probs

def training(prompt,ds = toyds , model = model, optimizer = optimizermodel, iteration = 12, inverted = invertedds):
    model.train()
    for i in range(iteration):
        tokens = tokening(dataset=ds, sentences= prompt)
        tokens = tokens.unsqueeze(0).to(device)
        sampled, prob = chat_generation(prompt)
        reward = rLHF(sampled)
        loss = -(prob * reward).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Iteration {i+1}: Word '{sampled}' | Reward: {reward} | Loss: {loss.item():.4f}")

extracorpus = []

def dynamic_learning(prompt : str, words : dict, corpus : list = extracorpus):
    for i in  '.,-_':
        if i in prompt:
            prompt = prompt.replace(i, ' ')
    
    sentence = prompt.split(' ')

    wordindex = list(words.values())[-1]
    for word in sentence:
        if word not in words.keys():
            words[word] = wordindex + 1
            wordindex += 1  

    prompt = f'USER : {prompt} BOT :'
    response , _ = chat_generation(prompt=prompt)
    prompt = f'USER : {prompt} BOT : {response}'
    corpus.append(prompt)

جمله = "من ترا دوست دارم"

prot = f'USER : {جمله} BOT :'
print(training(prot, model=model, ds=toyds,inverted= invertedds))
dynamic_learning(prompt=جمله, words = toyds, corpus=corpus)

t.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizermodel.state_dict(),
        "vocab": toyds
    }, "transformer_checkpoint.pth")
print("💾 Checkpoint saved")

