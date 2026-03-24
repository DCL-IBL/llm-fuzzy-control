from fuzzylogic.classes import Domain
from fuzzylogic.functions import R, S, gauss, constant
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
from transformers import DynamicCache
from peft import LoraConfig, get_peft_model

#import numpy as np
#from scipy.integrate import solve_ivp

import pdb
import json

f = open("rand_sim.json")
dat = json.loads(f.read())
f.close()
Hy = []
Hu = []
sequence_len = 10
for ind in range(0,len(dat['y'])-sequence_len):
    Hy.append(dat['u'][ind:ind+sequence_len])
    Hu.append(dat['y'][ind:ind+sequence_len])

y_domain = Domain("output", -1, 1, res=0.01)
y_domain.positive = R(0,1)
y_domain.negative = S(-1,0)
y_domain.zero = gauss(0, 100, c_m=1)

u_domain = Domain("input", -1, 1, res=0.01)
u_domain.positive = R(0,1)
u_domain.negative = S(-1,0)
u_domain.zero = gauss(0, 100, c_m=1)

llm_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
max_memory_map = {0: "5GiB", 1: "5GiB",2: "5GiB",}

llm = AutoModelForCausalLM.from_pretrained(
    llm_name,
    device_map="auto",
    max_memory=max_memory_map,
    dtype=torch.float16,
    attn_implementation="eager"
)
tokenizer = AutoTokenizer.from_pretrained(llm_name)

input_terms = ['positive','zero','negative']
output_terms = ['positive','zero','negative']
batch_size = 16
input_alternatives = len(input_terms)*batch_size

messages1 = [{"role":"system","content":"You are an assistant in making control decision."},
           {"role":"user","content":"Output tracking error is "}]
messages2 = [{"role":"assistant","content":"Control action is "}]

tokenized_messages1 = tokenizer.apply_chat_template(messages1, tokenize=True, 
                                                   add_generation_prompt=False,
                                                   continue_final_message=False,
                                                   return_tensors="pt")
vectorized_messages1 = llm.model.embed_tokens(tokenized_messages1[0]).unsqueeze(0)
vectorized_messages1 = vectorized_messages1.repeat(input_alternatives,1,1)

tokenized_messages2 = tokenizer.apply_chat_template(messages2, tokenize=True, 
                                                   add_generation_prompt=False, 
                                                   continue_final_message=True,
                                                   return_tensors="pt")
vectorized_messages2 = llm.model.embed_tokens(tokenized_messages2[0]).unsqueeze(0)
vectorized_messages2 = vectorized_messages2.repeat(input_alternatives,1,1)

msg_end_len = 3

input_terms_tok = tokenizer(input_terms,add_special_tokens=False)
input_terms_tok = torch.tensor(input_terms_tok['input_ids'],dtype=torch.long).repeat(1,sequence_len)
input_terms_vec0 = llm.model.embed_tokens(input_terms_tok)

output_terms_tok = tokenizer(output_terms,add_special_tokens=False)
output_terms_tok = torch.tensor(output_terms_tok['input_ids'],dtype=torch.long).repeat(batch_size,1)

lora_config = LoraConfig(
    r=16,                      # Low rank
    lora_alpha=32,             # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Layers to apply LoRA
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
llm = get_peft_model(llm, lora_config)
llm.print_trainable_parameters()

optimizer = torch.optim.AdamW(llm.parameters(), lr=1e-5)
num_epochs = 100
llm.train()

input_terms_vec_batch = input_terms_vec0.repeat(batch_size,1,1)
input_terms_tok_batch = input_terms_tok.repeat(batch_size,1)

batch_count = int(len(Hy)/batch_size)
for epoch in range(0,num_epochs):
    for batch in range(0,batch_count):
        batch_start = batch*batch_count
        #yt = [0.7,0.5,0.6,1.4,0,-1,7,8,10,0.5]
        #ut = [0.2,0.3,0.1,0.4,0,1,-7,-8,1,5]
        input_terms_vec = torch.tensor([]).half().cuda()
        target_mf = torch.tensor([])
        for ind in range(0,batch_size):
            yt = Hy[batch_start + ind]
            input_mf = [[y_domain.positive(y), y_domain.zero(y), y_domain.negative(y)] for y in yt]
            mf_vec_scale = torch.tensor(input_mf).permute(1,0).unsqueeze(2).repeat(1,1,input_terms_vec0.shape[2])
            input_terms_vec_part = mf_vec_scale.half().cuda()*input_terms_vec0
            input_terms_vec = torch.cat([input_terms_vec,input_terms_vec_part],dim=0)

            ut = Hu[ind]
            target_mf_part = [[y_domain.positive(y), y_domain.zero(y), y_domain.negative(y)] for y in ut]
            target_mf_part = torch.tensor(target_mf_part).permute(1,0)
            target_mf = torch.cat([target_mf,target_mf_part],dim=0)

        combined_embeds0 = torch.cat([
            vectorized_messages1[:,:-msg_end_len,:], 
            input_terms_vec, 
            vectorized_messages1[:,-msg_end_len:,:],
            vectorized_messages2],dim=1)

#for epoch in range(0,num_epochs):
        output_logits = torch.tensor([]).cuda()
        output_mfs = torch.tensor([]).cuda()
        combined_embeds = combined_embeds0.detach()
        input_terms_vec_batch = input_terms_vec_batch.detach()
        input_terms_tok_batch = input_terms_tok_batch.detach()
        cache = DynamicCache()
    #llm.eval()
    #pdb.set_trace()

        ut = []
        optimizer.zero_grad()
        for ind in range(0,sequence_len):    
            #with torch.no_grad():
            outputs = llm(inputs_embeds=combined_embeds,past_key_values=cache,use_cache=True)
            logits = outputs.logits[:,-1]
            probs = torch.functional.F.softmax(logits,dim=-1)
            output_mf = probs.gather(1,output_terms_tok.cuda())
            output_mfs = torch.cat([output_mfs,output_mf],dim=1)
            output_logits = torch.cat([output_logits,logits.unsqueeze(1)],dim=1)
            mf_vec_scale = output_mf.unsqueeze(2).repeat(1,1,input_terms_vec_batch.shape[2])
            combined_embeds = mf_vec_scale.half().cuda()*input_terms_vec_batch[:,0,:].unsqueeze(1)

        #pdb.set_trace()
        loss_ce = torch.nn.CrossEntropyLoss()(output_logits.permute((0,2,1)), input_terms_tok_batch.cuda())
        loss = torch.nn.MSELoss()(output_mfs.half(),target_mf.half().cuda())
        (loss_ce+100.0*loss).backward()
        optimizer.step()
        print(f'Epoch: {epoch}, Batch: {batch}, Loss MSE: {loss.item()}, Loss CE: {loss_ce.item()}')
        #cache = tuple(layer.detach() for layer in outputs.past_key_values) if outputs.past_key_values else None
        #for ind,L in enumerate(cache.layers):
        #    cache.layers[ind].values.detach()

llm = llm.merge_and_unload()

save_directory = 'fuzzy-llm'
llm.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

pdb.set_trace()
    
    #y_domain.c1 = constant(output_mf[0])
    #y_domain.c2 = constant(output_mf[1])
    #y_domain.c3 = constant(output_mf[2])

    #result = y_domain.c1 & y_domain.positive | y_domain.c2 & y_domain.zero | y_domain.c3 & y_domain.negative
    #rng = np.arange(-1,1,0.01)
    #avg = 0
    #sum_p = 0
    #for x in rng:
    #    p = result(x)
    #    sum_p += p
    #    avg += p*x
    #avg /= sum_p
    #ut.append(avg.item())
    #mf_vec_scale = torch.tensor([output_mf]).permute(1,0).unsqueeze(2).repeat(1,1,input_terms_vec0.shape[2])
    #mf_vec_scale = output_mf.unsqueeze(2).repeat(1,1,input_terms_vec0.shape[2])
    #combined_embeds = mf_vec_scale.half().cuda()*input_terms_vec0[:,0,:].unsqueeze(1)

