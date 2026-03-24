from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
from transformers import DynamicCache
from peft import LoraConfig, get_peft_model

#import numpy as np
#from scipy.integrate import solve_ivp

import pdb
import json

def triangular(x, a, b, c):
    # a < b < c (can be tensors or scalars, broadcastable with x)
    left  = (x - a) / (b - a)
    right = (c - x) / (c - b)
    return torch.clamp(torch.minimum(left, right), min=0.0, max=1.0)

def ramp_right(x, a, b):
    y = (x - a) / (b - a)
    return torch.clamp(y, min=0.0, max=1.0)

def ramp_left(x, a, b):
    y = - (x - b) / (b - a)
    return torch.clamp(y, min=0.0, max=1.0)

def neg_membership(x, c_neg=-1.0, w=1.0):
    # Triangle peaked at c_neg, support [c_neg - w, c_neg + w]
    a = c_neg
    b = c_neg + w
    return ramp_left(x, a, b)

def zero_membership(x, c_zero=0.0, w=1.0):
    # Triangle peaked at c_zero, support [c_zero - w, c_zero + w]
    a = c_zero - w
    b = c_zero
    c = c_zero + w
    return triangular(x, a, b, c)

def pos_membership(x, c_pos=0.0, w=1.0):
    # Triangle peaked at c_pos, support [c_pos - w, c_pos + w]
    a = c_pos
    b = c_pos + w
    return ramp_right(x, a, b)

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
msg_end_len = 3
sequence_len = 10
batch_size = 16
Nemb = llm.config.hidden_size
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

input_terms_tok = tokenizer(input_terms,add_special_tokens=False)
input_terms_tok = torch.tensor(input_terms_tok['input_ids'],dtype=torch.long).repeat(1,sequence_len)
input_terms_vec0 = llm.model.embed_tokens(input_terms_tok)

#bath =  [pos,pos,...,zer,zer,...,neg,neg,...]
input_terms_vec_batch = torch.cat([
    input_terms_vec0[0,:,:].unsqueeze(0).repeat(batch_size,1,1),
    input_terms_vec0[1,:,:].unsqueeze(0).repeat(batch_size,1,1),
    input_terms_vec0[2,:,:].unsqueeze(0).repeat(batch_size,1,1)],dim=0)
input_terms_tok_batch = torch.cat([
    input_terms_tok[0,:].unsqueeze(0).repeat(batch_size,1),
    input_terms_tok[1,:].unsqueeze(0).repeat(batch_size,1),
    input_terms_tok[2,:].unsqueeze(0).repeat(batch_size,1)],dim=0)

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
llm.train()

y_calc = []
u_calc = []
L_calc = []
yt = torch.zeros((batch_size,sequence_len)).cuda()
ut = torch.zeros((batch_size,sequence_len)).cuda()

params = {"Ts":0.01,"T":-2.0,"k":0.5}
state = (torch.rand(batch_size,1).cuda() - 0.5)/5.0

for t in range(0,1000):
    input_terms_tok_batch = input_terms_tok_batch.detach()
    input_terms_vec_batch = input_terms_vec_batch.detach()
    yt = yt.detach()
    ut = ut.detach()
    state = state.detach()
    
    input_mf = torch.cat([
        pos_membership(yt),
        zero_membership(yt),
        neg_membership(yt)],dim=0).unsqueeze(2).repeat(1,1,Nemb)
    input_terms_vec = input_mf.half().cuda()*input_terms_vec_batch

    output_mf = torch.cat([
        pos_membership(ut),
        zero_membership(ut),
        neg_membership(ut)],dim=0).unsqueeze(2).repeat(1,1,Nemb)
    output_terms_vec = output_mf.half().cuda()*input_terms_vec_batch

    combined_embeds = torch.cat([
        vectorized_messages1[:,:-msg_end_len,:], 
        input_terms_vec, 
        vectorized_messages1[:,-msg_end_len:,:],
        vectorized_messages2,
        output_terms_vec], dim=1).detach()
    
    outputs = llm(inputs_embeds=combined_embeds,use_cache=False)
    output_logits = outputs.logits[:,-1]
    probs = torch.functional.F.softmax(output_logits,dim=-1)
    new_mf = probs.gather(1,input_terms_tok_batch[:,0].unsqueeze(1).cuda())

    edges_val = torch.cat([
        torch.ones((batch_size,1)),
        torch.zeros((batch_size,1)),
        -torch.ones((batch_size,1))],dim=0).cuda()
    u_new = new_mf*edges_val
    p_block = u_new[0:batch_size,:]
    z_block = u_new[batch_size:2*batch_size,:]
    n_block = u_new[2*batch_size:3*batch_size,:]
    u_new = torch.cat([p_block,z_block,n_block],dim=1).sum(dim=1).unsqueeze(1)
    state += -params["Ts"]/params["T"]*state+params["k"]*params["Ts"]/params["T"]*u_new
    loss_ce = torch.nn.CrossEntropyLoss()(output_logits,input_terms_tok_batch[:,0].cuda())
    optimizer.zero_grad()
    total_loss = 1000.0*(state*state).max()+loss_ce
    total_loss.backward()
    optimizer.step()
    print(f't={t} u={u_new.mean().item():.4f}, y={state.mean().item():.4f}, Loss={total_loss.item():.3f}')
    y_calc.append(state.view(-1).tolist())
    u_calc.append(u_new.view(-1).tolist())
    L_calc.append(total_loss.item())
    yt = torch.hstack([yt[:,1:],state])
    ut = torch.hstack([ut[:,1:],u_new])

dat = {"u":u_calc,"y":y_calc,"loss":L_calc}
f = open("sim_result.json","w")
f.write(json.dumps(dat))
f.close()
pdb.set_trace()
