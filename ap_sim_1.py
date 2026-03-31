from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
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

def hypo_glucose(x):
    return ramp_left(x,90/18,105/18) # 70 - 105 mg/dl

def target_glucose(x):
    return triangular(x,90/18,105/18,180/18) # 70 - 180 mg/dl

def hyper_glucose(x):
    return ramp_right(x, 150/18, 300/18) # 105 - 180 mg/dl

def zero_dose(x):
    return ramp_left(x,0,0.01) # mU/kg

def low_dose(x):
    return triangular(x, 0.0, 0.01, 0.1) # mU/kg

def high_dose(x):
    return ramp_right(x, 0.05, 0.2) # mU/kg

def bolus_dose(x):
    return ramp_right(x, 0.1, 1.5) # mU/kg

x_vals = torch.linspace(0,0.2,50).cuda()
mf_vals = torch.vstack([zero_dose(x_vals).unsqueeze(0),low_dose(x_vals).unsqueeze(0),high_dose(x_vals).unsqueeze(0)])

llm_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
max_memory_map = {0: "5GiB", 1: "5GiB",2: "5GiB",}

llm = AutoModelForCausalLM.from_pretrained(
    llm_name,
    device_map="auto",
    #max_memory=max_memory_map,
    #dtype=torch.float16,
    #attn_implementation="eager"
)
tokenizer = AutoTokenizer.from_pretrained(llm_name)

input_terms = ['hypoglycemia','in range','hyperglycemia']
output_terms = ['zero','low','high']
msg_end_len = 3
sequence_len = 10
batch_size = 16
Nemb = llm.config.hidden_size
input_alternatives = batch_size #len(input_terms)*batch_size

messages1 = [{"role":"system","content":"You are an automatic insulin delivery advisor for type 1 diabetes. "},
           {"role":"user","content":"Historical glucose: "}]
messages2 = [{"role":"assistant","content":"Next insulin dose: "}]
messages3 = ". Historical insulin dosages: "

tokenized_messages1 = tokenizer.apply_chat_template(messages1, tokenize=True, 
                                                   add_generation_prompt=False,
                                                   continue_final_message=False,
                                                   return_tensors="pt")
vectorized_messages1 = llm.model.embed_tokens(tokenized_messages1['input_ids'][0].cuda()).unsqueeze(0)
vectorized_messages1 = vectorized_messages1.repeat(input_alternatives,1,1)

tokenized_messages2 = tokenizer.apply_chat_template(messages2, tokenize=True, 
                                                   add_generation_prompt=False, 
                                                   continue_final_message=True,
                                                   return_tensors="pt")
vectorized_messages2 = llm.model.embed_tokens(tokenized_messages2['input_ids'][0].cuda()).unsqueeze(0)
vectorized_messages2 = vectorized_messages2.repeat(input_alternatives,1,1)

tokenized_messages3 = tokenizer(messages3, add_special_tokens=False, return_tensors="pt")
vectorized_messages3 = llm.model.embed_tokens(tokenized_messages3['input_ids'][0].cuda()).unsqueeze(0)
vectorized_messages3 = vectorized_messages3.repeat(input_alternatives,1,1)

input_terms_tok = tokenizer(input_terms, add_special_tokens=False, return_tensors="pt", padding=True)
input_toks_per_term = input_terms_tok['input_ids'].shape[1]
input_terms_tok = input_terms_tok['input_ids'].repeat(1,sequence_len)
input_terms_vec0 = llm.model.embed_tokens(input_terms_tok.cuda())

output_terms_tok = tokenizer(output_terms, add_special_tokens=False, return_tensors="pt", padding=True)
output_toks_per_term = output_terms_tok['input_ids'].shape[1]
output_terms_tok = output_terms_tok['input_ids'].repeat(1,sequence_len)
output_terms_vec0 = llm.model.embed_tokens(output_terms_tok.cuda())

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

optimizer = torch.optim.AdamW(llm.parameters(), lr=1e-5) # 1e-5
llm.train()

y_calc = []
u_calc = []
L_calc = []
loss_ce_calc = []
zero_dose_p_calc = []
low_dose_p_calc = []
high_dose_p_calc = []

#BW = torch.round(90.0+10*torch.randn((batch_size,))).cuda()
BW = torch.tensor([70., 75., 80., 85., 90., 100., 105., 110., 115., 120., 125., 130., 135., 140., 145., 150.]).cuda()
Nstate = 8
state = torch.tensor([50,50,0,0,0,0,0,0]).cuda().repeat(batch_size,1)
#torch.rand(batch_size,Nstate).cuda()

meals = [(8*60,250), (12*60,250), (19*60,250)] # ingestion time, min/ Digested CHO mmol
# uin - mU/min (batch_size,)
def ap_model(t,state,uin):
    VG = 0.16*BW # Glucose distribution volume, L/kg*kg = L
    VI = 0.12*BW # Insulin distribution volume, L/kg*kg = L
    F01 = 0.0097*BW # Non-insulin glucose flux, mmol/min/kg*kg = mmol/min
    k12 = 0.0066 # Transfer rate from interstitial to plasma, 1/min
    ka1 = 0.006 # 1/min
    ka2 = 0.06 # 1/min
    ka3 = 0.03 # 1/min
    SIT = 51.2e-4 # 1/min/(mU*1/L) = L/min/mU
    SID = 8.2e-4 # 
    SIE = 520e-4 #
    kb1 = SIT*ka1
    kb2 = SID*ka2
    kb3 = SIE*ka3
    EGP0 = 0.0161*BW # Endogenous glucose production extrapolated to zero insulin concentration, mmol/kg/min*kg = mmol/min

    AG = 0.8 # Carbohydrate bioavailability
    tmaxG = 40 # Time to max carbohydrate, min

    tmaxI = 55 # Time-to-maximum of absorption of subcutaneously injected short-acting insulin, min
    ke = 0.138 # Insulin elimination from plasma, 1/min
    
    Q1 = state[:,0] # glucose masses in plasma, mmol
    Q2 = state[:,1] # glucose masses in interstitial, mmol
    x1 = state[:,2] # Insulin effect on glucose transport and distribution between plasma and interstitial, 1/min
    x2 = state[:,3] # Insulin effect on glucose transport and distribution between plasma and interstitial, 1/min
    x3 = state[:,4] # Insulin effect on glucose transport and distribution between plasma and interstitial, 1/min
    S1 = state[:,5] # Insulin sensitivity in accessible, mU
    S2 = state[:,6] # Insulin sensitivity in nonaccesible, mU
    I = state[:,7] # Insulin in plasma, mU
    
    G = Q1/VG # mmol/L
    F01c = F01*torch.ones((batch_size,)).cuda() # mmol/min
    sel = G>4.5
    if sel.sum().item() > 0:
        F01c[sel] = F01[sel]*G[sel]/4.5 # mmol/min
    FR = torch.zeros((batch_size,)).cuda()
    sel = G>=9
    if sel.sum().item() > 0:
        FR[sel] = 0.003*(G[sel]-9)*VG[sel]

    tod = t % (24*60)
    UG = 0
    ub = 0*BW
    for meal in meals:
        tm = meal[0]
        DG = meal[1]
        if tod>tm:
            tau = tod - tm
            UG = UG + DG*AG*tau*torch.exp(-tau/tmaxG)/(tmaxG**2) # mmol*(-)*min/min^2 = mmol/min
        

    Q1dot = EGP0*(1.0-x3) + UG - FR - (x1+F01c/(VG*G))*Q1 + k12*Q2
    Q2dot = x1*Q1 - (k12+x2)*Q2

    S1dot = uin - S1/tmaxI
    S2dot = S1/tmaxI - S2/tmaxI

    UI = S2/tmaxI

    Idot = UI/VI - ke*I

    x1dot = -ka1*x1+kb1*I
    x2dot = -ka2*x2+kb2*I
    x3dot = -ka3*x3+kb3*I

    dstate = torch.hstack([
        Q1dot.unsqueeze(1),
        Q2dot.unsqueeze(1),
        x1dot.unsqueeze(1),
        x2dot.unsqueeze(1),
        x3dot.unsqueeze(1),
        S1dot.unsqueeze(1),
        S2dot.unsqueeze(1),
        Idot.unsqueeze(1)])
    return (dstate,G)

u_new=0.0*torch.ones((batch_size,)).cuda()
Ts = 5
'''
for t in range(0,24*60*3):
    tt = torch.tensor(t*Ts).cuda()
    (k1,G) = ap_model(tt,state,u_new)
    state = state + k1*Ts
    mf_block_norm = torch.hstack([
        hypo_glucose(G).unsqueeze(1),
        target_glucose(G).unsqueeze(1),
        hyper_glucose(G).unsqueeze(1)
    ]).half()
    #pdb.set_trace()
    agg_mu = torch.einsum('br,rx->bx', mf_block_norm, mf_vals)
    u_new = BW*(torch.trapz(agg_mu * x_vals.unsqueeze(0), x_vals, dim=1) / (torch.trapz(agg_mu, x_vals, dim=1) + 1e-8))
    print(f't={t} u={round(u_new.mean().item())}, y={round(G.mean().item()*18)}')
    y_calc.append(G.view(-1).tolist())
    u_calc.append(u_new.view(-1).tolist())

dat = {"u":u_calc,"y":y_calc}
f = open("sim_result_1.json","w")
f.write(json.dumps(dat))
f.close()
pdb.set_trace()
'''

num_training_steps = int(24*60*10/Ts)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

yt = (state[:,0]/(0.16*BW)).unsqueeze(1).repeat(1,sequence_len*input_toks_per_term) #torch.zeros((batch_size,sequence_len)).cuda()
ut = torch.zeros((batch_size,sequence_len*output_toks_per_term)).cuda()
for t in range(0,num_training_steps):
    tt = torch.tensor(t*Ts).cuda()
    input_terms_vec0 = input_terms_vec0.detach()
    output_terms_vec0 = output_terms_vec0.detach()
    yt = yt.detach()
    ut = ut.detach()
    state = state.detach()

    input_terms_vec = hypo_glucose(yt).unsqueeze(2).repeat(1,1,Nemb)*input_terms_vec0[0,:,:].unsqueeze(0).repeat(batch_size,1,1) + target_glucose(yt).unsqueeze(2).repeat(1,1,Nemb)*input_terms_vec0[1,:,:].unsqueeze(0).repeat(batch_size,1,1) + hyper_glucose(yt).unsqueeze(2).repeat(1,1,Nemb)*input_terms_vec0[2,:,:].unsqueeze(0).repeat(batch_size,1,1)

    output_terms_vec = zero_dose(ut).unsqueeze(2).repeat(1,1,Nemb)*output_terms_vec0[0,:,:].unsqueeze(0).repeat(batch_size,1,1) + low_dose(ut).unsqueeze(2).repeat(1,1,Nemb)*output_terms_vec0[1,:,:].unsqueeze(0).repeat(batch_size,1,1) + high_dose(ut).unsqueeze(2).repeat(1,1,Nemb)*output_terms_vec0[2,:,:].unsqueeze(0).repeat(batch_size,1,1)
    
    combined_embeds = torch.cat([
        vectorized_messages1[:,:-msg_end_len,:], 
        input_terms_vec, 
        #vectorized_messages3,
        #output_terms_vec,
        vectorized_messages2
    ], dim=1).detach().bfloat16()
    
    outputs = llm(inputs_embeds=combined_embeds,use_cache=False)
    output_logits = outputs.logits[:,-1]

    #edges_val = torch.cat([
    #    (0.01*torch.ones((batch_size,)).cuda()*BW).unsqueeze(1),
    #    (0.05*torch.ones((batch_size,)).cuda()*BW).unsqueeze(1),
    #    (0.5*torch.ones((batch_size,)).cuda()*BW).unsqueeze(1)],dim=1)
    #u_new = new_mf*edges_val
    zero_dose_p = output_logits.gather(1,output_terms_tok[0,0].repeat(batch_size,1).cuda())
    low_dose_p = output_logits.gather(1,output_terms_tok[1,0].repeat(batch_size,1).cuda())
    high_dose_p = output_logits.gather(1,output_terms_tok[2,0].repeat(batch_size,1).cuda())
    
    #mf_block = torch.functional.F.softmax(torch.cat([zero_dose_p,low_dose_p,high_dose_p,bolus_dose_p],dim=1),dim=-1)
    #mf_block = torch.clamp(torch.cat([zero_dose_p,low_dose_p,high_dose_p],dim=1),min=0)
    #mf_block_norm = mf_block/(mf_block.sum(dim=-1).unsqueeze(1).repeat(1,3))
    mf_block_norm = torch.functional.F.softmax(torch.cat([zero_dose_p,low_dose_p,high_dose_p],dim=1),dim=-1)
    agg_mu = torch.einsum('br,rx->bx', mf_block_norm, mf_vals.bfloat16())
    u_new = BW*(torch.trapz(agg_mu * x_vals.unsqueeze(0), x_vals, dim=1) / (torch.trapz(agg_mu, x_vals, dim=1) + 1e-8))
    #inds = mf_block.argmax(dim=-1)
    #u_new = (edges_val.gather(1,inds.unsqueeze(1))*mf_block_norm.gather(1,inds.unsqueeze(1))).squeeze(1)
    #u_new = (edges_val*mf_block_norm).sum(dim=-1)
    
    #u_new = (mf_block*edges_val).sum(dim=1)
    
    #Fuzzy rules
    #G = state[:,0]/(0.16*BW)
    #in_class = torch.hstack([hypo_glucose(G).unsqueeze(1),target_glucose(G).unsqueeze(1),hyper_glucose(G).unsqueeze(1)]).argmax(dim=-1)
    #out_class = (output_terms_tok[:,0].repeat(16,1).cuda()).gather(1,in_class.unsqueeze(1)).squeeze(1)
    yt_class = torch.hstack([
        hypo_glucose(yt).mean(dim=-1).unsqueeze(1),
        target_glucose(yt).mean(dim=-1).unsqueeze(1),
        hyper_glucose(yt).mean(dim=-1).unsqueeze(1)
    ]).argmax(dim=-1)
    
    ut_class = torch.hstack([
        zero_dose(ut).mean(dim=-1).unsqueeze(1),
        low_dose(ut).mean(dim=-1).unsqueeze(1),
        high_dose(ut).mean(dim=-1).unsqueeze(1)
    ]).argmax(dim=-1)
    
    #for k in range(0,batch_size):
    #    if ut_class[k] == 2:
    #        yt_class[k] = 0

    out_class = (output_terms_tok[:,0].repeat(16,1).cuda()).gather(1,yt_class.unsqueeze(1)).squeeze(1)

    loss_ce = torch.nn.CrossEntropyLoss()(output_logits,out_class)

    #if loss_ce > 1.0:
    #    u_new = u_new/loss_ce
    #ts = 1.0
    #if t < 2000 and loss_ce > 1.0:
    #    u_new = u_new/loss_ce
    u_per_kg = u_new / BW
    
    (k1,G) = ap_model(tt,state,u_new)
    #(k1,G) = ap_model(tt,state,uin)
    #(k2,G) = ap_model(tt+Ts/2,state+k1*Ts/2,uin)
    #(k3,G) = ap_model(tt+Ts/2,state+k2*Ts/2,uin)
    #(k4,G) = ap_model(tt+Ts,state+k3*Ts,uin)
    #state = state + (k1+2*k2+2*k3+k4)*Ts/6.0
    state = state + k1*Ts

    state1 = state
    for k in range(0,36):
        (k1,G) = ap_model(tt,state1,u_new)
        state1 = state1 + k1*Ts

    optimizer.zero_grad()
    Gout = state[:,0]/(0.16*BW)
    Gout1 = state1[:,0]/(0.16*BW)
    errp = torch.sqrt(torch.clamp(Gout1 - 105/18,min=0))
    errn = torch.clamp(105/18 - Gout1,min=0)
    err1 = Gout1 - 105/18

    #if loss_ce > 1.0:
    #    total_loss = loss_ce
    #else:
    #if t < 2000:
    #    total_loss = loss_ce
    #else:
    total_loss = (err1*err1).max() #+0.1*u_per_kg.mean()
    total_loss.backward()
    #if total_loss > 0.1:
    optimizer.step()
    #lr_scheduler.step()
    print(f't={tt} u={round(u_new.mean().item())}, y={round(Gout.mean().item()*18)}/{round(Gout1.mean().item()*18)}, Loss={total_loss.item():.3f}/{loss_ce.item():.3f} MF: {zero_dose_p.mean().item():.3f}/{low_dose_p.mean().item():.3f}/{high_dose_p.mean().item():.3f}')
    
    y_calc.append(G.view(-1).tolist())
    u_calc.append(u_new.view(-1).tolist())
    L_calc.append(total_loss.item())
    loss_ce_calc.append(loss_ce.item())
    zero_dose_p_calc.append(zero_dose_p.mean().item())
    low_dose_p_calc.append(low_dose_p.mean().item())
    high_dose_p_calc.append(high_dose_p.mean().item())

    yt = torch.hstack([yt[:,input_toks_per_term:],Gout.unsqueeze(1).repeat(1,input_toks_per_term)])
    ut = torch.hstack([ut[:,output_toks_per_term:],u_per_kg.unsqueeze(1)])

dat = {
    "u":u_calc,
    "y":y_calc,
    "loss":L_calc,
    "loss_ce":loss_ce_calc,
    "zero_dose_p":zero_dose_p_calc,
    "low_dose_p":low_dose_p_calc,
    "high_dose_p":high_dose_p_calc
    }
f = open("training_10_day.json","w")
f.write(json.dumps(dat))
f.close()
pdb.set_trace()
#llm = llm.merge_and_unload()

llm.push_to_hub("jkralev/fuzzy-llm")
#tokenizer.push_to_hub("jkralev/fuzzy-llm")