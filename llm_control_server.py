from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np

import pdb
import json

from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

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


x_vals = torch.linspace(0,0.2,50).cuda().half()
mf_vals = torch.vstack([zero_dose(x_vals).unsqueeze(0),low_dose(x_vals).unsqueeze(0),high_dose(x_vals).unsqueeze(0)])

llm_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter_name = "jkralev/fuzzy-llm"

llm = AutoModelForCausalLM.from_pretrained(llm_name,device_map="auto")
llm.load_adapter(adapter_name)
llm = llm.cuda()
tokenizer = AutoTokenizer.from_pretrained(llm_name)
llm.eval()

input_terms = ['hypoglycemia','in range','hyperglycemia']
output_terms = ['zero','low','high']
msg_end_len = 3
sequence_len = 10
batch_size = 1
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

def get_control(yt,ut):
    input_terms_vec = hypo_glucose(yt).unsqueeze(2).repeat(1,1,Nemb)*input_terms_vec0[0,:,:].unsqueeze(0).repeat(batch_size,1,1) + target_glucose(yt).unsqueeze(2).repeat(1,1,Nemb)*input_terms_vec0[1,:,:].unsqueeze(0).repeat(batch_size,1,1) + hyper_glucose(yt).unsqueeze(2).repeat(1,1,Nemb)*input_terms_vec0[2,:,:].unsqueeze(0).repeat(batch_size,1,1)

    output_terms_vec = zero_dose(ut).unsqueeze(2).repeat(1,1,Nemb)*output_terms_vec0[0,:,:].unsqueeze(0).repeat(batch_size,1,1) + low_dose(ut).unsqueeze(2).repeat(1,1,Nemb)*output_terms_vec0[1,:,:].unsqueeze(0).repeat(batch_size,1,1) + high_dose(ut).unsqueeze(2).repeat(1,1,Nemb)*output_terms_vec0[2,:,:].unsqueeze(0).repeat(batch_size,1,1)
    
    combined_embeds = torch.cat([
        vectorized_messages1[:,:-msg_end_len,:], 
        input_terms_vec.half(), 
        #vectorized_messages3,
        #output_terms_vec.half(),
        vectorized_messages2
    ], dim=1).cuda().bfloat16()
    
    with torch.no_grad():
        outputs = llm(inputs_embeds=combined_embeds,use_cache=False)
    output_logits = outputs.logits[:,-1]

    zero_dose_p = output_logits.gather(1,output_terms_tok[0,0].repeat(batch_size,1).cuda())
    low_dose_p = output_logits.gather(1,output_terms_tok[1,0].repeat(batch_size,1).cuda())
    high_dose_p = output_logits.gather(1,output_terms_tok[2,0].repeat(batch_size,1).cuda())
    
    mf_block_norm = torch.functional.F.softmax(torch.cat([zero_dose_p,low_dose_p,high_dose_p],dim=1),dim=-1)
    agg_mu = torch.einsum('br,rx->bx', mf_block_norm, mf_vals.bfloat16())
    u_new_kg = (torch.trapz(agg_mu * x_vals.unsqueeze(0), x_vals, dim=1) / (torch.trapz(agg_mu, x_vals, dim=1) + 1e-8))

    #yt_class = torch.hstack([
    #    hypo_glucose(yt).mean(dim=-1).unsqueeze(1),
    #    target_glucose(yt).mean(dim=-1).unsqueeze(1),
    #    hyper_glucose(yt).mean(dim=-1).unsqueeze(1)
    #]).argmax(dim=-1)

    #out_class = (output_terms_tok[:,0].repeat(16,1).cuda()).gather(1,yt_class.unsqueeze(1)).squeeze(1)

    #loss_ce = torch.nn.CrossEntropyLoss()(output_logits,out_class)

    #if loss_ce > 1.0:
    #    u_new_kg = u_new_kg/loss_ce
    
    return u_new_kg.item()

class Handler(BaseHTTPRequestHandler):
    def _send_json(self, obj, status=200):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    #curl "http://127.0.0.1:8000/test?x=1&x=2&x=3&x=4&x=5&x=6&x=7&x=8&x=9&x=10&y=0&y=0&y=0&y=0&y=0&y=0&y=0&y=0&y=0&y=0"
    def do_GET(self):
        url = urlparse(self.path)
        query = parse_qs(url.query)  # dict: key -> [values]
        if 'x' in query.keys() and 'y' in query.keys():
            if len(query['x']) == 10 and len(query['y']) == 10:
                u_past = [float(x) for x in query['x']]
                y_past = [float(x) for x in query['y']]
                ut=torch.tensor(u_past).cuda().unsqueeze(0)
                yt=torch.tensor([])
                for y in y_past:
                    yt = torch.hstack([yt,torch.tensor(y).repeat(1,input_toks_per_term)])
                yt = yt.cuda()
                u_new = get_control(yt,ut)
                self._send_json({"u_new_kg": u_new})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length > 0 else b""
        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self._send_json({"error": "invalid JSON"}, status=400)
            return

        url = urlparse(self.path)
        query = parse_qs(url.query)
        # minimal example: echo both query and JSON body
        self._send_json({"path": url.path, "query": query, "json": data})

def run(host="0.0.0.0", port=8000):
    httpd = HTTPServer((host, port), Handler)
    print(f"Serving on http://{host}:{port}")
    httpd.serve_forever()

if __name__ == "__main__":
    run()