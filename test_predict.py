from torch import cuda
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("/data/clue/ChatYuan-large-v2")
model_trained = AutoModelForSeq2SeqLM.from_pretrained("./outputs/model_files/") 
print("end...")

device = 'cuda' if cuda.is_available() else 'cpu'
#device = torch.device('cuda=0') # cuda
model_trained.to(device)
def preprocess(text):
  return text.replace("\n", "_")
def postprocess(text):
  return text.replace("_", "\n")

def answer_fn(text, sample=False, top_p=0.6):
  '''sample：是否抽样。生成任务，可以设置为True;
     top_p：0-1之间，生成的内容越多样、
  '''
  text = preprocess(text)
  encoding = tokenizer(text=[text], truncation=True, padding=True, max_length=768, return_tensors="pt").to(device)
  if not sample: # 不进行采样
    out = model_trained.generate(**encoding, return_dict_in_generate=True, output_scores=False, max_length=128, num_beams=4, length_penalty=0.6)
  else: # 采样（生成）
    out = model_trained.generate(**encoding, return_dict_in_generate=True, output_scores=False, max_length=128, do_sample=True, top_p=top_p)
  out_text = tokenizer.batch_decode(out["sequences"], skip_special_tokens=True)
  return postprocess(out_text[0])
print("end...")


if __name__ == '__main__':
    text="这是关于哪方面的新闻： 故事,文化,娱乐,体育,财经,房产,汽车,教育,科技,军事,旅游,国际,股票,农业,游戏?如果日本沉没，中国会接收日本难民吗？"
    result=answer_fn(text, sample=False, top_p=0.6)
    print("result2:",result)
