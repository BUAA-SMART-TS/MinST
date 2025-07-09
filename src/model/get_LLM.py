# from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
#   BertModel, BertTokenizer
from transformers import GPT2Model, GPT2Tokenizer, GPT2LMHeadModel
from transformers import LlamaModel, LlamaTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import logging


path_dic = {
    'LLAMA': ['/home/incoming/LLM/llama3/llama3-8b-instruct', '/home/incoming/LLM/llama3/llama3-8b-instruct', '/home/incoming/LLM/llama3/llama3-8b-instruct'],
    'qwen': ['/home/incoming/LLM/qwen2/qwen2-7b-instruct', '/home/incoming/LLM/qwen2/qwen2-7b-instruct', '/home/incoming/LLM/qwen2/qwen2-7b-instruct'],
    'mistral': ['/home/incoming/LLM/mistral/mistral-7b-v0_1', '/home/incoming/LLM/mistral/mistral-7b-v0_1', '/home/incoming/LLM/mistral/mistral-7b-v0_1'],
    'WizardLM2': ['/home/incoming/LLM/wizardlm2/wizardlm2-8x22b', '/home/incoming/LLM/wizardlm2/wizardlm2-8x22b', '/home/incoming/LLM/wizardlm2/wizardlm2-8x22b'],
    'phi3': ['/home/incoming/LLM/phi/phi-3-mini-4k-instruct', '/home/incoming/LLM/phi/phi-3-mini-4k-instruct', '/home/incoming/LLM/phi/phi-3-mini-4k-instruct'],
    'GPT2': ['/home/incoming/LLM/misc/gpt2-medium', '/home/incoming/LLM/misc/gpt2-medium', '/home/incoming/LLM/misc/gpt2-medium'],
    'BERT': ['google-bert/bert-base-uncased', 'google-bert/bert-base-uncased', 'google-bert/bert-base-uncased']

}

def get_LLM(llm_model, llm_layers):
    print(llm_model)
    if llm_model == 'LLAMA' or llm_model == 'qwen' or llm_model == 'mistral' or llm_model == 'WizardLM2' or llm_model == 'phi3':
        # llama_config = LlamaConfig.from_pretrained(path_dic[llm_model][2])
        # llama_config.num_hidden_layers = llm_layers
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                path_dic[llm_model][1],
                trust_remote_code=True,
                local_files_only=True
            )
        except EnvironmentError as e:  # downloads the tokenizer from HF if not already done
            logging.info(e)
            print("Local tokenizer files not found. Atempting to download them..")
            tokenizer = LlamaTokenizer.from_pretrained(
                path_dic[llm_model][1],
                trust_remote_code=True,
                local_files_only=False
            )
        try:
            llm_model = AutoModelForCausalLM.from_pretrained(
                path_dic[llm_model][0],
                trust_remote_code=True,
                local_files_only=True,
                torch_dtype='auto',
                device_map="auto",
                ignore_mismatched_sizes=True
                # config=llama_config,
                # load_in_4bit=True
            )
        except EnvironmentError:  # downloads model from HF is not already done
            print("Local model files not found. Attempting to download...")
            llm_model = LlamaModel.from_pretrained(
                path_dic[llm_model][0],
                trust_remote_code=True,
                local_files_only=False,
                # config=llama_config,
                # load_in_4bit=True
            )
    elif llm_model == 'GPT2':
        # gpt2_config = GPT2Config.from_pretrained(path_dic[llm_model][2])

        # gpt2_config.num_hidden_layers = llm_layers
        # gpt2_config.output_attentions = True
        # gpt2_config.output_hidden_states = True
        try:
            logging.info('Use Path: {}'.format(path_dic[llm_model][0]))
            tokenizer = AutoTokenizer.from_pretrained(
                path_dic[llm_model][0],
                # trust_remote_code=True,
                # local_files_only=True
            )
        except EnvironmentError:  # downloads the tokenizer from HF if not already done
            print("Local tokenizer files not found. Atempting to download them..")
            tokenizer = GPT2Tokenizer.from_pretrained(
                path_dic[llm_model][1],
                trust_remote_code=True,
                local_files_only=False
            )
        try:
            llm_model = AutoModelForCausalLM.from_pretrained(
                path_dic[llm_model][0],
                # trust_remote_code=True,
                # local_files_only=True,
                # config=gpt2_config,
                output_attentions=True,
                output_hidden_states = True
            )
        except EnvironmentError:  # downloads model from HF is not already done
            print("Local model files not found. Attempting to download...")
            llm_model = AutoModelForCausalLM.from_pretrained(
                path_dic[llm_model][0],
                trust_remote_code=True,
                local_files_only=False,
                # config=gpt2_config,
                output_attentions=True,
                output_hidden_states=True
            )


        '''
    elif llm_model == 'BERT':
        bert_config = BertConfig.from_pretrained(path_dic[llm_model][2])

        bert_config.num_hidden_layers = llm_layers
        bert_config.output_attentions = True
        bert_config.output_hidden_states = True
        try:
            llm_model = BertModel.from_pretrained(
                path_dic[llm_model][0],
                trust_remote_code=True,
                local_files_only=True,
                config=bert_config,
            )
        except EnvironmentError:  # downloads model from HF is not already done
            print("Local model files not found. Attempting to download...")
            llm_model = BertModel.from_pretrained(
                path_dic[llm_model][0],
                trust_remote_code=True,
                local_files_only=False,
                config=bert_config,
            )

        try:
            tokenizer = BertTokenizer.from_pretrained(
                path_dic[llm_model][1],
                trust_remote_code=True,
                local_files_only=True
            )
        except EnvironmentError:  # downloads the tokenizer from HF if not already done
            print("Local tokenizer files not found. Atempting to download them..")
            tokenizer = BertTokenizer.from_pretrained(
                path_dic[llm_model][1],
                trust_remote_code=True,
                local_files_only=False
            )
    '''
    else:
        raise Exception('LLM model is not defined')

    return tokenizer, llm_model
