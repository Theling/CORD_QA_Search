from tqdm import tqdm
from collections import OrderedDict
import torch
import numpy as np
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'


class BERT_SQUAD_QA:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = lambda ipt_1, ipt_2: model(torch.tensor([ipt_1]).to(torch_device), \
                                 token_type_ids=torch.tensor([ipt_2]).to(torch_device))



    def split_doc(self, text, question, overlap_rate):
        token_ls = self.tokenizer.encode(text)
        question_ls = self.tokenizer.encode(question)
        question_len = len(question_ls)
    #     print(question_len)
        piece_length = 500 - question_len
        ret = []
        start_idx = 0
        while True:
            start_idx = max(0, start_idx - int((overlap_rate-1)*piece_length))
            end_idx = start_idx + piece_length
            content = self.tokenizer.decode(token_ls[start_idx: end_idx])
            content = content.replace('[CLS]', '').replace('[SEP]', '').strip()
            ret.append(content)
            start_idx = end_idx
            if start_idx > len(token_ls):
                break
    #     print([len(tokenizer.encode(con)) for con in ret])
        return ret
    

    def bert_pred(self, ipt_ids):
        tokens = self.tokenizer.convert_ids_to_tokens(ipt_ids)
        sep_index = ipt_ids.index(self.tokenizer.sep_token_id)
        num_seg_a = sep_index + 1
        num_seg_b = len(ipt_ids) - num_seg_a
        segment_ids = [0]*num_seg_a + [1]*num_seg_b
        assert len(segment_ids) == len(ipt_ids)
        n_ids = len(segment_ids)
        assert n_ids < 512
        print(len(ipt_ids))
        start_scores, end_scores = self.model(ipt_ids, segment_ids)
        # Plan 1:
    #     answer_start, answer_end = torch.argmax(start_scores), \
    #                                    torch.argmax(end_scores)
        # Plan 2:

        return start_scores, end_scores, {'tokens': tokens, 'sep_idx': sep_index}

    @staticmethod
    def reconstruct_text(tokens, start=0, stop=-1):
        tokens = tokens[start: stop]
        if '[SEP]' in tokens:
            sepind = tokens.index('[SEP]')
            tokens = tokens[sepind+1:]
        txt = ' '.join(tokens)
        txt = txt.replace(' ##', '')
        txt = txt.replace('##', '')
        txt = txt.strip()
        txt = " ".join(txt.split())
        txt = txt.replace(' .', '.')
        txt = txt.replace('( ', '(')
        txt = txt.replace(' )', ')')
        txt = txt.replace(' - ', '-')
        txt_list = txt.split(' , ')
        txt = ''
        nTxtL = len(txt_list)
        if nTxtL == 1:
            return txt_list[0]
        newList =[]
        for i,t in enumerate(txt_list):
            if i < nTxtL -1:
                if t[-1].isdigit() and txt_list[i+1][0].isdigit():
                    newList += [t,',']
                else:
                    newList += [t, ', ']
            else:
                newList += [t]
        answer = ''.join(newList) 
        if answer.startswith('. ') or answer.startswith(', '):
                answer = answer[2:]
        return answer
            

    def make_bert_squad_prediction(self, document, question):
        overlap_rate = 1.1
        doc_pieces = self.split_doc(document, question, overlap_rate)
        input_ids = [self.tokenizer.encode(question, dp) for dp in doc_pieces] 
        print([len(i) for i in input_ids])
        answers = []
        confidences = []
        for ipt_id in input_ids:
            start_scores, end_scores, info = self.bert_pred(ipt_id)
            sep_index = info['sep_idx']
            start_scores, end_scores = start_scores[:, sep_index:], \
                                end_scores[:, sep_index:]
            tokens_wo_question = info['tokens'][sep_index:]

            answer_start, answer_end = torch.argmax(start_scores), \
                                    torch.argmax(end_scores)
            tokens = self.tokenizer.convert_ids_to_tokens(ipt_id)
            answer = self.reconstruct_text(tokens_wo_question, answer_start, answer_end+1)
            total_score = start_scores[0,answer_start].item()+\
                        end_scores[0,answer_end].item()
            answers.append(answer)
            confidences.append(total_score)
        max_conf = max(confidences)
        argmax_conf = np.argmax(confidences)
        max_answer = answers[argmax_conf]
        return {'answer': answer,
                'confidence': max_conf,
                'text': document}
        

    def search_abstracts(self, hit_dictionary, question):
        result = OrderedDict()
        for k,v in tqdm(hit_dictionary.items()):
            abstract = v['abstract_full']
            if abstract:
                ans = self.make_bert_squad_prediction(abstract, question)
                if ans['answer']: result[k]=ans
        c_ls = np.array([result[key]['confidence'] for key in result])

        if len(c_ls) != 0:
            max_score = c_ls.max()
            total = 0.0
            exp_scores = np.exp(c_ls - max_score)
            for i,k in enumerate(result):
                result[k]['confidence'] = exp_scores[i]
                
        ret = {}
        for k in result:
            c = result[k]['confidence']
            ret[c] = result[k].copy()
        return ret

if __name__ == "__main__":
    from transformers import BertForQuestionAnswering
    from transformers import BertTokenizer
    from transformers import BartTokenizer, BartForConditionalGeneration
    import os
    os.environ["JAVA_HOME"] = "/Library/Java/JavaVirtualMachines/jdk-11.0.2.jdk/Contents/Home"
    from pyserini.search import pysearch

    minDate = '2020/04/02'
    luceneDir = 'lucene-index-covid-2020-04-03/'

    USE_SUMMARY = False
    FIND_PDFS = False

    import tensorflow as tf
    import tensorflow_hub as hub

    QA_MODEL = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    QA_TOKENIZER = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    QA_MODEL.to(torch_device)
    QA_MODEL.eval()

    query = 'Which non-pharmaceutical interventions limit tramsission'
    keywords = '2019-nCoV, SARS-CoV-2, COVID-19, non-pharmaceutical interventions, npi'

    import json
    searcher = pysearch.SimpleSearcher(luceneDir)
    hits = searcher.search(query + '. ' + keywords)
    n_hits = len(hits)
    ## collect the relevant data in a hit dictionary
    hit_dictionary = {}
    for i in range(0, n_hits):
        doc_json = json.loads(hits[i].raw)
        idx = str(hits[i].docid)
        hit_dictionary[idx] = doc_json
        hit_dictionary[idx]['title'] = hits[i].lucene_document.get("title")
        hit_dictionary[idx]['authors'] = hits[i].lucene_document.get("authors")
        hit_dictionary[idx]['doi'] = hits[i].lucene_document.get("doi")

    ## scrub the abstracts in prep for BERT-SQuAD
    for idx,v in hit_dictionary.items():
        abs_dirty = v['abstract']
        # looks like the abstract value can be an empty list
        v['abstract_paragraphs'] = []
        v['abstract_full'] = ''

        if abs_dirty:
            # looks like if it is a list, then the only entry is a dictionary wher text is in 'text' key
            # looks like it is broken up by paragraph if it is in that form.  lets make lists for every paragraph
            # and a new entry that is full abstract text as both could be valuable for BERT derrived QA


            if isinstance(abs_dirty, list):
                for p in abs_dirty:
                    v['abstract_paragraphs'].append(p['text'])
                    v['abstract_full'] += p['text'] + ' \n\n'

            # looks like in some cases the abstract can be straight up text so we can actually leave that alone
            if isinstance(abs_dirty, str):
                v['abstract_paragraphs'].append(abs_dirty)
                v['abstract_full'] += abs_dirty + ' \n\n'


    QA_model = BERT_SQUAD_QA(QA_TOKENIZER, QA_MODEL)
    print(QA_model.search_abstracts(hit_dictionary, query))