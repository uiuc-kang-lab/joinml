from typing import List
from itertools import product
import numpy as np
from joinml.proxy.proxy import Proxy
from py_stringmatching.tokenizer.alphanumeric_tokenizer import AlphanumericTokenizer
from joinml.config import Config
from tqdm import tqdm
import multiprocessing as mp
import threading
import time


available_proxy = {
    # "Affine",
    "Bag Distance",
    "Cosine",
    "Dice",
    "Editex",
    "Generalized Jaccard",
    "Hamming Distance",
    "Jaccard",
    "Jaro",
    "Jaro Winkler",
    "Levenshtein",
    "Monge Elkan",
    # "Needleman Wunsch",
    "Overlap Coefficient",
    "Partial Ratio",
    "Partial Token Sort",
    "Ratio",
    # "Smith Waterman",
    "Soft TF/IDF",
    "Soundex",
    "TF/IDF",
    "Token Sort",
    "Tversky Index"
}

class StringMatchingProxy(Proxy):
    
    def __init__(self, config: Config) -> None:
        super().__init__()
        proxy_name = config.proxy
        self.tokenizer = None
        self.parallelProxyCalculation = config.parallelProxyCalculation
        if self.parallelProxyCalculation:
            self.numProxyProcess = int(mp.cpu_count() * config.numProcessPerCPU)
        if proxy_name not in available_proxy:
            raise ValueError(f"Proxy {proxy_name} is not available.")
        # elif proxy_name == "Affine":
        #     from py_stringmatching.similarity_measure.affine import Affine
        #     self.proxy = Affine()
        #     self.sim_func = self.proxy.get_raw_score
        elif proxy_name == "Bag Distance":
            from py_stringmatching.similarity_measure.bag_distance import BagDistance
            self.proxy = BagDistance()
            self.sim_func = self.proxy.get_sim_score
        elif proxy_name == "Cosine":
            from py_stringmatching.similarity_measure.cosine import Cosine
            self.proxy = Cosine()
            self.tokenizer = AlphanumericTokenizer()
            self.sim_func = self.proxy.get_sim_score
        elif proxy_name == "Dice":
            from py_stringmatching.similarity_measure.dice import Dice
            self.tokenizer = AlphanumericTokenizer()
            self.proxy = Dice()
            self.sim_func = self.proxy.get_sim_score
        elif proxy_name == "Editex":
            from py_stringmatching.similarity_measure.editex import Editex
            self.proxy = Editex()
            self.sim_func = self.proxy.get_sim_score
        elif proxy_name == "Generalized Jaccard":
            from py_stringmatching.similarity_measure.generalized_jaccard import GeneralizedJaccard
            self.proxy = GeneralizedJaccard()
            self.tokenizer = AlphanumericTokenizer()
            self.sim_func = self.proxy.get_sim_score
        elif proxy_name == "Hamming Distance":
            from py_stringmatching.similarity_measure.hamming_distance import HammingDistance
            self.proxy = HammingDistance()
            def sim_func(str1, str2):
                if len(str1) > len(str2):
                    tmp = str1
                    str1 = str2
                    str2 = tmp
                if len(str1) < len(str2):
                    str1 = str1 + " " * (len(str2) - len(str1))
                return self.proxy.get_sim_score(str1, str2)
            self.sim_func = sim_func
        elif proxy_name == "Jaccard":
            from py_stringmatching.similarity_measure.jaccard import Jaccard
            self.proxy = Jaccard()
            self.tokenizer = AlphanumericTokenizer()
            self.sim_func = self.proxy.get_sim_score
        elif proxy_name == "Jaro":
            from py_stringmatching.similarity_measure.jaro import Jaro
            self.proxy = Jaro()
            self.sim_func = self.proxy.get_sim_score
        elif proxy_name == "Jaro Winkler":
            from py_stringmatching.similarity_measure.jaro_winkler import JaroWinkler
            self.proxy = JaroWinkler()
            self.sim_func = self.proxy.get_sim_score
        elif proxy_name == "Levenshtein":
            from py_stringmatching.similarity_measure.levenshtein import Levenshtein
            self.proxy = Levenshtein()
            self.sim_func = self.proxy.get_sim_score
        elif proxy_name == "Monge Elkan":
            from py_stringmatching.similarity_measure.monge_elkan import MongeElkan
            self.proxy = MongeElkan()
            self.tokenizer = AlphanumericTokenizer()
            self.sim_func = self.proxy.get_raw_score
        # elif proxy_name == "Needleman Wunsch":
        #     from py_stringmatching.similarity_measure.needleman_wunsch import NeedlemanWunsch
        #     self.proxy = NeedlemanWunsch()
        #     self.sim_func = self.proxy.get_raw_score
        elif proxy_name == "Overlap Coefficient":
            from py_stringmatching.similarity_measure.overlap_coefficient import OverlapCoefficient
            self.proxy = OverlapCoefficient()
            self.tokenizer = AlphanumericTokenizer()
            self.sim_func = self.proxy.get_sim_score
        elif proxy_name == "Partial Ratio":
            from py_stringmatching.similarity_measure.partial_ratio import PartialRatio
            self.proxy = PartialRatio()
            self.sim_func = self.proxy.get_sim_score
        elif proxy_name == "Partial Token Sort":
            from py_stringmatching.similarity_measure.partial_token_sort import PartialTokenSort
            self.proxy = PartialTokenSort()
            self.sim_func = self.proxy.get_sim_score
        elif proxy_name == "Ratio":
            from py_stringmatching.similarity_measure.ratio import Ratio
            self.proxy = Ratio()
            self.sim_func = self.proxy.get_sim_score
        # elif proxy_name == "Smith Waterman":
        #     from py_stringmatching.similarity_measure.smith_waterman import SmithWaterman
        #     self.proxy = SmithWaterman()
        #     self.sim_func = self.proxy.get_raw_score
        elif proxy_name == "Soft TF/IDF":
            from py_stringmatching.similarity_measure.soft_tfidf import SoftTfIdf
            self.proxy = SoftTfIdf()
            self.tokenizer = AlphanumericTokenizer()
            self.sim_func = self.proxy.get_raw_score
        elif proxy_name == "Soundex":
            from py_stringmatching.similarity_measure.soundex import Soundex
            self.proxy = Soundex()
            self.sim_func = self.proxy.get_sim_score
        elif proxy_name == "TF/IDF":
            from py_stringmatching.similarity_measure.tfidf import TfIdf
            self.proxy = TfIdf()
            self.tokenizer = AlphanumericTokenizer()
            self.sim_func = self.proxy.get_sim_score
        elif proxy_name == "Token Sort":
            from py_stringmatching.similarity_measure.token_sort import TokenSort
            self.proxy = TokenSort()
            self.sim_func = self.proxy.get_sim_score
        elif proxy_name == "Tversky Index":
            from py_stringmatching.similarity_measure.tversky_index import TverskyIndex
            self.proxy = TverskyIndex()
            self.tokenizer = AlphanumericTokenizer()
            self.sim_func = self.proxy.get_sim_score
        else:
            raise ValueError(f"Proxy {proxy_name} is not available.")
    
    def get_proxy_score_for_tables(self, table1: List[str], table2: List[str]) -> np.ndarray:
        if self.tokenizer is not None:
            table1 = [self.tokenizer.tokenize(x) for x in table1]
            table2 = [self.tokenizer.tokenize(x) for x in table2]
        
        scores = np.zeros((len(table1), len(table2)))
        self.__proxyPreProcess(self.proxy, table1 + table2)

        if(self.parallelProxyCalculation and self.__isParallelble()):
            inputQueue = mp.Queue()
            outputQueue = mp.Queue()
            maxInputQueueSize = 1000

            proxyProcess = []
            for _ in range(self.numProxyProcess):
                p = mp.Process(target=self.__parallelProcessing, args=(inputQueue,outputQueue, table2))
                proxyProcess.append(p)
                p.start()

            reduceThread = threading.Thread(target=self.__reduceTable, args=(scores, outputQueue))
            reduceThread.start()
            for id1 in range(len(table1)):
                while inputQueue.qsize()>=maxInputQueueSize:
                    time.sleep(1)
                inputQueue.put((id1,table1[id1]))
            
            for _ in range(self.numProxyProcess):
                inputQueue.put(None)
            
            for p in proxyProcess:
                p.join()
            
            outputQueue.put(None)
            reduceThread.join()
        else:
            for id1, id2 in tqdm(product(list(range(len(table1))), list(range(len(table2))))):
                scores[id1, id2] = self.sim_func(table1[id1], table2[id2])
        
        return scores
    
    def get_proxy_score_for_tuples(self, tuples: List[List[str]]) -> np.ndarray:
        scores = np.zeros(len(tuples))
        
        if self.tokenizer is not None:
            corpus = []
            for t in tuples:
                corpus.append(self.tokenizer.tokenize(t[0]))
                corpus.append(self.tokenizer.tokenize(t[1]))
            self.__proxyPreProcess(self.proxy,corpus)

        for i in range(len(tuples)):
            t = tuples[i]
            if len(t) != 2:
                raise ValueError("Each tuple must have exactly two elements.")
            if self.tokenizer is not None:
                t = [self.tokenizer.tokenize(x) for x in t]

            scores[i] = self.sim_func(t[0], t[1])

        return scores

    # tfIdf need document frequency table
    def __proxyPreProcess(self, proxy, corpus: list):
        from py_stringmatching.similarity_measure.soft_tfidf import SoftTfIdf
        from py_stringmatching.similarity_measure.tfidf import TfIdf
        if isinstance(proxy, SoftTfIdf) or isinstance(proxy, TfIdf):
            proxy.__init__(corpus)
    
    def __isParallelble(self) -> bool:
        from py_stringmatching.similarity_measure.hamming_distance import HammingDistance
        if isinstance(self.proxy,HammingDistance):
            return False
        return True
    
    #Note that parallel will cause slight error, around 1e^-17
    def __parallelProcessing(self,inputQueue: mp.Queue, outputQueue: mp.Queue, table2: list):
        while(True):
            input = inputQueue.get()
            if input == None:
                return
            table1Idx = input[0]
            table1String = input[1]
            score = np.zeros(len(table2))
            for idx,s in enumerate(table2):
                score[idx] = self.sim_func(s,table1String)
            outputQueue.put((table1Idx,score))

    def __reduceTable(self, output: np.ndarray, outputQueue: mp.Queue):
        for _ in tqdm(range(output.shape[0])):
            newItem = outputQueue.get()
            #print(newItem[0])
            if newItem == None:
                print("ERR: reduce terminate early, not enough rows recieved!")
                return
            output[newItem[0]] = newItem[1]

if __name__ == "__main__":
    config = Config()
    sentences = [
        "How can Internet speed be increased by hacking through DNS?",
        "Which is the best digital marketing institution in banglore?",
        "Is it possible to store the energy of lightning?",
        "What is purpose of life?",
        "Why do Swiss despise Asians?",
        "What are the best associate product manager (APM) programs that someone in their early 20s can join to learn product management and have a rewarding career in the company?",
        "How can I speak with more clarity and confidence?",
        "How can we make the world a better place to live in for the future generations?",
        "How do you potty train a 4 months Pitbull?",
        "What will happen if I cancel TQWL tickets before the charting is done?",
        "How do people die?",
        "How can I keep my dog from getting fleas?",
        "How do I add a second device to a Google Play account?",
        "How did early Europeans get protein before the Columbian Exchange?",
        "Why can't we fall asleep on some days?",
        "How can I install OBB/data file on an Android?",
        "What are some good free web scrapers / scraping techniques?",
        "What are the best places to recommend foreigners to visit in Nepal?",
        "Why do some people believe that life ends at death?",
        "What is the future for iOS Developers?",
    ]
    table1 = sentences[:10]
    table2 = sentences[10:]
    tuples = list(product(sentences, sentences))

    run_time = {}
    for proxy_name in available_proxy:
        print(f"Proxy: {proxy_name}")
        config.proxy = proxy_name
        proxy = StringMatchingProxy(config)
        start = time.time()
        for _ in range(1):
            print(proxy.get_proxy_score_for_tables(table1, table2))
            print(proxy.get_proxy_score_for_tuples(tuples))
        end = time.time()
        run_time[proxy_name] = end - start
    print(run_time)

# performance ranking
# run each method for 10 * (20*20 + 10*10) = 5000 pairs
#  'Dice': 0.055444955825805664, 
#  'Cosine': 0.05652213096618652, 
#  'Overlap Coefficient': 0.05605721473693848, 
#  'Jaccard': 0.0589451789855957, 
#  'Tversky Index': 0.06343483924865723, 
#  'Jaro': 0.06966876983642578, 
#  'Jaro Winkler': 0.07201099395751953, 
#  'Hamming Distance': 0.08649897575378418, 
#  'Bag Distance': 0.12410283088684082, 
#  'TF/IDF': 0.12971186637878418, 
#  'Soundex': 0.14389395713806152, 
#  'Ratio': 0.7833948135375977, 
#  'Levenshtein': 0.7902331352233887, 
#  'Needleman Wunsch': 0.8870401382446289, 
#  'Token Sort': 1.0537819862365723, 
#  'Smith Waterman': 1.0808098316192627, 
#  'Generalized Jaccard': 1.2929012775421143, 
#  'Soft TF/IDF': 1.3125100135803223, 
#  'Monge Elkan': 1.4151699542999268, 
#  'Affine': 3.93766713142395, 
#  'Partial Ratio': 5.961280107498169
#  'Partial Token Sort': 6.329225063323975, 
#  'Editex': 28.28290581703186, 

'''
def checkTwoArrayEqual(result1, result2) -> bool:
    for i in range(result1.shape[0]):
        for j in range(result2.shape[1]):
            if abs(result1[i,j] - result2[i,j]) > 1e-6:
                return False
    return True
'''