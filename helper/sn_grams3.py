'''
Version 3.2
Created on 11/12/2014, tested for Python 2.7
Update on 9/05/2016  (mixed sn-grams supported)
@authors: Juan Pablo Posadas, Grigori Sidorov
Class for obtaining syntactic n-grams from dependency trees using Stanford parser output
This version supports continuous, non continuous sn-grams and without stop words or content sn-grams. It also supports mixed sn-grams (first element of one type and the rest of elements of different type)
NOTE: Input tree should NOT be collapsed: -outputFormat "wordsAndTags, typedDependencies" -outputFormatOptions "basicDependencies"
NOTE: Since ",", "[", "]", and "\" are part of our metalanguage, we add a slash to them when they are part of the sentence (sn-grams), e.g., "\,", "\[", "\\"
This version allows to process separately nodes that have too many children (for a given threshold) and select min and max sizes of sn-grams. Default: 5, 2, 7
Ensure not to use input symbols that are not UTF-8.
'''

import copy, sys, re
import codecs

class ReduceDepInfo(object):
    '''
    This class obtains a reduced version of the original syntactic tree, eliminating the stop words
    '''
    def __init__(self, obj, dictionary, has_POS):
        '''
        Constructor
        '''
        self.word          = copy.deepcopy(obj.word) #Dictionary of original words according to their positions
        self.pos           = copy.deepcopy(obj.pos) #Dictionary with POS tags of words                
        self.dep           = copy.deepcopy(obj.dep) #Dictionary with the index of the father
        self.rel           = copy.deepcopy(obj.rel) #Dictionary with dependency relations 
        self.children      = copy.deepcopy(obj.children) #Dictionary with words that are dependent for a (key) word
        self.leaves        = [] #List of indexes of words that are leaves        
        self.root_idx      = obj.root_idx #Index of the root

        self.drop_nodes(dictionary, obj, has_POS)
    ########################################################################

    def drop_nodes(self, dictionary, obj, has_POS):                
        nodes = copy.deepcopy(list(obj.word.keys())) #List of indexes of the sentece elements without the root
        nodes.remove(obj.root_idx)        
        for node in nodes:
            if self.word[node].lower() in dictionary:#Case where we have to remove the node
                father = self.dep[node]                
                relation = self.rel[node]
                if node in self.children:                
                    for child in self.children[node]:
                        self.children[father].append(child)#Update the relations between nodes
                        self.rel[child] = relation + ";"+ self.rel[child]
                        self.dep[child] = father
                    self.children.pop(node)#Update the general dictionary of children
                        
                #Eliminate its occurrence in the tree
                self.word.pop(node)                                
                self.dep.pop(node)
                self.rel.pop(node)                
                self.children[father].remove(node)
                if has_POS == 1:
                    self.pos.pop(node)
                if len(self.children[father]) == 0:#Update the status of a node to a leaf
                    self.children.pop(father)        
        for i in list(self.word.keys()):#this cycle updates the list of leaves in the sentence
            if i not in list(self.children.keys()):
                self.leaves.append(i)
    ######################


class OnlyStopW(object):
    '''
    This class obtains a reduced version of the original syntactic tree, keeping only stop words
    '''
    def __init__(self, obj, dictionary, has_POS):
        '''
        Constructor
        '''
        self.word          = copy.deepcopy(obj.word) #Dictionary of original words according to their positions
        self.pos           = copy.deepcopy(obj.pos) #Dictionarywith POS tags of words        
        self.dep           = copy.deepcopy(obj.dep) #Dictionary with the index of the father
        self.rel           = copy.deepcopy(obj.rel) #Dictionary with dependency relations 
        self.children      = copy.deepcopy(obj.children) #Dictionary with words that are dependent for a (key) word
        self.leaves        = [] #List of indexes of words that are leaves        
        self.root_idx      = obj.root_idx #Index of the root

        self.word[self.root_idx] = "null"        
        self.rel[self.root_idx] = "null"
        self.drop_nodes(dictionary, obj, has_POS)
    ########################################################################

    def drop_nodes(self, dictionary, obj, has_POS):                
        nodes = copy.deepcopy(list(obj.word.keys())) #List of indexes of the sentece elements without the root
        nodes.remove(obj.root_idx)        
        for node in nodes:
            if self.word[node].lower() not in dictionary:#Case where we have to remove the node
                father = self.dep[node]                
                relation = self.rel[node]
                if node in self.children:                
                    for child in self.children[node]:
                        self.children[father].append(child)#Update the relations between nodes
                        self.rel[child] = relation + ";"+ self.rel[child]
                        self.dep[child] = father
                    self.children.pop(node)#Update the general dictionary of children
                        
                #Eliminate its appearence in the tree
                self.word.pop(node)                                
                self.dep.pop(node)
                self.rel.pop(node)                
                self.children[father].remove(node)
                if has_POS == 1:                    
                    self.pos.pop(node)
                if len(self.children[father]) == 0:#Update the status of a node to a leaf
                    self.children.pop(father)
                            
        for i in list(self.word.keys()):#this cycle updates the list of leaves in the sentence
            if i not in list(self.children.keys()):
                self.leaves.append(i)
    ######################
            
                                                                          
class DepInfo(object):
    '''
    This class represents the dependency information of a sentence
    ''' 
    def __init__(self, lines, has_POS):
        '''
        Constructor
        '''
        self.word          = {} #Dictionary of original words according to their positions
        self.pos           = {} #Dictionary with POS tags of words        
        self.rel           = {} #Dictionary with dependency relations 
        self.dep           = {} #Dictionary with the index of the father
        self.children      = {} #Dictionary with words that are dependent for a (key) word        
        self.leaves        = [] #List of indexes of words that are leaves        
        self.root_idx      = -1 #Index of the root
        self.prepare_indices(lines,has_POS)
        
    ######################
    def prepare_indices(self, lines,has_POS):
        pattern = r"(.*)\((.+)-(\d+),\s(.+)-(\d+)\)"
        
        if has_POS == 1:
            postags = lines.pop(0)#Obtain the information about POS tags from the list lines                
        for line in lines:
            matchObj = re.match(pattern, line)
            idx = int(matchObj.group(5))
            self.rel[idx] = matchObj.group(1)
                    
            p_idx = int(matchObj.group(3))
            self.dep [idx]  = p_idx
            self.word[idx]  = matchObj.group(4)
      
            if self.word[idx] == ',':
                self.word[idx] = "\,"
            elif self.word[idx] == '[':
                self.word[idx] = "\["
            elif self.word[idx] == ']':
                self.word[idx] = "\]"
            elif self.word[idx] == '\\':
                self.word[idx] = "\\\\"
            ###Case added to handle number that include colons (thousands)
            if self.word[idx].find(",")>-1:
                self.word[idx] = self.word[idx].replace(",","\,")
				
            self.children[p_idx] = self.children.get(p_idx, [])
            self.children[p_idx].append(idx)

            if self.dep [idx] == 0:
                self.root_idx = idx
                self.rel[idx] = "root"#Line added because of the FREELING output
      
        #Determine if a word is a leaf
        for i in list(self.word.keys()):
            if i not in list(self.children.keys()):
                self.leaves.append(i)
        
        if has_POS == 1:
            ####Next section stores the information of POS tags
            postags = postags.split(" ")
            aa = list(self.word.keys())
            aa.sort()
            for idx in aa:
                self.pos[idx] = postags[idx-1].split("/")[1]                
######################                                            


class BiSNgrams(object):
    '''
    classdocs
    '''
    def __init__(self, min_size, max_size, max_num_children, option):
        '''
        Constructor
        '''
        self.min_size         = min_size         #The minimum size for the sn-grams
        self.max_size         = max_size         #The maximum size for the sn-grams
        self.max_num_children = max_num_children #The maximum number of children per node
         
        if option in range(0,8):
            self.option         = option #Type of sn-grams to be obtained: 0 for WORD sn-grams; 1 for sn-grams of SR Tags ; 2 for both
        else: 
            print("Error: Invalid value for the parameter option")
            exit(1)
            
        self.subtrees       = [] #List that contains all the nodes that are not leaves
        self.DepNgrams      = []
        self.log            = [] #List that contains the nodes that have more children than the parameter max_num_children                
        self.dicPOSTags     = []
        self.dicSRTags      = []
        self.dicWordNgrams  = []
        self.dicWordSR      = []
        self.dicWordPOS     = []
        self.dicSRWord      = []
        self.dicSRPOS       = []
        self.dicPOSWord     = []
        self.dicPOSSR       = []
        
        for i in range(min_size, max_size+1):
            self.dicPOSTags.append({})
            self.dicSRTags.append({})
            self.dicWordNgrams.append({})
    
    def reset_vars (self):
        del self.subtrees[:]
        del self.DepNgrams[:] 
        del self.log[:]        
        del self.dicPOSTags[:]
        del self.dicSRTags[:]
        del self.dicWordNgrams[:] 
        del self.dicWordSR[:]
        del self.dicWordPOS[:]
        del self.dicSRWord[:]
        del self.dicSRPOS[:]
        del self.dicPOSWord[:]
        del self.dicPOSSR[:]       
        for i in range(self.min_size, self.max_size+1):
            self.dicPOSTags.append({})
            self.dicSRTags.append({})
            self.dicWordNgrams.append({}) 
            self.dicWordSR.append({})
            self.dicWordPOS.append({})
            self.dicSRWord.append({})
            self.dicSRPOS.append({})
            self.dicPOSWord.append({})
            self.dicPOSSR.append({})                          
    #########################################
    
    
    def print_parsed_sentence(self, sentence, has_POS):
        line = "*****Sentence: "
        for i in sorted(sentence.word.keys()):
            line += sentence.word[i]+" "
        print(line.rstrip(" "))
        
        #########'''
        line = ""
        for i in list(sentence.word.keys()):
            line  =     str(i)         + "\t"
            line +=     sentence.word[i]  + "\t"
            if has_POS == 1:
                line +=     sentence.pos [i]  + "\t"
            line +=     sentence.rel [i]  + "\t"
            line += str(sentence.dep [i]) + "\t"
            if i not in sentence.leaves:
                line += str(sentence.children[i])
            print(line)

        print("Leaf nodes are:")
        line = ""
        for i in sentence.leaves:
            line += sentence.word[i] + ", "
        print(line)
        #########'''
    
    def write_all_sn_grams (self, f2,sent_num):
        '''
        This method write in a file one kind of sn-gram according with the value of op
        '''        
        if self.option == 0:#Acording with the params, the sngrams are stored in the container                                
            d = self.dicWordNgrams                                        
            f2.write("Sentence "+sent_num+" ************sn-grams of words:\n")        
        elif self.option == 1:                
            d = self.dicSRTags                
            f2.write("Sentence "+sent_num+" ************sn-grams of tags of syntactic relations (SR tags):\n")
        elif self.option == 2:
            d = self.dicPOSTags                 
            f2.write("Sentence "+sent_num+" ************sn-grams of POS tags:\n")
        elif self.option == 3:
            d = self.dicWordSR
            f2.write("Sentence "+sent_num+" ************sn-grams Word-SR:\n")                    
        elif self.option == 4:
            d = self.dicWordPOS
            f2.write("Sentence "+sent_num+" ************sn-grams Word-POS:\n")                    
        elif self.option == 5:
            d = self.dicSRWord                    
            f2.write("Sentence "+sent_num+" ************sn-grams SR-Word:\n")
        elif self.option == 6:
            d = self.dicSRPOS                    
            f2.write("Sentence "+sent_num+" ************sn-grams SR-POS:\n")
        elif self.option == 7:
            d = self.dicPOSWord                    
            f2.write("Sentence "+sent_num+" ************sn-grams POS-Word:\n")
        elif self.option == 8:
            d = self.dicPOSSR 
            f2.write("Sentence "+sent_num+" ************sn-grams POS-SR:\n")
        
        for idx, dic in enumerate(d):
            f2.write ("\n************Size: " + str(idx + self.min_size) + "\n")
            if len(list(dic.keys())) > 0:
                for item in list(dic.keys()):
                    f2.write (item + "\t" + str(dic[item]) + "\n")
            else:
                f2.write("EMPTY\n")
        f2.write("\n")        
        #if self.option == 0 or self.option == 3:
        #    self.write_WordSngrams(f2,sent_num)
        #if self.option == 1 or self.option == 3:
        #    self.write_SRSngrams(f2,sent_num)
        #if self.option == 2 or self.option == 3:
        #    self.write_POSSngrams(f2,sent_num)
    ################################
    
    '''
    def write_WordSngrams(self,f2,sent_num):
        f2.write("Sentence "+sent_num+" ************sn-grams of words:\n")
        for idx, dic in enumerate(self.dicWordNgrams):
            f2.write ("\n************Size: " + str(idx + self.min_size) + "\n")
            if len(dic.keys()) > 0:
                for item in dic.keys():
                    f2.write (item + "\t" + str(dic[item]) + "\n")
            else:
                f2.write("EMPTY\n")
        f2.write("\n")
        
    def write_SRSngrams(self,f2,sent_num):
        f2.write("Sentence "+sent_num+" ************sn-grams of tags of syntactic relations (SR tags):\n")
        for idx, dic in enumerate(result.dicSRTags):
            f2.write ("\n************Size: " + str(idx + self.min_size) + "\n")
            if len(dic.keys()) > 0:
                for item in dic.keys():
                    f2.write (item + "\t" + str(dic[item]) + "\n")
            else:
                f2.write("EMPTY\n")
        f2.write("\n")
      
    def write_POSSngrams(self,f2,sent_num):
        f2.write("Sentence "+sent_num+" ************sn-grams of POS tags:\n")
        for idx, dic in enumerate(self.dicPOSTags):
            f2.write ("\n************Size: " + str(idx + self.min_size) + "\n")
            if len(dic.keys()) > 0:
                for item in dic.keys():
                    f2.write (item + "\t" + str(dic[item]) + "\n")
            else:
                f2.write("EMPTY\n")
        f2.write("\n")
    '''
        
    def print_sngrams(self):
        
        if self.option == 0:#Acording with the params, the sngrams are stored in the container                                
            d = self.dicWordNgrams                                        
            print("************sn-grams of words:")        
        elif self.option == 1:                
            d = self.dicSRTags                
            print("************sn-grams of tags of syntactic relations (SR tags):")
        elif self.option == 2:
            d = self.dicPOSTags                 
            print("************sn-grams of POS tags:")
        elif self.option == 3:
            d = self.dicWordSR
            print("************sn-grams Word-SR:")                    
        elif self.option == 4:
            d = self.dicWordPOS
            print("************sn-grams Word-POS:")                    
        elif self.option == 5:
            d = self.dicSRWord                    
            print("************sn-grams SR-Word:")
        elif self.option == 6:
            d = self.dicSRPOS                    
            print("************sn-grams SR-POS:")
        elif self.option == 7:
            d = self.dicPOSWord                    
            print("************sn-grams POS-Word:")
        elif self.option == 8:
            d = self.dicPOSSR 
            print("************sn-grams POS-SR:")
        
        for idx, dic in enumerate(d):
            print("\n************Size: " + str(idx + self.min_size))
            if len(list(dic.keys())) > 0:
                for item in list(dic.keys()):
                    print(item + "\t"+str(dic[item]))
            else:
                print("EMPTY")
            print("****************************************")        
    ###########################

        
    def process_sentence (self, lines, option2, dictionary,has_POS):
        '''
        This method calls the specific methods (general steps) for producing sn-grams according to the parameter "option"
        '''
        self.reset_vars()
        sentence = DepInfo(lines,has_POS)

        self.print_parsed_sentence(sentence,has_POS)
        #print "Valor de la opcion 2 "+str(option2)
        if option2 == 1:#In this case we reduce the 
            sentence = ReduceDepInfo(sentence, dictionary, has_POS)
            self.print_parsed_sentence(sentence)
        elif option2 == 2:
            sentence = OnlyStopW(sentence, dictionary, has_POS)
            self.print_parsed_sentence(sentence)
                            
        for i in list(sentence.word.keys()):            
            if i in list(sentence.children.keys()):
                self.subtrees.append(i)#Store all the possible roots of the subtrees                
                             
        if self.option in range(0,9):
            if self.min_size >= 0:
                if self.max_size >= self.min_size:
                    if self.min_size <= len(sentence.word): 
                        
                        if self.max_size > len(sentence.word):
                            line = "\tMessage: The value of the maximum size exceeds the length of the sentence.\n"
                            #print line                            
                        if self.min_size == 0 or self.max_size == 0:
                            line = "\tMessage: the program will obtain the sn-grams of all possible sizes.\n"
                            #print line                            
                        
                        log = self.get_all_DepNgrams(sentence)
                                                
                        if len(log) > 0:
                            line = "\tThe next words have more than "+ str(self.max_num_children) +" children:\n"
                            print(line)
                            
                            for item in log:
                                line = "\t\t"+sentence.word[item]+"\n"
                                print(line)
                                                                                                
                        if self.option in range(0,9):
                            self.store_all_DepNgrams(sentence, self.option)                            
                        #else:
                        #    self.store_all_DepNgrams(sentence, 0)
                        #    self.store_all_DepNgrams(sentence, 1)
                        #    self.store_all_DepNgrams(sentence, 2)
                        
                        #self.show_sngrams(sentence, log) #This method only shows the sn-grams obtained from the sentence                    
                    else:                        
                        line = "\tERROR: The value of the minimum size exceeds the length of the sentence\n"
                        # print line
                else:                    
                    line = "\tERROR: The maximum size must be greater than the minimum size\n"
                    print(line)
            else:                
                line = "\tERROR: The value of the minimum size is not allowed\n"
                print(line)
        else:            
            line = "\tERROR: Invalid value for the parameter option\n"
            print(line)
        

    def prepare_SNgram(self, line, sentence, op):
        '''        
        op = -1   for sngrams of index
        op = 0    for sngrams of words
        op = 1    for sngrams of sr tags
        op = 2    for sngrams of POS tags        
        op = 3    for Word/SR SNgrams
        op = 4    for Word/POS SNgrams
        op = 5    for SR/Word SNgrams
        op = 6    for SR/POS SNgrams
        op = 7    for POS/Word SNgrams
        op = 8    for POS/SR SNgrams
        '''
        ngram = ""
        for item in line:
            if type(item) is str:
                ngram += item
            elif type(item) is int:
                if op == -1:
                    ngram += str(item)
                elif op == 0:
                    ngram += sentence.word[item]
                elif op == 1:
                    ngram += sentence.rel[item]
                elif op == 2:
                    ngram += sentence.pos[item]                 
                elif op == 3:
                    ngram += sentence.word[item]
                    op = 1
                elif op == 4:
                    ngram += sentence.word[item]
                    op = 2
                elif op == 5:
                    ngram += sentence.rel[item]
                    op = 0
                elif op == 6:
                    ngram += sentence.rel[item]
                    op = 2
                elif op == 7:
                    ngram += sentence.pos[item]
                    op = 0
                elif op == 8:
                    ngram += sentence.pos[item]
                    op = 1                               
            else:
                ngram += self.prepare_SNgram(item, sentence, op)
        return ngram
    ######################
    
                   
    def is_continuous(self, ngram):
        '''
        This method tests if a sn-gram is continuous or not. It assumes that no punctuation characters are allowed in the sn-gram. Used for testing.
        '''
        answer = ""
        if ngram.count(",") > 0:
            answer = "NO"
        else:
            answer = "YES"                    
        return answer    
    ######################
    
                  
    def len_Ngram(self, ngram):
        n = 1
        n += ngram.count("[")
        n += ngram.count(",")
        n -= ngram.count("\[")
        n -= ngram.count("\,")
        return n  
    ########################
    
    
    def get_all_DepNgrams(self, sentence):
        '''
        This method begins the process of getting all the sn-grams of the dependency tree
        '''                         
        unigrams      = []  #Auxiliar variable that contains all the unigrams
        combinations  = []  #Auxiliar variable that contains all the combinations of a node with its children
        aux           = []
        log           = set()
        
    
        if sentence.root_idx > 0:
            unigrams, combinations, log = self.get_subtrees (sentence)#Call this method first for obtaining all the posible subtrees                    
            
            if len(unigrams) > 0:
                self.DepNgrams.append([sentence.root_idx])
                self.DepNgrams.extend(unigrams)            #Adds the unigrams to the general container
            for item in combinations:                      #Adds the first sn-grams to the general container
                if self.min_size != 0 or self.max_size != 0:                        
                    size = self.len_Ngram(self.prepare_SNgram(item[0], sentence, -1))
                    if size >= self.min_size and size <= self.max_size:        #Check the size of the new sn-grams                
                        self.DepNgrams.append(copy.deepcopy(item[0]))                        
                    if size < self.max_size:
                        aux.append(item)
                else:
                    self.DepNgrams.append(copy.deepcopy(item[0]))

            if self.min_size != 0 or self.max_size != 0:
                self.compound_sngrams(aux, sentence)   #This function generates the rest of sn-grams
            else:
                self.compound_sngrams(combinations, sentence)
        else:
            line = "\tError, no root found\n"
            print(line)        
        return(log)    
    ######################        
    
    
    def store_all_DepNgrams(self, sentence, op):
        '''
        This method stores the sn-grams in the container specified by the parameter "op"
        '''         
        if op == 0:#Acording with the params, the sngrams are stored in the container                                
            d = self.dicWordNgrams                                                
        elif op == 1:                
            d = self.dicSRTags                
        elif op == 2:
            d = self.dicPOSTags                 
        elif op == 3:
            d = self.dicWordSR                    
        elif op == 4:
            d = self.dicWordPOS                    
        elif op == 5:
            d = self.dicSRWord                    
        elif op == 6:
            d = self.dicSRPOS                    
        elif op == 7:
            d = self.dicPOSWord                    
        elif op == 8:
            d = self.dicPOSSR
        
        for item in self.DepNgrams:
            ngram = self.prepare_SNgram (item, sentence, op)
            size = self.len_Ngram(ngram)               
            dic = d[size-self.min_size]                                                                                
            #Update the dictionary of SNgrams contained in the sample (frequency in the text)                         
            if (ngram in dic) > 0:#Update the frequency of the ngram                
                dic[ngram] += 1 #If the sn-gram exists in the dictionary, update its frequency                 
            else:
                dic[ngram] = 1 #Otherwise, add the sn-gram to the dictionary                            
    ######################
    
        
    def compound_sngrams(self, original, sentence):
        combinations   = []
        candidates    = []    
        
        for combination in original:                     #This cycle initializes the list of combinations and list of candidates 
            if len(combination[1]) > 0:               
                size = self.len_Ngram(self.prepare_SNgram(combination[0], sentence, -1))
                combinations.append([combination[0],combination[1],size])
            if combination[0][0] != sentence.root_idx:
                size = self.len_Ngram(self.prepare_SNgram(combination[0], sentence, -1))       
                candidates.append([combination[0],combination[1],size])
                                      
        while len(candidates) > 0:                        #In this cycle, select a sn-gram to be replaced in the rest of combinations            
            candidate = candidates.pop(0)
            value = candidate[0][0]                       #Get the first number of the first candidate sn-gram
                              
            for combination in combinations:
                if value in combination[1]:                
                                        
                    position = combination[0].index(value,2)#First get the position of the element          
          
                    sngram = copy.deepcopy(combination)
                    sngram[0].pop(position)                 #Delete the element in the sn-gram
                    sngram[0].insert(position,candidate[0]) #Insert the new part into the sn-gram
                    sngram[1].remove(value)                 #Update its list of posible combinations
                    sngram[2] = self.len_Ngram(self.prepare_SNgram(sngram[0], sentence, -1))#Obtain the size of the new sngram                                        

                    if (self.min_size > 0) and (self.max_size > 0):            #Case when the user specifies the max and min size of sn-grams                        
                        if sngram[2] in range(self.min_size, self.max_size+1):#Case when the sn-grams from the list substitution have to be inserted                            
                            self.DepNgrams.append(copy.deepcopy(sngram[0]))    #Update the list of all sn-grams
                        if sngram[2] < self.max_size:                                                    
                            if sngram[0][0] == sentence.root_idx:
                                if len(sngram[1]) > 0:
                                    combinations.append(copy.deepcopy(sngram))
                            else:
                                if len(sngram[1]) > 0:
                                    combinations.append(copy.deepcopy(sngram))
                                candidates.append(copy.deepcopy(sngram))
                    else:                                                 #Case when there is no restriction on the size of the sn-grams
                        self.DepNgrams.append(copy.deepcopy(sngram[0]))   #Update the list of all sn-grams                                                    
                        if sngram[0][0] == sentence.root_idx:
                            if len(sngram[1]) > 0:
                                combinations.append(copy.deepcopy(sngram))
                        else:
                            if len(sngram[1]) > 0:
                                combinations.append(copy.deepcopy(sngram))
                            candidates.append(copy.deepcopy(sngram))
                                               
    
    ######################              
    def get_subtrees (self, sentence):   # A function that gets all the possible subtrees in the tree
        unigrams     = []    #List of all possible unigrams    
        combinations = []    #List of all possible combinations of nodes and their children
        counter      = 0     #Counts the number of children inserted in the aux list
        aux          = []    #Auxiliar variable that contains the highest number of children allowed
        log          = set() #Variable that contains IDs of the nodes that have more children than it is allowed
      
        
        for node in self.subtrees:
            if self.max_num_children != 0:                      
                aux = []     #Reset the container for the next iteration
                counter = 0  #Reset the variable for the next iteration
                for child in sentence.children[node]:
                    
                    if self.min_size == 1 or self.min_size == 0 or self.max_size == 0: #This code obtains all unigrams of the sentence                
                        unigrams.append ([child])
                    
                    aux.append(child)
                    counter += 1
                    if counter > self.max_num_children:
                        aux.pop()
                        combinations.extend(self.get_next_combinations(node, aux, sentence))#We save new sn-grams in the global dictionary                    
                        counter = 0
                        aux = []
                        aux.append(child)
                        log.add(node)
                                                                
                if len(aux) > 0:                                                          #Analyze the rest of the children 
                    combinations.extend(self.get_next_combinations(node, aux, sentence))  #We save new sn-grams in the global dictionary
            
            else:            #In this case, there is no limitation on the number of children per node, so all the children are processed                
                combinations.extend(self.get_next_combinations(node, sentence.children[node], sentence))
                
                for child in sentence.children[node]:                
                    if self.min_size == 1 or self.min_size == 0 or self.max_size == 0:#This code obtains all unigrams of the sentence                
                        unigrams.append ([child])                                                                                                                                                        
                                                
        return (unigrams, combinations, log)
    
    
    ######################                  
    def get_next_combinations (self, value, children, sentence):
        ngram         = [] #Auxiliary variable for storing the sn-gram
        options       = [] #Auxiliary variable for storing the all the nodes that can be changed in a sn-gram
        combinations  = [] #Auxiliary variable for generating a combination
        lista         = [] #Auxiliary variable for all sn-grams during analysis of a sub-tree
            
        #Initialize the list of combinations    
        for p in range(0, len(children)):
            combinations.append (0)
      
        #Generate sn-grams    
        for r in range (1, len(children) + 1):                 
            for j in range (1, r + 1):
                combinations [j - 1] = j - 1

        #################### The first combination
            options = []
            ngram   = []
            ngram.append (value)
            ngram.append ("[")
            for z in range (0, r):
                ngram.append(children [combinations [z]])

                if children[combinations[z]] not in sentence.leaves:
                    options.append(children [combinations [z]])

                ngram.append (",")
            ngram.pop (len(ngram) - 1)          
            ngram.append ("]")            
            lista.append (copy.deepcopy([ngram,options]))

            ################### The rest
            top = self.Combination (len(children), r)
      
            for j in range(2, top + 1):
                m = r
                val_max = len(children)

                while combinations [m - 1] + 1 == val_max:
                    m       -= 1
                    val_max -= 1

                combinations [m - 1] += 1

                for k in range (m + 1, r + 1):
                    combinations [k - 1] = combinations [k - 2] + 1
            
                options = []
                ngram   = []
                ngram.append(value)
                ngram.append("[")                
                for z in range(0, r):
                    ngram.append (children [combinations [z]])

                    if children[combinations[z]] not in sentence.leaves:
                        options.append(children [combinations [z]])

                    ngram.append (",")
                ngram.pop (len(ngram) - 1)
                ngram.append ("]")
                lista.append (copy.deepcopy([ngram,options]))
              
        return (lista)          
    
    ######################                  
    def Combination (self, sz, r):
        if sz == r:
            numerator = 1
        else:
            numerator = sz
            for i in range (1, sz):
                numerator *= sz - i
        
            aux = r
            for i in range (1, r):
                aux *= r - i
        
            divisor = sz - r
            for i in range (1, sz - r):
                divisor *= sz - r - i
                
            numerator = numerator // (aux * divisor)
      
        return (numerator)

############
def process_one_sentence (lines, result, sent_num, f2, option2, dictionary, has_POS):
    print("Sentence " + str(sent_num))
    result.process_sentence (lines, option2, dictionary,has_POS)
    #result.print_sngrams()
    result.write_all_sn_grams (f2,str(sent_num))
    
    return sent_num + 1 

############### MAIN ################################
if __name__ == '__main__':
        
    encod = 'utf-8'   #'utf-8' or other encoding like '1252'
    dictionary = []    #variable that contains the stop words
    
    #Cases:
    #python MultiSNGrams_3.py input output
	#Note: Default values are min_size = 2; max_size = 7; max_num_children = 5, option = 0 (word sn-grams), option2 = -1 (not prune)    
    #python MultiSNGrams_3.py input output dictionary
	#Note: Default values are min_size = 2; max_size = 7; max_num_children = 5, option = 0 (word sn-grams), option2 = -1 (not prune)    
	#python MultiSNGrams_3.py input output min_size max_size max_num_children option
    #python MultiSNGrams_3.py input output min_size max_size max_num_children option option2 dictionary
    
    if len(sys.argv) < 3:
        print("Usage with at least two parameters:")
        print("python SNGrams3.py input output")
        exit(1)
    elif len(sys.argv) > 9:
        print("Usage with at most eight parameters:")
        print("python SNGrams3.py input output dictionary min_size max_size max_num_children option")
        exit(1)
    elif len(sys.argv) not in [3,4,7,9]:
        print("Mising parameters")
        exit(1)        

    input_file      = sys.argv[1]
    output_file     = sys.argv[2]

    #############These are parameters of configuration for the class BiSNgrams    
    min_size         = 2    #These are the parameters of configuration for the class BiSNgrams 
    max_size         = 7
    max_num_children = 5
    option           = 0    #Type of sn-grams to be obtained: 0 for WORD sn-grams; 1 for sn-grams of SR Tags; 2 for POS tags; 3 for all types of sngrams        
                            #option = 0    for sngrams of words
                            #option = 1    for sngrams of SR tags
                            #option = 2    for sngrams of POS tags        
                            #option = 3    for Word/SR SNgrams
                            #option = 4    for Word/POS SNgrams
                            #option = 5    for SR/Word SNgrams
                            #option = 6    for SR/POS SNgrams
                            #option = 7    for POS/Word SNgrams
                            #option = 8    for POS/SR SNgrams
    valid_options    = []   #List of valid options according with the format of input text
    has_POS          = 1    #Value of 1 indicates the input contains POS tags otherwise value of 0. Default value is 1    
    option2          = -1   #Valid only for prune trees: 1 for no stopwords; 2 for only stopwords
    
                            
    if len(sys.argv) == 4:                
        dictionary_file = sys.argv[3]
        option2 = 1
        try:
            print(dictionary_file) 
            f3 = codecs.open (dictionary_file, "rU", encoding = encod)  #b - Binary, for Unix line endings
            for item in f3.readlines():
                item = item.rstrip()                                
                dictionary.append(item)
                                    
            f3.close()
            if len(dictionary) == 0:
                print("ERROR: Empty dictionary")
                exit(1)
                
        except IOError as e:
            print(dictionary_file + "I/O error({0}): {1}".format(e.errno, e.strerror))
            exit(1)

    if len(sys.argv) == 7:
        min_size         = int(sys.argv[3])    #These are the parameters of configuration for the class BiSNgrams 
        max_size         = int(sys.argv[4])
        max_num_children = int(sys.argv[5])
        option           = int(sys.argv[6])    #value 0: for sn-grams of words; 
                                               #value 1: for sn-grams of sr tags; 
                                               #value 2: for sn-grams of words and sr tags (equal to call with option 0 and then with option 1)
            
    if len(sys.argv) == 9:
        min_size         = int(sys.argv[3])    #These are the parameters of configuration for the class BiSNgrams 
        max_size         = int(sys.argv[4])
        max_num_children = int(sys.argv[5])
        option           = int(sys.argv[6])    
        option2          = int(sys.argv[7])    #value for the kind of prune of the tree
        dictionary_file  = str(sys.argv[8])    #path of the stopwords dictionary
        try:
            print(dictionary_file) 
            f3 = codecs.open (dictionary_file, "rU", encoding = encod)  #b - Binary, for Unix line endings
            for item in f3.readlines():
                item = item.rstrip()                                
                dictionary.append(item)
                                    
            f3.close()
            if len(dictionary) == 0:
                print("ERROR: Empty dictionary")
                exit(1)
                
        except IOError as e:
            print(dictionary_file + "I/O error({0}): {1}".format(e.errno, e.strerror))
            exit(1)
                                                                           
    try:
        f1 = codecs.open (input_file,  "rU", encoding = encod)
        #**Read the input file and identify the format (includes POS tags or not)
        first_ln = f1.readlines()[0]
        #print first_ln        
        m = re.search('-[0-9]*\)', first_ln)
        #patron = re.compile(r'*[0-200])$')        
        if m:#Case where there are NOT POS tags
            #print "No POS"
            valid_options    = [0,1,3,5]
            has_POS = 0
            if option not in valid_options:
                print("ERROR: The selected option requieres POS tags but the input file does not contain POS tags")
                print("Select one of the following options")
                print("option = 0    for sngrams of words")
                print("option = 1    for sngrams of SR tags")                                    
                print("option = 3    for Word/SR SNgrams")                
                print("option = 5    for SR/Word SNgrams")                            
                exit(1)                            
        #else:#Case where there are POS tags
            #print "Si POS"        
        f1.close()
        f1 = codecs.open (input_file,  "rU", encoding = encod)
    except IOError as e:
        print(input_file + "I/O error({0}): {1}".format(e.errno, e.strerror))
        exit(1)
        
    try:
        f2 = codecs.open (output_file, "wb", encoding = encod)  #b - Binary, for Unix line endings
    except IOError as e:
        print(output_file + "I/O error({0}): {1}".format(e.errno, e.strerror))
        exit(1)
    

    sent_num = 1;
    result = BiSNgrams(min_size, max_size, max_num_children, option)
    lines  = []
    
    ###########Process the input file        
    if has_POS == 1: #Case where the POS tags are included in the input file
        print("Case with POS tags")
        flag = 0 #Auxiliar variable that helps to parse the text    
        for ln in f1.readlines():            
            #if (not ln) or (ln == ""):
            #    break;
            ln = ln.strip()
            if ln == "":#Var flag counts the empty lines in the text
                flag+=1
            else:
                lines.append(ln)                    
            if flag == 2:                
                if len (lines) > 0:
                    sent_num = process_one_sentence (lines, result, sent_num, f2, option2, dictionary, has_POS)                
                    del lines [:]
                    flag = 0        
        ######################
    else: #Case where there are NO POS tags in the input file
        print("Case with no POS tags detected")
        while True :
            ln = f1.readline ()
            #print ln
            if (not ln) or (ln == ""):
                break;
            
            ln = ln.strip()
            if ln == "":   #Sentences are separated by EMPTY line
                if len (lines) > min_size:
                    sent_num = process_one_sentence (lines, result, sent_num, f2, option2, dictionary, has_POS)
                    del lines [:]
            else:
                lines.append (ln)        
    #########################
    #print lines
    if len(lines) > 0: #Last piece in previous (while)
        sent_num = process_one_sentence (lines, result, sent_num, f2, option2, dictionary, has_POS)
            
    f1.close ()
    f2.close ()
           
    print("Done.")