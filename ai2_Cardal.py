'''
Kaggle "The Allen AI Science Challenge" competition
Oct 2015 - Feb 2016
Kaggle username: "Cardal" 
'''
from sklearn import preprocessing, grid_search, cross_validation, ensemble, metrics, linear_model, neighbors, svm, kernel_ridge, cross_decomposition
from sklearn.metrics import roc_curve, auc
from sklearn.feature_extraction import DictVectorizer

import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import EnglishStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn

import cPickle
import pandas as pd
import numpy as np 
import urllib2
import scipy
import scipy.optimize
import time
import sys
import csv
import re
import gc
import os

# This condition is here since I don't have PyLucene on my Windows system
if (len(sys.argv) >= 3) and (sys.argv[1] == 'prep') and (int(sys.argv[2]) >= 21):
    import lucene
    from java.io import File, StringReader
    from org.apache.lucene.analysis.core import WhitespaceAnalyzer
    from org.apache.lucene.analysis.miscellaneous import LimitTokenCountAnalyzer
    from org.apache.lucene.analysis.standard import StandardAnalyzer
    from org.apache.lucene.document import Document, Field, StoredField, StringField, TextField
    from org.apache.lucene.search.similarities import BM25Similarity 
    from org.apache.lucene.index import IndexWriter, IndexWriterConfig, DirectoryReader, MultiFields, Term
    from org.apache.lucene.queryparser.classic import MultiFieldQueryParser, QueryParser
    from org.apache.lucene.search import BooleanClause, IndexSearcher, TermQuery
    from org.apache.lucene.store import MMapDirectory, SimpleFSDirectory
    from org.apache.lucene.util import BytesRefIterator, Version



#################################################################################################
# I/O functions
#################################################################################################

def read_input_file(base_dir, filename, max_rows=999999999, use_cols=None, index_col=0, sep=','):
    '''
    Read an input file
    '''
#     print '=> Reading input file %s' % filename
    dataf = pd.read_table('%s/%s' % (base_dir, filename), index_col=index_col, nrows=max_rows, sep=sep)
    if 'correctAnswer' in dataf.columns:
        dataf = dataf[[(ca  in ['A','B','C','D']) for ca in dataf['correctAnswer']]] 
    dataf['ID'] = dataf.index
    return dataf

def save_to_pkl(filename, data):
    with open(filename, "wb") as f:
        cPickle.dump(data, f, cPickle.HIGHEST_PROTOCOL)

def load_from_pkl(filename):
    if not os.path.exists(filename):
        return None
    with open(filename, "rb") as f:
        data = cPickle.load(f)
    return data

def create_dirs(dirs):
    '''
    Make sure the given directories exist. If not, create them
    '''
    for dir in dirs:
        if not os.path.exists(dir):
            print 'Creating directory %s' % dir
            os.mkdir(dir)

def save_submission(submission_filename, preds):
    out_df = pd.DataFrame({'id': preds.keys(), 'correctAnswer': preds.values()})
    out_df = out_df.set_index('id')
    out_df.to_csv(submission_filename)
    print 'Saved submission file: %s' % submission_filename
    

#################################################################################################
# Stat utilities
#################################################################################################

def normalize_scores(scores, num_scores_per_row, pow=1, laplace=1E-100):
    '''
    Normalize the scores of each row. Use pow=0 for ranks.
    '''
    assert (num_scores_per_row > 0) and (laplace > 0)
    num_rows = len(scores) / num_scores_per_row
    assert num_rows*num_scores_per_row == len(scores)
    scores_mat = np.array(scores).reshape((num_rows, num_scores_per_row))
    if pow == 0:
        # Compute ranks    
        ranks = np.ones((num_rows, num_scores_per_row)) * (-1)
        for r in range(num_rows):
            ranks[r] = scipy.stats.rankdata(scores_mat[r], 'average') / num_scores_per_row 
        assert np.all(ranks != -1)
        normed = ranks
    else:
        # Normalize using given power
        scores_mat += laplace # to handle cases where all scores are 0, and also as a Laplace correction
        assert np.all(scores_mat > 0)
        normed = (scores_mat ** pow) / np.array([np.sum(scores_mat ** pow, axis=1)]).transpose()
    return normed.flatten()


#################################################################################################
# Parsers
#################################################################################################
                
class WordParser(object):
    '''
    WordParser - base class for parsers
    '''
    def __init__(self, min_word_length=2, max_word_length=25, ascii_conversion=True):
        self.min_word_length = min_word_length
        self.max_word_length = max_word_length
        self.ascii_conversion = ascii_conversion
    def filter_words_by_length(self, words):
        return [word for word in words if len(word)>=self.min_word_length and len(word)<=self.max_word_length]
    def convert_ascii(self, text):
        if self.ascii_conversion:
            return AsciiConvertor.convert(text)
        return text
    def parse(self, text, calc_weights=False):
        if calc_weights:
            return text, {}
        else:
            return text
    
class NltkTokenParser(WordParser):
    '''
    NLTK parser, supports tags (noun, verb, etc.)
    '''
    # See nltk.help.upenn_tagset('.*')
    TAG_TO_POS = {'NN': wn.NOUN, 'NNP': wn.NOUN, 'NNPS': wn.NOUN, 'NNS': wn.NOUN, 
                  'VB': wn.VERB, 'VBD': wn.VERB, 'VBG' : wn.VERB, 'VBN': wn.VERB, 'VBP': wn.VERB, 'VBZ': wn.VERB,
                  'RB': wn.ADV,  'RBR': wn.ADV , 'RBS' : wn.ADV , 'RP' : wn.ADV, 
                  'JJ': wn.ADJ , 'JJR': wn.ADJ , 'JJS' : wn.ADJ }

    def __init__(self, min_word_length=2, word_func=None, tolower=True, ascii_conversion=True, tuples=[1], ignore_special_words=True,
                 tag_weight_func=lambda tag: 1.0, word_func_requires_tag=True):
        self.word_func = word_func
        self.tolower = tolower
        self.tuples = tuples
        self.ignore_special_words = ignore_special_words
        self.tag_weight_func = tag_weight_func
        self.word_func_requires_tag = word_func_requires_tag
        assert set([1]).issuperset(self.tuples)
        WordParser.__init__(self, min_word_length=min_word_length, ascii_conversion=ascii_conversion)

    def parse(self, text, calc_weights=False):
        text = self.convert_ascii(text)
        tokens = nltk.word_tokenize(text)
        if calc_weights or self.word_func_requires_tag:
            tagged_tokens = nltk.pos_tag(tokens)
        else: # save time - don't use tags
            tagged_tokens = zip(tokens,[None]*len(tokens))
        ##tagged_tokens = nltk.pos_tag(tokens)
        words, weights, self.tags = [], [], []
        for word,tag in tagged_tokens:
            if len(word)>=self.min_word_length and len(word)<=self.max_word_length:
                words.append(word.lower() if self.tolower else word)
                weights.append(self.tag_weight_func(tag) if calc_weights else 0) 
                self.tags.append(tag)
        self.word_weights = {}
        # Filter special words
        if self.ignore_special_words:
            filtered = np.array([SpecialWords.filter1(word) for word in words])
            if np.all(filtered == False): # we're about to filter all the words -> instead, don't filter anything
                filtered = [True]*len(words)
        else:
            filtered = [True]*len(words) # no filtering

        if self.word_func is not None:
            fwords = []
            for word,wgt,fltr,tag in zip(words, weights, filtered, self.tags):
                if fltr:
                    try:
                        fword = str(self.word_func(word, NltkTokenParser.TAG_TO_POS.get(tag,None)))
                    except UnicodeDecodeError:
                        fword = word
#                     fword = self._apply_word_func(word, tag)
                    if type(fword)==list:
                        fwords += fword
                        if calc_weights:
                            for fw in fword:
                                self.word_weights[fw] = np.max([self.word_weights.get(fw,-1.0), wgt])
                    else:
                        fwords.append(fword)
                        if calc_weights:
                            self.word_weights[fword] = np.max([self.word_weights.get(fword,-1.0), wgt])
            words = fwords
        else:
            fwords = []
            for word,wgt,fltr in zip(words, weights, filtered):
                if fltr:
                    fwords.append(word)
                    if calc_weights:
                        self.word_weights[word] = np.max([self.word_weights.get(word,-1.0), wgt])
            words = fwords
        ret_words = []
        if 1 in self.tuples:
            ret_words += words
        if calc_weights:
            return ret_words, self.word_weights
        else:
            return ret_words

class SimpleWordParser(WordParser):
    '''
    SimpleWordParser - supports tuples
    '''
    def __init__(self, stop_regexp='[\-\+\*_\.\:\,\;\?\!\'\"\`\\\/\)\]\}]+ | [\*\:\;\'\"\`\(\[\{]+|[ \t\r\n\?]', 
                 min_word_length=2, word_func=None, tolower=True, ascii_conversion=True, ignore_special_words=True,
                 split_words_regexp=None, # for additional splitting of words, provided that all parts are longer than min_word_length, eg, split_words_regexp='[\-\+\*\/\,\;\:\(\)]' 
                 tuples=[1]):
        self.stop_regexp = re.compile(stop_regexp)
        self.word_func = word_func
        self.tolower = tolower
        self.ignore_special_words = ignore_special_words
        self.split_words_regexp = None if split_words_regexp is None else re.compile(split_words_regexp)
        self.tuples = tuples
        assert set([1,2,3,4]).issuperset(self.tuples)
        WordParser.__init__(self, min_word_length=min_word_length, ascii_conversion=ascii_conversion)
        
    def parse(self, text, calc_weights=False):
        if self.tolower:
            text = text.lower()
        text = ' ' + text.strip() + ' ' # add ' ' at the beginning and at the end so that, eg, a '.' at the end of the text will be removed, and "'''" at the beginning will be removed 
        text = self.convert_ascii(text)
        words = re.split(self.stop_regexp, text)
        if self.split_words_regexp is not None:
            swords = []
            for word in words:
                w_words = re.split(self.split_words_regexp, word)
                if len(w_words) == 1:
                    swords.append(w_words[0])
                else:
                    if np.all([len(w)>=self.min_word_length for w in w_words]):
                        swords += w_words
                    else:
                        swords.append(word) # don't split - some parts are too short
            words = swords
        if self.ignore_special_words:
            words = SpecialWords.filter(words)
        if self.word_func is not None:
            fwords = []
            for word in words:
                try:
                    fword = str(self.word_func(word))
                except UnicodeDecodeError:
                    fword = word
                fwords.append(fword)
            words = fwords
        words = self.filter_words_by_length(words)
        ret_words = []
        if 1 in self.tuples:
            ret_words += words
        if 2 in self.tuples:
            ret_words += ['%s %s'%(words[i],words[i+1]) for i in range(len(words)-1)]
        if 3 in self.tuples:
            ret_words += ['%s %s %s'%(words[i],words[i+1],words[i+2]) for i in range(len(words)-2)]
            if 2 in self.tuples:
                ret_words += ['%s %s'%(words[i],words[i+2]) for i in range(len(words)-2)]
        if 4 in self.tuples:
            ret_words += ['%s %s %s %s'%(words[i],words[i+1],words[i+2],words[i+3]) for i in range(len(words)-3)]
            if 3 in self.tuples:
                ret_words += ['%s %s %s'%(words[i],words[i+2],words[i+3]) for i in range(len(words)-3)]
                ret_words += ['%s %s %s'%(words[i],words[i+1],words[i+3]) for i in range(len(words)-3)]
            if 2 in self.tuples:
                ret_words += ['%s %s'%(words[i],words[i+3]) for i in range(len(words)-3)]
                if 3 not in self.tuples:
                    ret_words += ['%s %s'%(words[i],words[i+2]) for i in range(len(words)-2)]
        if calc_weights:
            return ret_words, {}
        else:
            return ret_words
  

#################################################################################################
# Corpus preparation
#################################################################################################

class CorpusReader(object):            
    '''
    CorpusReader - base class for corpus readers
    '''
    PAGE_NAME_PREFIX    = '<PAGE>'
    SECTION_NAME_PREFIX = '<SECTION>'

    PART_NAMES_IGNORE = set(['introduction', 'summary'])
        
    def __init__(self, min_chars_per_line=50, min_words_per_section=50, debug_flag=False):
        self.min_chars_per_line = min_chars_per_line
        self.min_words_per_section = min_words_per_section
        self.debug_flag = debug_flag
        self._reset(outfile=None, stop_words=None, pos_words=None, page_name_word_sets=None, corpus_words=None, 
                    min_pos_words_in_page_name=-1, min_pos_words_in_section=-1, 
                    use_all_pages_match_pos_word=False, use_all_pages_match_sets=False, always_use_first_section=False,
                    action=None)
        self.sections_to_use = None
        
    def _reset(self, outfile, stop_words, pos_words, page_name_word_sets, corpus_words, 
               min_pos_words_in_page_name, min_pos_words_in_section, use_all_pages_match_pos_word, use_all_pages_match_sets, always_use_first_section,
               action):
        if (stop_words is not None) and (pos_words is not None) and (len(stop_words.intersection(pos_words)) > 0):
            print 'Stop words contain pos words - removing from pos words: %s' % stop_words.intersection(pos_words)
            pos_words = pos_words.difference(stop_words)
        assert (stop_words is None) or len(stop_words.intersection(pos_words))==0
        self.outfile = outfile
        self.stop_words, self.pos_words, self.page_name_word_sets, self.corpus_words = stop_words, pos_words, page_name_word_sets, corpus_words
        self.min_pos_words_in_page_name, self.min_pos_words_in_section = min_pos_words_in_page_name, min_pos_words_in_section
        self.use_all_pages_match_pos_word, self.use_all_pages_match_sets = use_all_pages_match_pos_word, use_all_pages_match_sets
        self.always_use_first_section = always_use_first_section
        self.action = action
        self._outf, self._locdic = None, None
        self.num_pages, self.num_sections = 0, 0
        self.num_section_action = 0
        self.pages_in_corpus = set() # names of pages that are actually in the corpus 

    def set_sections_to_use(self, sections_to_use):
        if sections_to_use is None:
            self.sections_to_use = sections_to_use
        else:
            self.sections_to_use = set(sections_to_use)

    def _start_action(self):
        self.pages_in_corpus = set()
        if self.action == 'write':
            self._outf = open(self.outfile,'w')
        elif self.action == 'locdic':
            self._locdic = LocationDictionary(save_locations=False, doc_name_weight=0)
        else:
            raise ValueError('Unsupported action (%s)' % self.action)
        
    def _end_action(self):
        if self._outf is not None:
            self._outf.close()
            self._outf = None
        # Write pages_in_corpus
        if self.action == 'write':
            save_to_pkl('%s.pages.pkl' % self.outfile, self.pages_in_corpus)
        gc.collect()

    @staticmethod
    def part_name_from_words(words, number):
        if (len(words) == 1) and (words[0] in CorpusReader.PART_NAMES_IGNORE):
            words = []
        return '%s __%d' % (' '.join(words), number)
    
    @staticmethod
    def words_from_part_name(part_name):
        words = part_name.split(' ')
        assert words[-1].startswith('__')
        return words[:-1]
        
    def _add_page(self, page_name, page_name_words):
#         print 'Adding page "%s"' % page_name
        self.num_pages += 1
#         if page_name != 'Hayashi track': return
        if self.action == 'write':
#             print 'writing page %s' % page_name
            self._outf.write('%s%s\n' % (CorpusReader.PAGE_NAME_PREFIX, CorpusReader.part_name_from_words(page_name_words, self.num_pages)))

    def _check_page_name(self, page_name, page_name_words):
        '''
        Returns True if page should be used; False if it should be skipped
        '''
        if self.use_all_pages_match_sets and (tuple(sorted(page_name_words)) in self.page_name_word_sets):
#             if self.debug_flag:
#                 print 'Page "%s" matches page_name_word_sets -> using page' % page_name
            return True
        num_pos_words_in_page_name = len(set(page_name_words).intersection(self.pos_words))
        if self.use_all_pages_match_pos_word and (num_pos_words_in_page_name > 0):
#             if self.debug_flag:
#                 print 'Page "%s" matches pos word(s) (%s) -> using page' % (page_name, set(page_name_words).intersection(self.pos_words))
            return True
        if num_pos_words_in_page_name >= self.min_pos_words_in_page_name:
#             if self.debug_flag:
#                 print 'Page "%s" has enough pos words (%s -> %d) -> using page' % (page_name, ','.join(page_name_words), num_pos_words_in_page_name)
            return True
        return False
        
    def _add_section(self, page_name, page_name_words, section_name, section_name_words, section_number, section_words):
        '''
        Returns 1 if the section was added, 0 otherwise
        '''
        self.num_sections += 1
#         if page_name != 'Hayashi track': return
        if ((not self.always_use_first_section) or (section_number > 1)) and (len(section_words) < self.min_words_per_section):
            if self.debug_flag:
                print 'section "%s" (%d) too short (%d words)' % (section_name, section_number, len(section_words))
            return 0
        if not self._check_page_name(page_name, page_name_words):
            return 0
        if (self.sections_to_use is not None) and (section_name not in self.sections_to_use):
            if self.debug_flag:
                print 'section "%s" (%d) not in sections_to_use set' % (section_name, section_number)
            return 0          
        if self.stop_words is not None:
            section_words = [w for w in section_words if not w in self.stop_words]
        num_pos_words_in_section = len(set(section_words).intersection(self.pos_words))
        if ((not self.always_use_first_section) or (section_number > 1)) and (num_pos_words_in_section < self.min_pos_words_in_section):
            if self.debug_flag:
                print 'section "%s" (%d) has too few pos words (%d)' % (section_name, section_number, num_pos_words_in_section)
            return 0
        if self.debug_flag:
            print 'page "%s" section "%s" (%d) has %d pos words (total %d words)' % (page_name, section_name, section_number, num_pos_words_in_section, len(section_words))
        if self.corpus_words is not None:
            section_words = [w for w in section_words if w in self.corpus_words]
        if self.action == 'write':
            self._outf.write('%s%s\n' % (CorpusReader.SECTION_NAME_PREFIX, CorpusReader.part_name_from_words(section_name_words, section_number)))
#             self._outf.write('%d pos words: %s\n' % (num_pos_words_in_section, set(section_words).intersection(self.pos_words))) # DEBUG...
            self._outf.write('%s\n' % ' '.join(section_words))
            self.num_section_action += 1
        elif self.action == 'locdic':
            self._locdic.add_words('%s/%s' % (CorpusReader.part_name_from_words(page_name_words, self.num_pages), 
                                              CorpusReader.part_name_from_words(section_name_words, section_number)), 
                                   page_name_words + section_name_words, section_words)
            self.num_section_action += 1
        self.pages_in_corpus.add(page_name)
        return 1
    
    @staticmethod
    def build_locdic_from_outfile(filename, parser=SimpleWordParser(),
                                  min_word_docs_frac=0, max_word_docs_frac=0.2, min_word_count_frac=0, max_word_count_frac=0.01,
                                  doc_name_weight=0):
        locdic = LocationDictionary(save_locations=False, doc_name_weight=doc_name_weight)
        locdic.set_search_word_filter(min_word_docs_frac=min_word_docs_frac, max_word_docs_frac=max_word_docs_frac, 
                                      min_word_count_frac=min_word_count_frac, max_word_count_frac=max_word_count_frac)
        num_pages, num_sections = 0, 0
        page_name, section_name = None, None
        num_lines = 0
        if type(filename)==str:
            assert file is not None
            filenames = [filename]
        else:
            assert not np.any([(fn is None) for fn in filename])
            filenames = filename # list of file names
        for ifname,fname in enumerate(filenames):
            print 'Building locdic from file #%d: %s' % (ifname, fname)
            with open(fname,'rt') as infile:
                for text in infile:
    #                 print '%s' % text
                    if len(text)==0: 
                        print 'Reached EOF'
                        break # EOF
                    if text.startswith(CorpusReader.PAGE_NAME_PREFIX):
                        page_name = text[len(CorpusReader.PAGE_NAME_PREFIX):].strip()
                        section_name = None
                        num_pages += 1
                    elif text.startswith(CorpusReader.SECTION_NAME_PREFIX):
                        section_name = text[len(CorpusReader.SECTION_NAME_PREFIX):].strip()
                        num_sections += 1
                    else:
                        assert (page_name is not None) and (section_name is not None)
                        #section_words = text.split(' ')
                        section_words = parser.parse(text, calc_weights=False) #True)
                        if False:
                            print 'Adding words: %s (weights: %s)' % (section_words, weights)
                        locdic.add_words('F%d/%s/%s' % (ifname, page_name, section_name), CorpusReader.words_from_part_name(page_name) + CorpusReader.words_from_part_name(section_name), 
                                         section_words)
                    num_lines += 1
                    if num_lines % 100000 == 0:
                        print ' read %d lines: %d pages, %d sections -> %d words' % (num_lines, num_pages, num_sections, len(locdic.word_ids))
    #                     skjdkjjkkj()
        return locdic


class WikiReader(CorpusReader):
    '''
    WikiReader - read a Wiki corpus
    '''
    # List of words that appear in >10% of (a sample of) Wiki docs, after manual removal of some words
    # List does not include the NLTK stop words 
    WIKI_STOP_WORDS = set(['also', 'one', 'known', 'new', 'two', 'may', 'part', 'used', 'many', 'made', 'since',
                           'including', 'later', 'well', 'became', 'called', 'three', 'named', 'second', 'several', 'early',
                           'often', 'however', 'best', 'use', 'although', 'within'])
    
    # Regexp for page names to ignore    
    IGNORE_PAGES = re.compile('(\S[\:]\S)|([\(]disambiguation[\)])|(Wikipedia)')

    NEW_PAGE_SUBSTR     , NEW_PAGE_RE      = '<title>'    , re.compile('<title>(.*)</title>')
    CATEGORY_SUBSTR     , CATEGORY_RE      = '[[Category:', re.compile('[\[]+Category[\:]([^\]\|]*)([\|][^\]\|]*)*[\]]+') # eg, "[[Category:Social theories]]"
    CATEGORY_SUBSTR2    , CATEGORY_RE2     = '{{'         , re.compile('[\{][\{]([^\}]+)[\}][\}]\s*<[\/]text>') # eg, "{{music-stub}}</text>" (in SimpleWiki)
    PAGE_REDIRECT_SUBSTR, PAGE_REDIRECT_RE = '<redirect'  , re.compile('\s*<redirect') # eg, " <redirect title="Light pollution" />" (in SimpleWiki)
    NEW_SECTION_SUBSTR  , NEW_SECTION_RE   = '=='         , re.compile('\s*=[=]+([^=]*)=[=]+\s*')

    CATEGORY_PAGE_NAME = re.compile('Category[\:](.*)')

    # Text replacements
    RE_REMOVALS0 = [(sr,re.compile(rr)) for sr,rr in [('[', '[\[]+[^\]\:]+[\:][^\]]+[\]]+'),
                                                      ('{{', '[\{][\{][^\}]+[\}][\}]')]] 
    BLANKS_CONVERSIONS = ['&quot;', '&amp;nbsp;', '&amp;', '&nbsp;', '|']
    OTHER_CONVERSIONS = [('&lt;', '<'), ('&gt;', '>'),
                         (chr(195)+chr(164), 'a'), (chr(195)+chr(167), 'c'), (chr(195)+chr(169), 'e'), (chr(195)+chr(184), 'o'), (chr(197)+chr(143), 'o'),
                         (chr(194)+chr(188), '1/4'),
                         (chr(194)+chr(183), '*'), 
                         (chr(226)+chr(128)+chr(152), "'"  ), (chr(226)+chr(128)+chr(153), "'"  ),
                         (chr(226)+chr(128)+chr(156), '"'  ), (chr(226)+chr(128)+chr(157), '"'  ), 
                         (chr(226)+chr(128)+chr(147), ' - '), (chr(226)+chr(128)+chr(148), ' - '), (chr(226)+chr(136)+chr(146), '-')]
    RE_REMOVALS = [(sr,re.compile(rr)) for sr,rr in [#('{{', '[\{][\{][^\}]+[\}][\}]'),
                                                     #('&lt;!--', '&lt;![\-][\-][^\&]*(&lt;[^\&]*&gt;[^\&]*)*[\-][\-]&gt;'), 
                                                     #('&lt;', '&lt;[^\&]*&gt;'),
                                                     ('<ref', '<ref[^\>]*>[^\<\>]+<[\/]ref>'), 
                                                     #('[http:', '[\[]http:[^\]]+[\]]'),
                                                     ('<', '<[^\>]*>'),
                                                     ("''", "[']+"),
                                                     ('wikt:', 'wikt:\S+')]]
    STR_REMOVALS = ['[[',']]', '#*:','#*']  

    def __init__(self, wiki_type, debug_flag=False):
        assert wiki_type in ['wiki', 'simplewiki', 'wiktionary', 'wikibooks', 'wikiversity']
        self.wiki_type = wiki_type
        if self.wiki_type == 'wiki':
            min_chars_per_line, min_words_per_section = 50, 50
        elif self.wiki_type == 'simplewiki':
            min_chars_per_line, min_words_per_section = 1, 1
        elif self.wiki_type == 'wiktionary':
            min_chars_per_line, min_words_per_section = 1, 3
        elif self.wiki_type == 'wikibooks':
            min_chars_per_line, min_words_per_section = 1, 10
        elif self.wiki_type == 'wikiversity':
            min_chars_per_line, min_words_per_section = 1, 3
        CorpusReader.__init__(self, min_chars_per_line=min_chars_per_line, min_words_per_section=min_words_per_section, debug_flag=debug_flag)
        if self.wiki_type == 'wiktionary':
            self.set_sections_to_use(['Noun'])
        
    def search_categories(self, category, target_categories, max_depth=5):
        checked_categories = set()
        categories_to_check = set([(category, 0)])
        while len(categories_to_check) > 0:
            cat, depth = categories_to_check.pop()
#             print 'Checking %s' % cat
            if cat in target_categories:
                return depth
            checked_categories.add(cat)
            if self.parent_categories.has_key(cat) and depth+1 <= max_depth:
#                 if len(self.parent_categories[cat].difference(checked_categories).intersection(['Medicine','Medical specialties','Psychiatry','Neurophysiology','Mind'])) > 0:
#                     print '%s -> adding %s' % (cat, sorted(self.parent_categories[cat].difference(checked_categories)))
                categories_to_check.update([(c,depth+1) for c in self.parent_categories[cat].difference(checked_categories)])
        return -1
    
    def read_sub_categories(self, wikifile, max_read_lines=None):
        print '=> Reading sub categories'
        self.all_categories, self.parent_categories = set(), {}
        category = None
        num_lines = 0
        with open(wikifile,'rt') as infile:
            for text in infile:
                if len(text)==0: 
                    print 'Reached EOF'
                    break # EOF
                num_lines += 1
                if (max_read_lines is not None) and (num_lines > max_read_lines):
                    break
                if num_lines % 1000000 == 0:
                    print 'Read %d lines, total of %d categories...' % (num_lines, len(self.all_categories))
                    gc.collect()
                if WikiReader.NEW_PAGE_SUBSTR in text:
                    new_page = re.findall(WikiReader.NEW_PAGE_RE, text)
                    if len(new_page)>0:
                        assert len(new_page)==1
                        page_name = new_page[0]
#                         print 'page "%s"' % page_name
                        cmatch = re.match(WikiReader.CATEGORY_PAGE_NAME, page_name)
                        if cmatch is not None: 
                            category = cmatch.groups(0)[0].strip()
                            self.all_categories.add(category)
#                             if category == 'mac os software':
#                                 print '%s' % text
                            assert (not self.parent_categories.has_key(category)) #or (len(self.parent_categories[category])==0)
                            self.parent_categories[category] = set()
#                             print 'Category "%s"' % category
                        else:
#                             print ' no match'
                            category = None
                        continue
                if category is not None:
                    p_category = None
                    if WikiReader.CATEGORY_SUBSTR in text:
                        p_category = re.match(WikiReader.CATEGORY_RE, text)
                    elif WikiReader.CATEGORY_SUBSTR2 in text:
                        p_category = re.match(WikiReader.CATEGORY_RE2, text)
                    if p_category is not None:
                        assert len(p_category.groups())>=1
                        parent_name = p_category.groups(0)[0].strip()
                        self.all_categories.add(parent_name)
                        self.parent_categories[category].add(parent_name)

        print 'Read %d lines, %d categories' % (num_lines, len(self.all_categories))
       
    def read_pages_in_categories(self, wikifile, use_categories=None, max_read_lines=None):
        print '=> Reading pages in %s categories' %  ('ALL' if use_categories is None else len(use_categories))
        pages_in_categories = set()
        page_name, page_categories = None, set()
        num_lines = 0
        with open(wikifile,'rt') as infile:
            for text in infile:
                if len(text)==0: 
                    print 'Reached EOF'
                    break # EOF
                num_lines += 1
                if (max_read_lines is not None) and (num_lines > max_read_lines):
                    break
                if num_lines % 1000000 == 0:
                    print 'Read %d lines, %d pages so far...' % (num_lines, len(pages_in_categories))
                    gc.collect()
                if WikiReader.NEW_PAGE_SUBSTR in text:
                    new_page = re.findall(WikiReader.NEW_PAGE_RE, text)
                    if len(new_page)>0:
                        assert len(new_page)==1
                        # Check previous page
                        if len(page_categories) > 0:
                            assert page_name is not None
                            pages_in_categories.add(page_name)
                        page_name = new_page[0]
                        page_categories = set()
                        continue
                category = None
                if WikiReader.CATEGORY_SUBSTR in text:
                    category = re.match(WikiReader.CATEGORY_RE, text)
                elif WikiReader.CATEGORY_SUBSTR2 in text:
                    category = re.match(WikiReader.CATEGORY_RE2, text)
                if category is not None:
                    assert len(category.groups())>=1
                    assert page_name is not None
                    cat_name = category.groups(0)[0].strip()
                    if (re.search(WikiReader.IGNORE_PAGES, page_name) is None) and ((use_categories is None) or (cat_name in use_categories)):
#                             print 'Found category %s for page %s' % (cat_name, page_name)
                        page_categories.add(cat_name)
                        #self.all_categories.add(cat_name)

        # Check last page
        if len(page_categories) > 0:
            assert page_name is not None
            pages_in_categories.add(page_name)
            
        print 'Read %d lines, %d pages in %d categories' % (num_lines, len(pages_in_categories), len(use_categories))
        return pages_in_categories

    @staticmethod
    def text_replcaments(text):
        for sr,rr in WikiReader.RE_REMOVALS0:
            if sr in text:
                text = re.sub(rr, ' ', text)
        for bc in WikiReader.BLANKS_CONVERSIONS:
            text = text.replace(bc, ' ')
#         print '----------->'
#         print '%s' % text
        for oc,cc in WikiReader.OTHER_CONVERSIONS:
            text = text.replace(oc, cc)
        # DEAL WITH HIGH ASCIIs...
        if True:
            text = ''.join([(c if ord(c)<128 else ' ') for c in text])
        else:
            if len([c for c in text if ord(c)>=128])>0:
                print '%s' % ''.join([(c if ord(c)<128 else ' chr(%d) '%ord(c)) for c in text])
                assert False, 'chr > 127'
#         print '++++++'
#         print 'before re_removals: %s' % text
#         print '++++++'
        for sr,rr in WikiReader.RE_REMOVALS:
            if sr in text:
                text = re.sub(rr, ' ', text)
        for sr in WikiReader.STR_REMOVALS:
            text = text.replace(sr, '')
        return text

    def read(self, wikifile, outfile, use_pages=None, max_read_lines=None, 
             only_first_section_per_page=False, max_sections_per_page=99999999,
             parser = SimpleWordParser(tolower=True, ascii_conversion=True, ignore_special_words=False),
             stop_words=set(), pos_words=set(), 
             page_name_word_sets=None, corpus_words=None,
             min_pos_words_in_page_name=1, min_pos_words_in_section=5,
             use_all_pages_match_pos_word=False, use_all_pages_match_sets=False, always_use_first_section=False,
             action='write'):
        print '=> Reading Wiki corpus (%s)' % self.wiki_type
        if use_pages is not None:
            print 'Using set of %d pages' % len(use_pages)

        self._reset(outfile=outfile, stop_words=stop_words, pos_words=pos_words, page_name_word_sets=page_name_word_sets, corpus_words=corpus_words,
                    min_pos_words_in_page_name=min_pos_words_in_page_name, min_pos_words_in_section=min_pos_words_in_section,
                    use_all_pages_match_pos_word=use_all_pages_match_pos_word, use_all_pages_match_sets=use_all_pages_match_sets, 
                    always_use_first_section=always_use_first_section,
                    action=action)
    
        skip_lines_first_1char  = set(['\n']) #,'=','{','}','|','!',';','#']) #,' ','*'])
        skip_lines_first_6chars = set(['<media', '[[File', '[[Imag', '[[Cate'])
        content_lines_re = re.compile('^([^<])|([<]text)')
        if self.wiki_type == 'wiki':
            skip_lines_first_1char.update(['=','{','}','|','!',';','#',' ','*'])
            
        self._start_action()
        page_name, section_name, section_name_words, section_in_page = None, None, [], 0
        page_name_words, section_words, section_text = [], [], ''
        num_sections_added_in_page = 0
        skip_page = True
        start_time = time.time()
        num_lines = 0
        with open(wikifile,'rt') as infile:
            for text in infile:
#                 if not skip_page:
#                 print '%s' % text
#                 if len(re.findall(WikiReader.NEW_PAGE_RE, text)) > 0:
#                     print '======== %s' % text
#                 if re.match('\s*[\{][\{]([^\}]+)[\}][\}]\s*<[\/]text>', text) is not None:
#                     print text
#                 continue
                if len(text)==0: 
                    print 'Reached EOF'
                    break # EOF
                num_lines += 1
                if (max_read_lines is not None) and (num_lines > max_read_lines):
                    break
                if num_lines % 1000000 == 0:
                    print 'Read %d lines, %d pages, %d sections so far, %d sections actioned...' % (num_lines, self.num_pages, self.num_sections, self.num_section_action)
                    gc.collect()
                if WikiReader.NEW_PAGE_SUBSTR in text:
                    new_page = re.findall(WikiReader.NEW_PAGE_RE, text)
                    if len(new_page)>0:
                        assert len(new_page)==1
                        # Add last section from previous page
                        if (not skip_page) and ((not only_first_section_per_page) or (section_in_page == 1)) and (num_sections_added_in_page < max_sections_per_page):
#                             print '---------------------------------'
#                             print 'FULL: %s' % section_text
                            section_text = WikiReader.text_replcaments(section_text)
#                             print '---------------------------------'
#                             print 'GOT: %s' % section_text
#                             print '---------------------------------'
#                             print ''
                            section_words = parser.parse(section_text)
                            num_sections_added_in_page += self._add_section(page_name, page_name_words, section_name, section_name_words, section_in_page, section_words)
                        page_name = new_page[0]
                        page_name_words = parser.parse(page_name)
#                         print '------> new page: %s (%d)' % (page_name, self.num_pages)
                        section_in_page, section_name, section_name_words, section_words, section_text = 1, '', [], [], ''
                        num_sections_added_in_page = 0
                        skip_page = (re.search(WikiReader.IGNORE_PAGES, page_name) is not None) or ((use_pages is not None) and (page_name not in use_pages))
                        skip_page = skip_page or (not self._check_page_name(page_name, page_name_words))
#                         if ('light' not in page_name_words) or ('year' not in page_name_words):
#                             skip_page = True
#                         else:
#                             skip_page = False
#                             print 'not skipping page "%s": %s' % (page_name, page_name_words)
#                         if 'Wikipedia' in page_name:
#                             print 'page %s -> skip = %s' % (page_name, skip_page)
                        if not skip_page:
                            self._add_page(page_name, page_name_words)
                        continue
#                 if page_name != 'Hayashi track': continue
                if skip_page: continue
                if (section_in_page == 1) and (len(section_words) == 0) and (WikiReader.PAGE_REDIRECT_SUBSTR in text):
                    if re.match(WikiReader.PAGE_REDIRECT_RE, text):
#                         print 'Redirect -> skipping page "%s"' % page_name
                        skip_page = True
                if skip_page: continue
                if WikiReader.NEW_SECTION_SUBSTR in text:
                    new_section = re.match(WikiReader.NEW_SECTION_RE, text)
                    if new_section is not None:
                        assert len(new_section.groups())==1
                        # Add previous section 
                        if ((not only_first_section_per_page) or (section_in_page == 1)) and (num_sections_added_in_page < max_sections_per_page):
#                             print '---------------------------------'
#                             print 'FULL: %s' % section_text
                            section_text = WikiReader.text_replcaments(section_text)
#                             print '---------------------------------'
#                             print 'GOT: %s' % section_text
#                             print '---------------------------------'
#                             print ''
                            section_words = parser.parse(section_text)
                            num_sections_added_in_page += self._add_section(page_name, page_name_words, section_name, section_name_words, section_in_page, section_words)
                        section_in_page += 1 
                        section_name, section_words, section_text = new_section.groups(0)[0], [], ''
                        section_name = WikiReader.text_replcaments(section_name)
                        section_name_words = parser.parse(section_name)
#                         print '---> new section: %s (%d)' % (section_name, section_in_page)
                        continue
                if text[ 0] in skip_lines_first_1char: continue
                if text[:6] in skip_lines_first_6chars: continue
                if len(text) < self.min_chars_per_line: continue
                text = text.strip()
                if re.match(content_lines_re, text) is None: continue
                section_text += ' ' + text.strip() + ' '

        # Add last section
        if (not skip_page) and ((not only_first_section_per_page) or (section_in_page == 1)) and (num_sections_added_in_page < max_sections_per_page):
            section_text = WikiReader.text_replcaments(section_text)
            section_words = parser.parse(section_text)
            num_sections_added_in_page += self._add_section(page_name, page_name_words, section_name, section_name_words, section_in_page, section_words)
    
        end_time = time.time()
        print 'read_wiki total time = %.1f secs.' % (end_time-start_time)
        print 'Read %d lines, %d pages, %d sections; applied action on %d sections' % (num_lines, self.num_pages, self.num_sections, self.num_section_action)
        self._end_action()
        
        return self._locdic
        

class HtmlReader(CorpusReader):
    '''
    HtmlReader - read an HTML corpus
    '''
    RE_SUBSTITUTE = [re.compile(rk) for rk in ['<img [^>]*alt="([^\"]*)"[^>]*>']]
    RE_REMOVE     = [re.compile(rr) for rr in ['<a [^>]+>[^<]*<[\/]a>', '<a [^>]+>\s*<strong>[^<]*<[\/]strong>\s*<[\/]a>',
                                               '<[^>]*>', 'http[\:][^\"\>]+[\"\> ]']]
    RE_REPLACE = [(re.compile(rr),rs) for rr,rs in [
                    ('&#8216;', "'"), ('&#8217;', "'"), ('&#8220;', '"'), ('&#8221;', '"'), 
                    ('&quot;', '"'), ('&lsquo;', "'"), ('&rsquo;', "'"), ('&ldquo;', '"'), ('&rdquo;', '"'),
                    ('&#8206;', '.'), ('&#8195', '.'), ('&#8230;', ' ... '), 
                    ('&#8212;', ' - '), ('&#8211;', ' - '), ('&#8722;', ' - '),
                    ('&#8594;', ' - '), ('&#8592;', ' - '), ('&#8596;', ' - '), ('#8595;', ' - '), ('&#8593;', ' - '), # various arrows
                    ('&#8804;', ' '), # <=
                    ('&#8801;', ' '), # = with 3 lines
                    ('&#730;', ' degrees '),
                    ('&nbsp;', ' '), ('&deg;', ' degrees '), ('&#8203;', ''), ('&#9786;', ''),
                    ('&#38;', '&'), ('&#8226;', ' '), ('&#9702;', ' '), ('&#8729;', ' '), ('&#8227;', ' '), ('&#8259;', ' '), ('&#176;', ' degrees '), ('&#8734;', ' infinity '),
                    ('&#36;', '$'), ('&#8364;', ' euro '), ('&#163;', ' pound '), ('&#165;', ' yen '), ('&#162;', ' cent '),
                    ('&#169;', ' '), ('&#174;', ' '), ('&#8471;', ' '), ('&#8482;', ' '), ('&#8480;', ' '),
                    ('&#945;', 'alpha'), ('&#946;', 'beta'), ('&#947;', 'gamma'), ('&#948;', 'delta'), ('&#949;', 'epsilon'), ('&#950;', 'zeta'),
                    ('&#951;', 'eta'), ('&#952;', 'theta'), ('&#953;', 'iota'), ('&#954;', 'kappa'), ('&#955;', 'lambda'), ('&#956;', 'mu'), ('&#957;', 'nu'),
                    ('&#958;', 'xi'), ('&#959;', 'omicron'), ('&#960;', 'pi'), ('&#961;', 'rho'), ('&#963;', 'sigma'), ('&#964;', 'tau'), ('&#965;', 'upsilon'),
                    ('&#966;', 'phi'), ('&#967;', 'chi'), ('&#968;', 'psi'), ('&#969;', 'omega'), ('&#913;', 'Alpha'), ('&#914;', 'Beta'), ('&#915;', 'Gamma'),
                    ('&#916;', 'Delta'), ('&#917;', 'Epsilon'), ('&#918;', 'Zeta'), ('&#919;', 'Eta'), ('&#920;', 'Theta'), ('&#921;', 'Iota'), ('&#922;', 'Kappa'),
                    ('&#923;', 'Lambda'), ('&#924;', 'Mu'), ('&#925;', 'Nu'), ('&#926;', 'Xi'), ('&#927;', 'Omicron'), ('&#928;', 'Pi'), ('&#929;', 'Rho'),
                    ('&#931;', 'Sigma'), ('&#932;', 'Tau'), ('&#933;', 'Upsilon'), ('&#934;', 'Phi'), ('&#935;', 'Chi'), ('&#936;', 'Psi'), ('&#937;', 'Omega'),
                    ('&#186;', ' '), ('&#180;', '`'), 
                    ('&#189;', ' 1/2'), ('&#188;', '1/4'), ('&#190;', '3/4'),
                    ('&#181;', 'm'),  ('&#299;', 'i'), ('&#333;', 'o'), ('&#257;', 'a'), 
                    ('&#215;', '*'), ('&#8226;', '*'), ('&#183;', '*'), ('&#247;', '/')] +
                   [('&#%3d;'%x,'A') for x in range(192,199)] + [('&#%3d;'%x,'C') for x in range(199,200)] + [('&#%3d;'%x,'E') for x in range(200,204)] +
                   [('&#%3d;'%x,'I') for x in range(204,208)] + [('&#%3d;'%x,'D') for x in range(208,209)] + [('&#%3d;'%x,'N') for x in range(209,210)] +
                   [('&#%3d;'%x,'O') for x in range(210,215)+[216]] + [('&#%3d;'%x,'U') for x in range(217,221)] + [('&#%3d;'%x,'Y') for x in range(221,222)] +
                   [('&#%3d;'%x,'S') for x in range(223,224)] + [('&#%3d;'%x,'a') for x in range(224,231)] + [('&#%3d;'%x,'c') for x in range(231,232)] +
                   [('&#%3d;'%x,'e') for x in range(232,236)] + [('&#%3d;'%x,'i') for x in range(236,240)] + [('&#%3d;'%x,'n') for x in range(241,242)] +
                   [('&#%3d;'%x,'o') for x in range(242,247)+[248]] + [('&#%3d;'%x,'u') for x in range(249,253)] + [('&#%3d;'%x,'y') for x in [253,255]]                       
                   ] 

    @staticmethod
    def parse_text(text):
        for rk in HtmlReader.RE_SUBSTITUTE:
            found = True
            while found:
                rkm = re.search(rk, text)
                if rkm is None:
                    found = False
                else:
                    text = text[:rkm.start()] + rkm.groups(0)[0] + text[rkm.end():]
        for rr in HtmlReader.RE_REMOVE:
            text = re.sub(rr, ' ', text)
        for rr,rs in HtmlReader.RE_REPLACE:
            text = re.sub(rr, rs, text)
        return text
        
    def read(self, htmldir, outfile, stop_words=set(), pos_words=set(), page_name_word_sets=None, corpus_words=None,
             page_title_regexps=['<h1[^>]*>([^<]+)<[\/]h1>', '<table [^>]*>\s*<caption>([^<]+)<[\/]caption>', '<table [^>]*>\s*<caption>\s*<strong>([^<]+)<[\/]strong>\s*<[\/]caption>', '<table title="([^\"]+)"[^>]*>'], 
             page_title_ignore_suffixes=['-1', '-2', '- Advanced'],
             ignore_sections=set(), section_regexp='<h[1-4][^>]*>([^<]+)<[\/]h[1-4]>',
             min_pos_words_in_page_name=0, min_pos_words_in_section=0, 
             use_all_pages_match_pos_word=False, use_all_pages_match_sets=False, always_use_first_section=False, 
             action='write'):

        self._reset(outfile=outfile, stop_words=stop_words, pos_words=pos_words, page_name_word_sets=page_name_word_sets, corpus_words=corpus_words,
                    min_pos_words_in_page_name=min_pos_words_in_page_name, min_pos_words_in_section=min_pos_words_in_section, 
                    use_all_pages_match_pos_word=use_all_pages_match_pos_word, use_all_pages_match_sets=use_all_pages_match_sets,
                    always_use_first_section=always_use_first_section,
                    action=action)
        
        parser = SimpleWordParser(tolower=True, ascii_conversion=True, ignore_special_words=False)
    
        section_re = re.compile(section_regexp)
        
        self._start_action()
        page_name, section_name, section_in_page = None, None, 0
        page_name_words, section_words = [], []
        start_time = time.time()

        filenames = ['%s/%s'%(htmldir,fname) for fname in os.listdir(htmldir) if fname.endswith('.html')]
        assert len(filenames)>0 
        for ifname,fname in enumerate(filenames):
            print 'Reading %s' % fname
#             if fname!='C:/Chaim/chaim_private/Kaggle/AI2/corpus/CK12/OEBPS/table_66-1.html': continue
            with open (fname, 'rb') as myfile:
                text = myfile.read()
#             page_name = fname[:-5] if fname.endswith('.html') else fname
            pn_match = None
            for ptr in page_title_regexps:
                pn_match = re.search(ptr, text)
                if pn_match is not None: break
            if pn_match is None:
                print 'Could not find page title in file %s - skipping' % fname
                continue
            page_name = pn_match.groups(0)[0].strip()
            for ptis in page_title_ignore_suffixes:
                if page_name.endswith(ptis):
                    page_name = page_name[:-len(ptis)]
                    break
            page_name = HtmlReader.parse_text(page_name)
            text = text[pn_match.end():]
            page_name_words = parser.parse(page_name)
            page_name = CorpusReader.part_name_from_words(page_name_words, ifname)
            print 'page name = %s' % page_name  
            self._add_page(page_name, page_name_words)
            parts = re.split(section_re, text)
            assert len(parts)%2==1
#             print 'Found %d parts/section-names: %s' % (len(parts), parts[:5])
            parts = [''] + parts # 1st section has no name
            for ipart in range(0,len(parts),2):
#                 print 'ipart %s' % ipart
                if parts[ipart] is None:
                    section_name = ''
                else:
                    section_name = parts[ipart].strip().lower()
                    section_name = HtmlReader.parse_text(section_name)
                text = parts[ipart+1]
                if np.any([(re.match(isr, section_name) is not None) for isr in ignore_sections]):
#                     print '----- ignoring section %s' % section_name 
                    continue
                section_name_words = parser.parse(section_name)
                section_in_page = (ipart - 1) / 2
#                 print 'section <%s>' % section_name
#                 print 'text: %s' % text
                text = HtmlReader.parse_text(text)
                words = parser.parse(text)
                section_words = words
#                 print 'words: %s' % section_words
                self._add_section(page_name, page_name_words, section_name, section_name_words, section_in_page, section_words)
        
        end_time = time.time()
        print 'read_html total time = %.1f secs.' % (end_time-start_time)
        print 'Read %d pages, %d sections; applied action on %d sections' % (self.num_pages, self.num_sections, self.num_section_action)
        self._end_action()
        
        return self._locdic

class TextReader(CorpusReader):
    '''
    TextReader - read a "text" corpus
    '''    
    def read(self, dir, outfile, stop_words=set(), pos_words=set(),
             first_line_regexp='^CHAPTER', 
             ignore_sections=set(), section_end_regexp='^\s*$',
             action='write'):

        self._reset(outfile=outfile, stop_words=stop_words, pos_words=pos_words, page_name_word_sets=set(), corpus_words=None,
                    min_pos_words_in_page_name=0, min_pos_words_in_section=0, 
                    use_all_pages_match_pos_word=True, use_all_pages_match_sets=True, always_use_first_section=False,
                    action=action)
        
        parser = SimpleWordParser(tolower=True, ascii_conversion=True, ignore_special_words=False)
    
        first_line_re = re.compile(first_line_regexp)
        section_end_re = re.compile(section_end_regexp)
        
        self._start_action()
        page_name, section_name, section_in_page = None, None, 0
        page_name_words, section_words = [], []
        start_time = time.time()

        filenames = ['%s/%s'%(dir,fname) for fname in os.listdir(dir) if fname.endswith('.text')]
        assert len(filenames)>0 
        for ifname,fname in enumerate(filenames):
            print 'Reading %s' % fname
            page_name = fname[:-5]
            page_name_words = []
            print 'page name = %s' % page_name  
            self._add_page(page_name, page_name_words)
            section_in_page = 0
            section_name, section_name_words = '', []
            with open (fname, 'rb') as myfile:
                found_first_line = False
                text = ''
                for line in myfile:
                    line = line.strip()
#                     print 'LINE: "%s"' % line
                    if found_first_line:
                        if re.match(section_end_re, line) is not None:
#                             print '--- section ended ---'
#                             print '%s' % text
                            # Add previous section
                            section_words = parser.parse(text)
                            self._add_section(page_name, page_name_words, section_name, section_name_words, section_in_page, section_words)
                            section_in_page += 1
                            section_name, section_name_words = '', []
                            text = ''
                        else:
                            text += ' ' + line
                    else:
                        if re.match(first_line_re, line) is not None:
#                             print '===== found first line ====='
                            found_first_line = True
            assert found_first_line, 'Could not find first line in file %s' % fname
            # Add last section
            section_words = parser.parse(text)
            self._add_section(page_name, page_name_words, section_name, section_name_words, section_in_page, section_words)
        
        end_time = time.time()
        print 'read_text total time = %.1f secs.' % (end_time-start_time)
        print 'Read %d pages, %d sections; applied action on %d sections' % (self.num_pages, self.num_sections, self.num_section_action)
        self._end_action()
        
        return self._locdic

class TextSummaryReader(CorpusReader):
    '''
    TextSummaryReader - read a text summary corpus
    '''    
    def read(self, dir, outfile, stop_words=set(), pos_words=set(),
             section_start_regexp='^(Summary|(Lesson [sS]ummary))',
             vocab_start_regexp='^Vocabulary',
             section_end_regexp='.*((Review)|(Questions)|(Activity)|(Explore More)|(Practice)|(References)|(CHAPTER)|(Making Connections)|(Example [0-9]+)|(Introduction))', 
             section_end_regexp_after_empty_line='[0-9]+[\.] ', # references
             section_end_num_empty_lines=2, 
             vocab_line_prefix=(149, 32),
             ignore_line_regexp='^(([0-9]+[\.][0-9]+([\.]?)\s+)|(FIGURE [0-9]+))',
             action='write'):

        self._reset(outfile=outfile, stop_words=stop_words, pos_words=pos_words, page_name_word_sets=set(), corpus_words=None,
                    min_pos_words_in_page_name=0, min_pos_words_in_section=0, 
                    use_all_pages_match_pos_word=True, use_all_pages_match_sets=True, always_use_first_section=False,
                    action=action)
        
        parser = SimpleWordParser(tolower=True, ascii_conversion=True, ignore_special_words=False)
    
        section_start_re = re.compile(section_start_regexp)
        vocab_start_re = re.compile(vocab_start_regexp)
        section_end_re = re.compile(section_end_regexp)
        section_end_re_after_empty_line = re.compile(section_end_regexp_after_empty_line)
        ignore_line_re = re.compile(ignore_line_regexp)
        
        self._start_action()
        page_name, section_name, section_in_page = None, None, 0
        page_name_words, section_words = [], []
        sections_added = set() # to avoid adding the same section (line) twice
        start_time = time.time()

        filenames = ['%s/%s'%(dir,fname) for fname in os.listdir(dir) if fname.endswith('.text')]
        assert len(filenames)>0 
        for ifname,fname in enumerate(filenames):
            print 'Reading %s' % fname
            page_name = fname[:-5]
            page_name_words = []
            self._add_page(page_name, page_name_words)
            section_in_page = 0
            section_name, section_name_words = '', []
            with open (fname, 'rb') as myfile:
                is_in_section, is_in_vocab = False, False
                num_empty_lines = 0
                for line in myfile:
                    skip_line = False
                    line = line.strip()
#                     print 'LINE: "%s"' % line
                    if len(line)==0:
                        num_empty_lines += 1
                        if num_empty_lines >= section_end_num_empty_lines:
                            is_in_section, is_in_vocab = False, False
                        continue
                    if is_in_vocab:
                        assert not is_in_section
                        if not ((ord(line[0]) == vocab_line_prefix[0]) and (ord(line[1]) == vocab_line_prefix[1])):
                            is_in_vocab = False
                    else:
                        if re.match(section_end_re, line) is not None:
                            is_in_section = False
                        elif (num_empty_lines >= 1) and (re.match(section_end_re_after_empty_line, line) is not None):
                            is_in_section = False
                        elif re.match(section_start_re, line) is not None:
                            is_in_section = True
                            skip_line = True
                        elif re.match(vocab_start_re, line) is not None:
                            is_in_section, is_in_vocab = False, True
                            skip_line = True
                    if re.match(ignore_line_re, line): 
                        skip_line = True
                    num_empty_lines = 0
                    if (not skip_line) and (is_in_section or is_in_vocab):
                        # Add line as a section
#                         if is_in_vocab:
#                             print '%s' % ' '.join(['%d'%ord(c) for c in line[:10]])
#                             print '%s' % line
#                             kjfgkjdfhg()
#                         print '%s: %s...' % ('V' if is_in_vocab else 'S', line[:40])
                        section_words = parser.parse(line)
                        section_words_str = ' '.join(section_words)
                        if section_words_str in sections_added:
#                             print '----> already added: %s' % section_words_str
                            pass
                        else:
                            self._add_section(page_name, page_name_words, section_name, section_name_words, section_in_page, section_words)
                            section_in_page += 1
                            sections_added.add(section_words_str)
        
        end_time = time.time()
        print 'read_text total time = %.1f secs.' % (end_time-start_time)
        print 'Read %d pages, %d sections; applied action on %d sections' % (self.num_pages, self.num_sections, self.num_section_action)
        self._end_action()
        
        return self._locdic

class SimpleLineReader(CorpusReader):
    '''
    SimpleLineReader - read a corpus that is a simple text file, each line is treated as a separate section
    '''    
    def read(self, filenames, outfile, stop_words=set(), pos_words=set(), action='write'):
        self._reset(outfile=outfile, stop_words=stop_words, pos_words=pos_words, page_name_word_sets=set(), corpus_words=None,
                    min_pos_words_in_page_name=0, min_pos_words_in_section=0, 
                    use_all_pages_match_pos_word=True, use_all_pages_match_sets=True, always_use_first_section=False,
                    action=action)
        
        parser = SimpleWordParser(tolower=True, ascii_conversion=True, ignore_special_words=False)
    
        self._start_action()
        page_name, section_name, section_in_page = None, None, 0
        page_name_words, section_words = [], []
        sections_added = set() # to avoid adding the same section (line) twice        
        start_time = time.time()

        assert len(filenames)>0 
        for ifname,fname in enumerate(filenames):
            print 'Reading %s' % fname
            page_name = fname
            page_name_words = []
            self._add_page(page_name, page_name_words)
            section_in_page = 0
            section_name, section_name_words = '', []
            with open (fname, 'rb') as myfile:
                for line in myfile:
                    line = line.strip()
                    if len(line)==0: continue
                    section_words = parser.parse(line)
                    section_words_str = ' '.join(section_words)
                    if section_words_str in sections_added:
                        print '----> Already added: %s' % section_words_str
                    else:
                        self._add_section(page_name, page_name_words, section_name, section_name_words, section_in_page, section_words)
                        section_in_page += 1
                        sections_added.add(section_words_str)
        
        end_time = time.time()
        print 'read_text total time = %.1f secs.' % (end_time-start_time)
        print 'Read %d pages, %d sections; applied action on %d sections' % (self.num_pages, self.num_sections, self.num_section_action)
        self._end_action()
        
        return self._locdic

class SentenceReader(CorpusReader):
    '''
    SentenceReader - read a corpus and treat each sentence as a separate section
    '''

    def read(self, dir, outfile, stop_words=set(), pos_words=set(), sentence_sep='. ', action='write'):

        self._reset(outfile=outfile, stop_words=stop_words, pos_words=pos_words, page_name_word_sets=set(), corpus_words=None,
                    min_pos_words_in_page_name=0, min_pos_words_in_section=0, 
                    use_all_pages_match_pos_word=True, use_all_pages_match_sets=True, always_use_first_section=False,
                    action=action)
        
        parser = SimpleWordParser(tolower=True, ascii_conversion=True, ignore_special_words=False)
    
        filenames = ['%s/%s'%(dir,fname) for fname in os.listdir(dir) if fname.endswith('.text')]
        assert len(filenames)>0 

        self._start_action()
        page_name, section_name, section_in_page = None, None, 0
        page_name_words, section_words = [], []
        sections_added = set() # to avoid adding the same section (line) twice        
        start_time = time.time()

        for ifname,fname in enumerate(filenames):
            print 'Reading %s' % fname
            page_name = fname
            page_name_words = []
            self._add_page(page_name, page_name_words)
            section_in_page = 0
            section_name, section_name_words = '', []
            prev_section_words = []
            with open (fname, 'rb') as myfile:
                text = myfile.read()
                sents = text.split(sentence_sep)
                for sent in sents:
                    sent = sent.strip()
                    if len(sent)==0: continue
                    section_words = prev_section_words + parser.parse(sent)
                    section_words_str = ' '.join(section_words)
                    if section_words_str in sections_added:
                        print '----> Already added: %s' % section_words_str
                    else:
                        if len(section_words) < self.min_words_per_section:
                            prev_section_words = section_words # save for next sentence
                        else:
                            self._add_section(page_name, page_name_words, section_name, section_name_words, section_in_page, section_words)
                            section_in_page += 1
                            sections_added.add(section_words_str)
                            prev_section_words = []
        
        end_time = time.time()
        print 'read_text total time = %.1f secs.' % (end_time-start_time)
        print 'Read %d pages, %d sections; applied action on %d sections' % (self.num_pages, self.num_sections, self.num_section_action)
        self._end_action()
        
        return self._locdic


#################################################################################################
# LuceneCorpus
#################################################################################################

class LuceneCorpus(object):
    def __init__(self, index_dir, filenames, parser, similarity=None):
        self._index_dir = index_dir
        self._filenames = filenames
        self._parser = parser
        self._similarity = similarity
        lucene.initVM()
        self._analyzer = WhitespaceAnalyzer(Version.LUCENE_CURRENT)
        self._store = SimpleFSDirectory(File(self._index_dir))
        self._searcher = None

    def prp_index(self):
        '''
        Prepare the index given our "corpus" file(s)
        '''
        print '=> Preparing Lucene index %s' % self._index_dir
        writer = self._get_writer(create=True) #IndexWriter(dir, analyzer, True, IndexWriter.MaxFieldLength(512))
        print '   Currently %d docs (dir %s)' % (writer.numDocs(), self._index_dir)
        num_pages, num_sections = 0, 0
        page_name, section_name = None, None
        num_lines = 0
        for ifname,fname in enumerate(self._filenames):
            print '   Adding lines to index from file #%d: %s' % (ifname, fname)
            with open(fname,'rt') as infile:
                for text in infile:
    #                 print '%s' % text
                    if len(text)==0:
                        print 'Reached EOF'
                        break # EOF
                    if text.startswith(CorpusReader.PAGE_NAME_PREFIX):
                        page_name = text[len(CorpusReader.PAGE_NAME_PREFIX):].strip()
                        section_name = None
                        num_pages += 1
                    elif text.startswith(CorpusReader.SECTION_NAME_PREFIX):
                        section_name = text[len(CorpusReader.SECTION_NAME_PREFIX):].strip()
                        num_sections += 1
                    else:
                        assert (page_name is not None) and (section_name is not None)
                        #section_words = text.split(' ')
                        if self._parser is None:
                            luc_text = text
                        else:
                            section_words = self._parser.parse(text, calc_weights=False) #True)
                            if False:
                                print 'Adding words: %s (weights: %s)' % (section_words, weights)
                            luc_text = ' '.join(section_words)
                        doc = Document()
                        doc.add(Field("text", luc_text, Field.Store.YES, Field.Index.ANALYZED))
                        writer.addDocument(doc)
                    num_lines += 1
                    if num_lines % 100000 == 0:
                        print '    read %d lines so far: %d pages, %d sections' % (num_lines, num_pages, num_sections)

        print '   Finished - %d docs (dir %s)' % (writer.numDocs(), self._index_dir)
        writer.close()

    def search(self, words, max_docs, weight_func=lambda n: np.ones(n), score_func=lambda s: s):
        '''
        Search the index for the given words, return total score
        '''
        searcher = self._get_searcher()
        if type(words)==str:
            search_text = words
            search_text = AsciiConvertor.convert(search_text)
            for c in '/+-&|!(){}[]^"~*?:':
                search_text = search_text.replace('%s'%c, '\%s'%c)
        else:
            search_text = ' '.join(words)
        print 'search_text: %s' % search_text
        query = QueryParser(Version.LUCENE_CURRENT, "text", self._analyzer).parse(search_text)
        hits = searcher.search(query, max_docs)
#         print "Found %d document(s) that matched query '%s':" % (hits.totalHits, query)

        score_sum = 0.0
        weights = weight_func(len(hits.scoreDocs))
        for hit,weight in zip(hits.scoreDocs, weights):
            score_sum += weight * score_func(hit.score)
#             print ' score %.3f , weight %.5f -> %.5f' % (hit.score, weight, weight*hit.score)
#             print hit.score, hit.doc, hit.toString()
#             doc = searcher.doc(hit.doc)
#             print doc.get("text").encode("utf-8")
#         print 'score_sum = %.5f' % score_sum
        return score_sum

    def _get_writer(self, analyzer=None, create=False):
        config = IndexWriterConfig(Version.LUCENE_CURRENT, self._analyzer)
        if create:
            config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
        if self._similarity is not None:
            config.setSimilarity(self._similarity)
        writer = IndexWriter(self._store, config)
        return writer

    def _get_searcher(self):
        if self._searcher is None:
            self._searcher = IndexSearcher(DirectoryReader.open(self._store))
            if self._similarity is not None:
                self._searcher.setSimilarity(self._similarity)
        return self._searcher


#################################################################################################
# Data preparation - read original files, extract relevant info, write it in standard format
#################################################################################################

def read_ai2_data(dirname, outfile):
    '''
    Read the data downloaded from the AI2 data web-site
    '''
    print '=> Preparing AI2 data summary'
    outf = open(outfile,'w')

    # (1) Read questions+answers from tests
    print '--> Reading tests'
    for tdir,tfile in [('%s/Regents'%dirname           , 'Regents_Train.tsv'),
                       ('%s/Aristo_Multi-state'%dirname, 'Aristo_Multi-state_Train.tsv'),
                       ('%s/Aristo_Multi-state'%dirname, 'Aristo_Multi-state_Test.tsv' )]:
        print 'Reading test file %s' % tfile
        reg = read_input_file(tdir, filename=tfile, sep='\t', max_rows=100000)
        num_qst = 0
        for qst_ans,corr in np.array(reg[['question','AnswerKey']]):
            # Get question and all answers
            qst_re = re.compile('^(.+) [\(]A[\)] (.+?) [\(]B[\)] (.+?) [\(]C[\)] (.+?)(?: [\(]D[\)] (.+))?$')
            mr = re.match(qst_re, qst_ans)
            assert mr is not None
            qst, answers = mr.groups()[0], mr.groups()[1:]
            qst = qst.strip()
            ans = answers[ord(corr)-ord('A')]
            assert ans is not None
            ans = ans.strip()
            print 'Q: %s ; A: %s' % (qst, ans)
            outf.write('%s %s\n\n' % (qst, ans))
            num_qst += 1
        print ' Wrote %d question+answer pairs' % num_qst
    
    # (2) Read Aristo tables
    print '--> Reading Aristo tables'
    dir1 = 'AristoTablestore-Nov2015Snapshot/Tables/regents' 
    dir2 = 'AristoTablestore-Nov2015Snapshot/Tables/monarch' 
    filenames  = ['%s/%s'%(dir1,fname) for fname in os.listdir('%s/%s'%(dirname,dir1)) if fname.endswith('.tsv')]
    filenames += ['%s/%s'%(dir2,fname) for fname in os.listdir('%s/%s'%(dirname,dir2)) if fname.endswith('.tsv')]
    for fname in filenames:
        print 'Reading table file: %s' % fname
        num_line = -1
        with open('%s/%s' % (dirname, fname), 'rt') as infile:
            for line in infile:
                num_line += 1
                if num_line == 0: continue # skip header
                line = re.sub('\t', ' ', line)
                outf.write('%s\n\n' % line)
        print ' Wrote %d lines' % num_line
 
    # (3) Read BiologyHowWhyCorpus
    print '--> Reading BiologyHowWhyCorpus'
    filenames  = ['BiologyHowWhyCorpus/%s'%fname for fname in os.listdir('%s/BiologyHowWhyCorpus'%dirname) if fname.endswith('.xml')]
    for fname in filenames:
        print 'Reading file: %s' % fname
        qst_re = re.compile('\s*<text>([^<]+)<[\/]text>')
        ans_re1 = re.compile('\s*<justification>([^<]+)<[\/]justification>')
        ans_re2a = re.compile('\s*<justification>([^<]+)$')
        ans_re2b = re.compile('([^<]+)<[\/]justification>\s*$')
        num_ans = 0
        with open('%s/%s' % (dirname, fname), 'rt') as infile:
            qst, ans = None, None
            in_ans = False
            for line in infile:
                if in_ans:
                    assert ans is not None
                    am2b = re.match(ans_re2b, line)
                    if am2b is None:
                        ans += ' ' + line.strip()
                    else:
                        ans += ' ' + am2b.groups()[0].strip()
                        num_ans += 1
                        outf.write('%s ; %s\n\n' % (qst, ans))
                        ans, in_ans = None, False
                else:
                    qm = re.match(qst_re, line)
                    if qm is not None:
                        qst = qm.groups()[0].strip()
                    else:
                        am1 = re.match(ans_re1, line)
                        if am1 is not None:
                            assert qst is not None
                            ans = am1.groups()[0].strip()
                            num_ans += 1
                            outf.write('%s ; %s\n\n' % (qst, ans))
                        else:
                            am2a = re.match(ans_re2a, line)
                            if am2a is not None:
                                assert qst is not None
                                ans = am2a.groups()[0].strip()
                                in_ans = True
        print ' Wrote %d answers' % num_ans
    
    outf.close()
 

def read_quizlet_data(outfile, filenames, dirname=None, file_suffixes=['.f1.txt', '.f2.txt', '.f3.txt', '.f4.txt'], out_format='txt'):
    '''
    Read data obtained from Quizlet.
    Data comes in several formats.
    '''
    print '=> Preparing Quizlet data'
    assert out_format in ['txt', 'csv']
    # Open output file
    if out_format == 'csv':
        outf = open(outfile,'wb')
        outw = csv.writer(outf)
        outw.writerow(['id','question','correctAnswer','answerA','answerB','answerC','answerD'])
    else:
        outw = open(outfile,'w')
    # Read input files
    if filenames is None:
        filenames = ['%s/%s'%(dirname,fname) for fname in os.listdir(dirname) if np.any([fname.endswith(fs) for fs in file_suffixes])]
        assert len(filenames)>0
    total_questions = 0
    for fname in filenames:
        print '--> Quizlet file: %s' % fname
        format = [i for i,fs in enumerate(file_suffixes) if fname.endswith(fs)][0]
        qst_num = 0
        with open(fname, 'rb') as infile:
            text = infile.read()
            if format == 0:
                f1_qst = '(.*)\s+A (.*)\s+B (.*)\s+C (.*)\s+D (.*)\s+(?:[\?][\?][\?])\s+'
                search_from = 0
                while search_from < len(text):
                    sr = re.search(f1_qst, text[search_from:])
                    if sr is None:
                        search_from = len(text)
                    else:
                        qst, ansA, ansB, ansC, ansDcorr = sr.groups()
                        qst, ansA, ansB, ansC = qst.strip(), ansA.strip(), ansB.strip(), ansC.strip()
                        ansD = None
                        posD = 2
                        while posD < len(ansDcorr):
                            dsr = re.search(' ([ABCD]) (.*)$', ansDcorr[posD:])
                            if dsr is None:
                                posD = len(ansDcorr)
                            else:
                                corrI, corrA = dsr.groups()[0], dsr.groups()[1].strip()
                                ansD2 = ansDcorr[:(posD+dsr.start())].strip()
                                if corrA in [ansA, ansB, ansC, ansD2]:
                                    ansD = ansD2
                                    assert (corrI=='A' and corrA==ansA) or (corrI=='B' and corrA==ansB) or (corrI=='C' and corrA==ansC) or (corrI=='D' and corrA==ansD)  
                                    break
                                posD += dsr.start() + 1
                        assert ansD is not None, 'Not found correct answer for %s' % qst
                        print 'Q %d: %s' % (qst_num, qst)
                        iscorr = False
                        for ai,ans in enumerate([ansA, ansB, ansC, ansD]):
                            print '%s A%d: %s' % ('*' if ord(corrI[0])-ord('A')==ai else ' ', ai+1, ans)
                        
                        search_from += sr.end()
                        qst_id = '%d_quizlet-%s_%s' % (total_questions, fname.split('/')[-1].split('.')[0], qst_num)
                        qst_num += 1
                        total_questions += 1
       
                        if out_format == 'csv':
                            #             'id'   ,'question','correctAnswer', 'answerA','answerB','answerC','answerD'])
                            outw.writerow([qst_id, qst      ,corrI]         + [ansA, ansB, ansC, ansD])
                        else:
                            outw.write('%s  %s\n' % (qst, [ansA, ansB, ansC, ansD][ord(corrI[0])-ord('A')]))
            elif format == 1:
                f2_qst = '([ABCD]) (.*)\s+A[\.] (.*)\s+B[\.] (.*)\s+C[\.] (.*)\s+D[\.] (.*)\s+(?:[\?][\?][\?])\s+'
                search_from = 0
                while search_from < len(text):
                    sr = re.search(f2_qst, text[search_from:])
                    if sr is None:
                        search_from = len(text)
                    else:
                        corrI, qst, ansA, ansB, ansC, ansD = sr.groups()
                        qst, ansA, ansB, ansC, ansD = qst.strip(), ansA.strip(), ansB.strip(), ansC.strip(), ansD.strip()
                        print 'Q %d: %s' % (qst_num, qst)
                        for ai,ans in enumerate([ansA, ansB, ansC, ansD]):
                            print '%s A%d: %s' % ('*' if ord(corrI)-ord('A')==ai else ' ', ai+1, ans)
                        
                        search_from += sr.end()
                        qst_id = '%d_%s_%s' % (total_questions, fname.split('/')[-1].split('.')[0], qst_num)
                        qst_num += 1
                        total_questions += 1       
                        if out_format == 'csv':
                            #             'id'   ,'question','correctAnswer', 'answerA','answerB','answerC','answerD'])
                            outw.writerow([qst_id, qst      ,corrI]         + [ansA, ansB, ansC, ansD])
                        else:
                            outw.write('%s  %s\n' % (qst, [ansA, ansB, ansC, ansD][ord(corrI[0])-ord('A')]))
            elif format == 2:
                f3_qst_a = '(?:([^\?]+[\?]) ([^\?]+))|(?:([A-Z][^A-Z]+) ([A-Z][^A-Z]+))\s+(?:[\?][\?][\?])'
                #f2_qst_b = '([A-Z][^A-Z]+) ([A-Z][^A-Z]+)\s+(?:[\?][\?][\?])'
                search_from = 0
                while search_from < len(text):
                    sr = re.search(f3_qst_a, text[search_from:])
                    if sr is None:
                        search_from = len(text)
                    else:
                        q1, a1, q2, a2 = sr.groups()
                        if q1 is None:
                            qst, ans = q2, a2
                        else:
                            qst, ans = q1, a1
                        assert (qst is not None) and (ans is not None)
                        qst, ans = qst.strip(), ans.strip()
                        print 'Q %d: %s' % (qst_num, qst)
                        print '%s A%d: %s' % ('*', 1, ans)

                        search_from += sr.end()
                        qst_id = '%d_%s_%s' % (total_questions, fname.split('/')[-1].split('.')[0], qst_num)
                        qst_num += 1
                        total_questions += 1
        
                        if out_format == 'csv':
                            #             'id'   ,'question','correctAnswer', 'answerA','answerB','answerC','answerD'])
                            outw.writerow([qst_id, qst      ,'A']           + [ans,   '-', '-', '-'])
                        else:
                            outw.write('%s  %s\n' % (qst, ans))
            elif format == 3:
                assert out_format == 'txt'
                outw.write(text) # write entire file as is
                
    if out_format == 'csv':
        outf.close()
    else:
        outw.close()
    print 'Wrote Quizlet data to %s' % (total_questions, outfile)

    
def studystack_check_question_answer(qst, ans):
    '''
    Parsing of StudyStack questions+answers
    '''
    # Remove 'A. bla B. blaa' etc. from question/answer (eg, see card 1712202 , 1149349 - take answer from question)
    for abcd_re in ['.+([\(\;\. ]?(A)[\)\. ](.*)[\(\;\. ](B)[\)\. ](.*)[\(\;\. ](C)[\)\. ](.*)[\(\;\. ](D)[\)\. ](.*))$',
                    '.+([\(\;\. ]?(a)[\)\. ](.*)[\(\;\. ](b)[\)\. ](.*)[\(\;\. ](c)[\)\. ](.*)[\(\;\. ](d)[\)\. ](.*))$',
                    '.+([\(\;\. ]?(A)[\)\. ](.*)[\(\;\. ](B)[\)\. ](.*)[\(\;\. ](C)[\)\. ](.*))$',
                    '.+([\(\;\. ]?(a)[\)\. ](.*)[\(\;\. ](b)[\)\. ](.*)[\(\;\. ](c)[\)\. ](.*))$',  
                    '.+([\(\;\. ]?(A)[\)\. ](.*)[\(\;\. ](B)[\)\. ](.*))$',
                    '.+([\(\;\. ]?(a)[\)\. ](.*)[\(\;\. ](b)[\)\. ](.*))$',
                    ]:
        sm = re.search(abcd_re, qst)
        if sm is not None:
            abcd_text = sm.groups()[0]
            abcd_to_ans = dict(zip(sm.groups()[1::2], sm.groups()[2::2]))
            if abcd_to_ans.has_key(ans):
                # Replace answer by relevant text
                ans2 = abcd_to_ans[ans]
#                 print '   found ans %s -> %s' % (ans, ans2)
            else:
                ans2 = ans
            # Remove list of answers from question
            if re.match('^[abcd\-\ \.\(\)]*all of the above\s*$', ans.lower()) is None:
                qst2 = qst.replace(abcd_text, ' ') #re.sub(abcd_re, ' ', qst)
            else:
                qst2 = qst  
            return (qst2, ans2, True)
        
    for parn_re in ['.+([\(]([^\)\,]+)[,]([^\)\,]+)[,]([^\)\,]+)[,]([^\)\,]+)[\)]).*$',
                    '.+([\(]([^\)\,]+)[,]([^\)\,]+)[,]([^\)\,]+) or ([^\)\,]+)[\)]).*$',
                    '.+([\(]([^\)\,]+)[,]([^\)\,]+)[,]([^\)\,]+)[\)]).*$',
                    '.+([\(]([^\)\,]+)[,]([^\)\,]+) or ([^\)\,]+)[\)]).*$',
                    '.+([\(]([^\)\,]+)[,]([^\)\,]+)[\)]).*$',
                    '.+([\(]([^\)\,]+) or ([^\)\,]+)[\)]).*$']:
        sm = re.search(parn_re, qst)
        if sm is not None:
            parn_text = sm.groups()[0]
            parn_options = [opt.strip().lower() for opt in sm.groups()[1:]]
            if ans.strip().lower() in parn_options:
                # Remove list of answers from question
                qst2 = qst.replace(parn_text, ' ') #re.sub(abcd_re, ' ', qst)
                ans2 = ans  
                return (qst2, ans2, True)

    # Nothing found
    return (qst, ans, False)
    

#################################################################################################
# WikiCorpusBuilder
#################################################################################################

class WikiCorpusBuilder(object):
    ALL_CATEGORIES_FILE      = 'all_categories.pkl'
    PARENT_CATEGORIES_FILE   = 'parent_categories.pkl'
    USE_CATEGORIES_FILE      = 'use_categories.pkl'
    PAGES_IN_CATEGORIES_FILE = 'pages_in_categories.pkl'
    COMMON_WORDS_FILE        = 'common_words.pkl'
    UNCOMMON_WORDS_FILE      = 'uncommon_words.pkl'
    STOP_WORDS_FILE          = 'stop_words.pkl'
    EXAMS_WORDS_FILE         = 'exams_words.pkl'
    POSITIVE_WORDS_FILE      = 'positive_words.pkl'
    ANSWERS_FILE             = 'all_answers.pkl'
    CORPUS_FILE              = 'corpus.txt'
    
    def __init__(self, wiki_name, wiki_dir, wiki_file, debug_flag=False):
        self.wiki_name = wiki_name
        self.wiki_dir  = wiki_dir
        self.wiki_file = wiki_file
        self.wikir = WikiReader(wiki_name, debug_flag=debug_flag)
        
    def read_categories(self, reread=False):
        print '=> Reading categories for %s' % self.wiki_name
        categories_file = '%s/%s_%s' % (self.wiki_dir, self.wiki_name, WikiCorpusBuilder.ALL_CATEGORIES_FILE)
        parents_file    = '%s/%s_%s' % (self.wiki_dir, self.wiki_name, WikiCorpusBuilder.PARENT_CATEGORIES_FILE)
        gc.collect()
        if reread or (not os.path.exists(categories_file)) or (not os.path.exists(parents_file)):
            self.wikir.read_sub_categories(wikifile='%s/%s' % (self.wiki_dir, self.wiki_file), max_read_lines=99900000000)
            save_to_pkl(categories_file, self.wikir.all_categories)
            save_to_pkl(parents_file, self.wikir.parent_categories)
        else:
            self.wikir.all_categories = load_from_pkl(categories_file)
            self.wikir.parent_categories = load_from_pkl(parents_file)
        print 'There are a total of %d categories' % len(self.wikir.all_categories)
            
    def read_pages_in_categories(self, target_categories, max_cat_depth, important_categories, reread=False):
        print '=> Reading pages in target categories for %s' % self.wiki_name
        self.target_categories = target_categories
        self.max_cat_depth = max_cat_depth
        use_categories_file      = '%s/%s_%s' % (self.wiki_dir, self.wiki_name, WikiCorpusBuilder.USE_CATEGORIES_FILE)
        pages_in_categories_file = '%s/%s_%s' % (self.wiki_dir, self.wiki_name, WikiCorpusBuilder.PAGES_IN_CATEGORIES_FILE)
        if reread or (not os.path.exists(use_categories_file)) or (not os.path.exists(pages_in_categories_file)):
            if self.target_categories is None:
                self.use_categories = self.wikir.all_categories
            else:
                self.use_categories = set([cat for cat in self.wikir.all_categories 
                                           if self.wikir.search_categories(cat, self.target_categories, max_depth=self.max_cat_depth) >= 0])
            save_to_pkl(use_categories_file, self.use_categories)
            self.pages_in_categories = self.wikir.read_pages_in_categories(wikifile='%s/%s' % (self.wiki_dir, self.wiki_file), 
                                                                           use_categories=self.use_categories, max_read_lines=99900000000)
            save_to_pkl(pages_in_categories_file, self.pages_in_categories)
        else:
            self.use_categories = load_from_pkl(use_categories_file)
            self.pages_in_categories = load_from_pkl(pages_in_categories_file)

        print 'Using %d categories related to %s target categories with depth <= %d' % \
                (len(self.use_categories), 'x' if self.target_categories is None else len(self.target_categories), self.max_cat_depth)
        print 'Missing important categories: %s' % str([cat for cat in important_categories if cat not in self.use_categories])
        print 'There are %d pages in the %d categories' % (len(self.pages_in_categories), len(self.use_categories))

    def find_common_words(self, wiki_common_words_min_frac=0.2, wiki_uncommon_words_max_frac=0.01, use_wiki_stop_words=True, 
                          max_read_lines=100000000, reread=False):
        print '=> Finding common/uncommon words'
        self.wiki_common_words_min_frac = wiki_common_words_min_frac
        self.wiki_uncommon_words_max_frac = wiki_uncommon_words_max_frac
        self.use_wiki_stop_words = use_wiki_stop_words
        common_words_file   = '%s/%s_%.4f_%s'   % (self.wiki_dir, self.wiki_name, self.wiki_common_words_min_frac, WikiCorpusBuilder.COMMON_WORDS_FILE)
        uncommon_words_file = '%s/%s_%.4f_%s'   % (self.wiki_dir, self.wiki_name, self.wiki_uncommon_words_max_frac, WikiCorpusBuilder.UNCOMMON_WORDS_FILE)
        stop_words_file     = '%s/%s_%.4f_%s%s' % (self.wiki_dir, self.wiki_name, self.wiki_common_words_min_frac, 'wsw_' if self.use_wiki_stop_words else '', WikiCorpusBuilder.STOP_WORDS_FILE)
        # Read first X lines from Wiki corpus, and get the set of Wiki stop-words (words that appear in many documents),
        # as well as the "uncommon" words (words that appear in a small fraction of the documents)
        if reread or (not os.path.exists(common_words_file)) or (not os.path.exists(uncommon_words_file)) or (not os.path.exists(stop_words_file)):
            wiki_locdic = self.wikir.read(wikifile='%s/%s' % (self.wiki_dir, self.wiki_file), 
                                          outfile='%s/%s_locdic1.txt' % (self.wiki_dir, self.wiki_name), # ignored...
                                          #only_first_section_per_page=True, max_read_lines=max_read_lines, 
                                          only_first_section_per_page=False, max_sections_per_page=1, max_read_lines=max_read_lines,
                                          stop_words=SpecialWords.ignore_words, pos_words=set(), 
                                          min_pos_words_in_page_name=0, min_pos_words_in_section=0, action='locdic')
            self.wiki_common_words   = set([word for dc,word in wiki_locdic.sort_words_by_num_docs() if dc>(self.wiki_common_words_min_frac  *wiki_locdic.get_num_docs())])
            self.wiki_uncommon_words = set([word for dc,word in wiki_locdic.sort_words_by_num_docs() if dc<(self.wiki_uncommon_words_max_frac*wiki_locdic.get_num_docs())])
            self.stop_words = set(SpecialWords.ignore_words).union(self.wiki_common_words)
            if self.use_wiki_stop_words:
                self.stop_words.update(WikiReader.WIKI_STOP_WORDS)
            wiki_locdic = None
            gc.collect()
            save_to_pkl(common_words_file  , self.wiki_common_words)
            save_to_pkl(uncommon_words_file, self.wiki_uncommon_words)
            save_to_pkl(stop_words_file    , self.stop_words)
        else:
            self.wiki_common_words   = load_from_pkl(common_words_file)
            self.wiki_uncommon_words = load_from_pkl(uncommon_words_file)
            self.stop_words          = load_from_pkl(stop_words_file)
            
        print 'There are %d common words (>%.4f docs)'   % (len(self.wiki_common_words), self.wiki_common_words_min_frac)
        print 'There are %d uncommon words (<%.4f docs)' % (len(self.wiki_uncommon_words), self.wiki_uncommon_words_max_frac)
        print 'Using %d stop words (%s wiki stop words)' % (len(self.stop_words), 'with' if self.use_wiki_stop_words else 'without')

    def create_corpus(self, train_b, valid_b, min_pos_words_in_page_name, min_pos_words_in_section, 
                      only_first_section_per_page=False, max_sections_per_page=99999999,
                      use_all_pages_match_pos_word=True, use_all_pages_match_answer=True, pages_to_use=None, always_use_first_section=False,
                      max_read_lines=99900000000, reread=False):
        print '=> Creating corpus'        
        self.min_pos_words_in_page_name   = min_pos_words_in_page_name
        self.min_pos_words_in_section     = min_pos_words_in_section
        self.only_first_section_per_page  = only_first_section_per_page
        self.max_sections_per_page        = max_sections_per_page
        self.use_all_pages_match_pos_word = use_all_pages_match_pos_word
        self.use_all_pages_match_answer   = use_all_pages_match_answer
        self.always_use_first_section     = always_use_first_section
        exams_words_file = '%s/%s_%s' % (self.wiki_dir, self.wiki_name, WikiCorpusBuilder.EXAMS_WORDS_FILE)
        pos_words_file   = '%s/%s_%.4f_%s%s' % (self.wiki_dir, self.wiki_name, self.wiki_common_words_min_frac, 'wsw_' if self.use_wiki_stop_words else '', WikiCorpusBuilder.POSITIVE_WORDS_FILE)
        answers_file     = '%s/%s_%s' % (self.wiki_dir, self.wiki_name, WikiCorpusBuilder.ANSWERS_FILE)
        corpus_file      = '%s/%s_%.4f_%s%.4f_%d_%d_%s_%s_%s' % (self.wiki_dir, self.wiki_name, self.wiki_common_words_min_frac, 'wsw_' if self.use_wiki_stop_words else '',
                                                                 self.wiki_uncommon_words_max_frac, self.min_pos_words_in_page_name, self.min_pos_words_in_section,
                                                                 self.use_all_pages_match_pos_word, self.use_all_pages_match_answer, 
                                                                 self.always_use_first_section)
        if pages_to_use is not None:
            corpus_file = '%s_pn%d' % (corpus_file, len(pages_to_use))
        corpus_file = '%s_%s' % (corpus_file, WikiCorpusBuilder.CORPUS_FILE)
        print 'Corpus file: %s' % corpus_file
        gc.collect()
        
        # Get the corpus of the train+validation sets
        if reread or (not os.path.exists(pos_words_file)) or (not os.path.exists(answers_file)):
            # Get all the words that appear in the exams
            if valid_b is None:
                all_exams = train_b[['ID','question','answer']]
            else:
                all_exams = pd.concat([train_b[['ID','question','answer']], valid_b[['ID','question','answer']]])
            parser = SimpleWordParser()
            exams_locdic = build_training_location_dictionary(all_exams, parser=parser, use_answers=True,
                                                              min_word_docs_frac=0, max_word_docs_frac=1.0, min_word_count_frac=0, max_word_count_frac=1.0,
                                                              ascii_conversion=True)
            self.exams_words = exams_locdic.word_ids.keys()
            # Set the "positive_words" as all the words from the train(+validation) files that are uncommon in Wiki
            self.pos_words = set(self.exams_words).intersection(self.wiki_uncommon_words)
            # Get all the answers (each answer = a set of words)
            self.all_answers = set()
            for answer in all_exams['answer']:
                self.all_answers.add(tuple(sorted(parser.parse(answer))))
            save_to_pkl(exams_words_file, self.exams_words)
            save_to_pkl(pos_words_file, self.pos_words)
            save_to_pkl(answers_file, self.all_answers)
        else:
            self.exams_words = load_from_pkl(exams_words_file)
            self.pos_words   = load_from_pkl(pos_words_file)
            self.all_answers = load_from_pkl(answers_file)
            
        print 'There are %d positive words (%d wiki uncommon words, %d words from exams)' % (len(self.pos_words), len(self.wiki_uncommon_words), len(self.exams_words))
        print 'There are a total of %d unique answers' % len(self.all_answers)
        print 'Using %d stop words' % (len(self.stop_words))
        if pages_to_use is None:
            use_pages = self.pages_in_categories
        else:
            use_pages = pages_to_use
        print 'Considering %d pages' % len(use_pages)
        
        if reread or (not os.path.exists(corpus_file)):
            print 'Writing %s corpus to %s' % (self.wiki_name, corpus_file)
            ld = self.wikir.read(wikifile='%s/%s' % (self.wiki_dir, self.wiki_file),
                                 outfile=corpus_file, 
                                 only_first_section_per_page=self.only_first_section_per_page, max_sections_per_page=self.max_sections_per_page, 
                                 use_pages=use_pages,
                                 max_read_lines=max_read_lines, 
                                 stop_words=self.stop_words, pos_words=self.pos_words, 
                                 page_name_word_sets=self.all_answers, corpus_words=None, ##set(exams_locdic.word_ids.keys()),
                                 min_pos_words_in_page_name=self.min_pos_words_in_page_name, min_pos_words_in_section=self.min_pos_words_in_section,
                                 use_all_pages_match_pos_word=self.use_all_pages_match_pos_word, use_all_pages_match_sets=self.use_all_pages_match_answer, 
                                 always_use_first_section=self.always_use_first_section,
                                 action='write')
            print 'Done writing corpus'
        
        gc.collect()
        return corpus_file
    

#################################################################################################
# Parsing & NLP utilities
#################################################################################################

class AsciiConvertor(object):
    ascii_orig = ['0','1','2','3','4','5','6','7','8','9',
                  'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
                  'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
                  '+','-','=','*','/','\\','_','~','>','<','%','$','#','@','&',
                  '.',',',';',':','!','?',
                  '\'']
    ascii_conv = {138: 's', 140: 'o', 142: 'z', 
                  150: '-', 151: '-', 152: '~', 154: 's', 156: 'o', 158: 'z', 159: 'y', 
                  192: 'a', 193: 'a', 194: 'a', 195: 'a', 196: 'a', 197: 'a', 198: 'a', 199: 'c', 200: 'e', 201: 'e', 202: 'e', 203: 'e', 204: 'i', 205: 'i',
                  206: 'i', 207: 'i', 209: 'n', 210: 'o', 211: 'o', 212: 'o', 213: 'o', 214: 'o', 215: '*', 216: 'o', 217: 'u', 218: 'u', 219: 'u', 220: 'u',
                  221: 'y', 223: 's', 224: 'a', 225: 'a', 226: 'a', 227: 'a', 228: 'a', 229: 'a', 230: 'a', 231: 'c', 232: 'e', 233: 'e', 234: 'e', 235: 'e',
                  236: 'i', 237: 'i', 238: 'i', 239: 'i', 241: 'n', 242: 'o', 243: 'o', 244: 'o', 245: 'o', 246: 'o', 248: 'o', 249: 'u', 250: 'u',
                  250: 'u', 251: 'u', 252: 'u', 253: 'y', 255: 'y' 
                  }
    ascii_mapping = None

    @staticmethod
    def convert(text):
        if AsciiConvertor.ascii_mapping is None:
            print 'Building ascii dict'
            AsciiConvertor.ascii_mapping = [' ']*256
            for c in AsciiConvertor.ascii_orig:
                AsciiConvertor.ascii_mapping[ord(c)] = c
            for oc,c in AsciiConvertor.ascii_conv.iteritems():
                AsciiConvertor.ascii_mapping[oc] = c
        return ''.join(map(lambda c: AsciiConvertor.ascii_mapping[ord(c)], text))


class SpecialWords(object):
    '''
    Stop words
    '''
    ignore_words = None
    my_stopwords = set(['', 'and', 'or', 'the', 'of', 'a', 'an', 'to', 'from',
                        'be', 'is', 'are', 'am', 'was', 'were', 'will', 'would', 
                        'do', 'does', 'did',
                        'have', 'has', 'had', 
                        'can', 'could', 'should', 'ought',
                        'may', 'might',
                        'by', 'in', 'into', 'out', 'on', 'over', 'under', 'for', 'at', 'with', 'about', 'between', 
                        'that', 'this', 'these', 'those', 'there', 'than', 'then', 
                        'we', 'our', 'they', 'their', 'you', 'your', 'he', 'his', 'she', 'her', 'it', 'its', 
                        'which', 'what', 'how', 'where', 'when', 'why', 
                        'not', 'no', 'if', 'maybe', 
                        'more', 'most', 'less', 'least', 'some', 'too', 
                        'best', 'better', 'worst', 'worse',
                        'same', 'as', 'like', 'different', 'other', 'so',
                        ])
     
    
    @staticmethod
    def read_nltk_stopwords(keep_my_words=True):
        # Stopwords taken from nltk's nlp/stopwords/english
        SpecialWords.ignore_words = set([word.strip().lower() for word in 
                                         ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", 
                                          "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", 
                                          "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", 
                                          "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", 
                                          "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", 
                                          "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", 
                                          "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", 
                                          "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", 
                                          "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", 
                                          "don", "should", "now"]])
        if keep_my_words:
            SpecialWords.ignore_words.difference_update(['above','after','again','against','all','any','before','below','between','both','down','during',
                                                         'each','few','further','into','just','more','most','no','not','now','off','once','only','out','over','own',
                                                         'same','through','under','until','up'])
        print '-> Set %d NLTK stopwords (%s my words)' % (len(SpecialWords.ignore_words), 'kept' if keep_my_words else 'did not keep')

    @staticmethod
    def filter(words):
        if SpecialWords.ignore_words is None:
            SpecialWords.read_nltk_stopwords()
#             SpecialWords.ignore_words = SpecialWords.my_stopwords
            
#         print 'filtering: %s' % ' ; '.join(words)
#         print 'filtering: %s' % ' ; '.join([word for word in words if word not in SpecialWords.ignore_words])
        fwords = [word for word in words if word not in SpecialWords.ignore_words]
        if len(fwords) > 0:
            return fwords
        else: # do not filter out ALL the words
            return words

    @staticmethod
    def filter1(word):
        if SpecialWords.ignore_words is None:
            SpecialWords.read_nltk_stopwords()
        return word not in SpecialWords.ignore_words
    
    
#################################################################################################
# Statistical/math functions
#################################################################################################

def calc_auc(labels, preds, two_sides=False):
    fpr, tpr, thresholds = roc_curve(labels, preds)
    score = auc(fpr,tpr)
    if two_sides and (score < 0.5):
        score = 1.0 - score
    return score

def calc_accuracy(labels_dict, preds_dict, ids=None):
    assert set(labels_dict.keys())==set(preds_dict.keys())
    if ids is None:
        ids = labels_dict.keys()
    return np.mean([labels_dict[k]==preds_dict[k] for k in ids])

def multiple_tests_correction(p, n):
    '''
    Multiple testing correction using Sidak's method: P(min{p_1,...,p_n} <= p) = 1 - (1-p)**n
    p - the probability to fix (the minimum p-value obtained from n tests)
    n - the number of tests performed
    '''
    assert p>=0 and p<=1 and n>0
    q = p*n;
    # If p*n is very small, then it's a good approximation of 1-(1-p)**n
    if q < 0.0001:
        return q
    # If n is large, then we can use (1-p)**n = exp(-p*n)
    if n > 10:
        return 1.0 - np.exp(-q)
    # Exact computation
    return 1.0 - (1.0-p)**n
    
def hg_test(M, n, N, k, dir=None, mult_test_correct=True):
    '''
    Check for over/under-representation using the hypergeometric test
    M - total number of samples
    n - number of samples of type A
    N - number of samples in subset
    k - number of samples of type A in subset
    dir - +1/-1 for over/under-representation, or None for both
    mult_test_correct - whether to apply multiple testing correction 
    '''
    if M==0 or N==0 or n==0:
        assert k==0
        return {'ratio': 0, 'prob': 1.0, 'dir': 1}
    # Compute p-value (HG test)
    if dir is None:
        if (k+0.0)/N >= (n+0.0)/M:
            # Over-representation
            pval = scipy.stats.hypergeom.sf(k-1, M, n, N) # prob to get >= k
            dir = 1
        else:
            # Under-representation
            pval = scipy.stats.hypergeom.cdf(k , M, n, N) # prob to get <= k
            dir = -1
    #     print 'k=%d,M=%d,n=%d,N=%d -> ratio %.3f -> pval = %.2e , dir %d' % (k,M,n,N, (k+0.0)/N, pval, dir)
        pvalue = np.clip(pval, 0, 1) # hypergeom sometimes returns values <0 or >1
        if mult_test_correct:
            pvalue = multiple_tests_correction(pvalue, n=2)
    else:
        if dir == 1:
            # Over-representation
            pval = scipy.stats.hypergeom.sf(k-1, M, n, N) # prob to get >= k
        else:
            assert dir == -1
            # Under-representation
            pval = scipy.stats.hypergeom.cdf(k , M, n, N) # prob to get <= k
        pvalue = np.clip(pval, 0, 1) # hypergeom sometimes returns values <0 or >1
    return {'ratio': (k+0.0)/N, 'prob': pvalue, 'dir': dir} 


#################################################################################################
# Feature extraction
#################################################################################################

class FeatureExtractor(object):
    '''
    This is the main class that runs the various search functions and prepares the features.
    Each feature is a score (or value) for the relevant question,answer pair.
    '''
    
    def __init__(self, base_dir, recalc=False, norm_scores_default=False, print_level=1):
        self.base_dir = base_dir
        self.recalc = recalc
        self.norm_scores_default = norm_scores_default
        self.print_level = print_level
    
    def _words_to_names(self, words):
        names = []
        for word in words:
            if len(word) == 0:
                return ''
            names.append(word[0].upper() + word[1:])
        return names
    
    def prepare_word_sets(self, corpus_dir, train_b, valid_b, test_b):
        if self.print_level > 0:
            print '-> Preparing word sets'
        word_sets_file = '%s/word_sets.pkl' % corpus_dir
        self.word_sets = load_from_pkl(word_sets_file)
        if self.word_sets is None:
            # Prepare list of words (and pairs) that appear in training set
            parser = SimpleWordParser(tuples=[1,2])
            words = set()
            for exam in [train_b, valid_b, test_b]:
                if exam is not None:
                    words.update(np.concatenate([self._words_to_names(parser.parse(qst)) for qst in exam['question']]))
                    words.update(np.concatenate([self._words_to_names(parser.parse(ans)) for ans in exam['answer']]))
            words.difference_update(['']) # ignore empty word
            words = sorted(words)
            if self.print_level > 1:
                print '%d word sets: ...%s...' % (len(words), words[::5000])
            self.word_sets = words
            save_to_pkl(word_sets_file, self.word_sets)            
        
    def prepare_corpuses(self, corpus_dir, train_b, valid_b, prp_wiki_corpuses=True):
        '''
        Prepare all the corpus files we shall be using. This needs to be done only once.
        '''
        if self.print_level > 0:
            print '-> Preparing corpuses'
        
        # Prepare CK-12 HTML corpus
        self.ck12html_corpus = '%s/CK12/OEBPS/ck12.txt' % corpus_dir
        if not os.path.exists(self.ck12html_corpus): 
            # Doc per HTML section (h1-4)
            htmlr = HtmlReader(min_chars_per_line=1, min_words_per_section=20)
            locdic = htmlr.read(htmldir='%s/CK12/OEBPS' % corpus_dir, 
                                outfile=self.ck12html_corpus,
                                ignore_sections=set(['explore more.*', 'review', 'practice', 'references']), 
                                stop_words=None, pos_words=set([]), corpus_words=None,
                                min_pos_words_in_page_name=0, min_pos_words_in_section=0, action='write')
            
        # Prepare CK-12 HTML paragraph corpus
        self.ck12html_para_corpus = '%s/CK12/OEBPS/ck12_paragraphs.txt' % corpus_dir
        if not os.path.exists(self.ck12html_para_corpus):
            # Doc per HTML paragraph
            htmlr = HtmlReader(min_chars_per_line=1, min_words_per_section=25)
            locdic = htmlr.read(htmldir='%s/CK12/OEBPS' % corpus_dir, 
                                outfile=self.ck12html_para_corpus,
                                ignore_sections=set(['explore more', 'review', 'references']), 
                                section_regexp='(?:<p[^\>]*>)|(?:<h[1-4][^>]*>([^<]+)<[\/]h[1-4]>)',
                                stop_words=None, pos_words=set([]), corpus_words=None,
                                min_pos_words_in_page_name=0, min_pos_words_in_section=0, action='write')
    
        # Prepare CK-12 text corpus
        self.ck12text_corpus = '%s/CK12/ck12_text.txt' % corpus_dir
        if not os.path.exists(self.ck12text_corpus):
            textr = TextReader(min_chars_per_line=1, min_words_per_section=25)
            locdic = textr.read(dir='%s/CK12' % corpus_dir, 
                                outfile=self.ck12text_corpus,
                                first_line_regexp='^(CHAPTER|The Big Idea)', # see "Peoples-Physics-Book-Basic_b_v10_zgo_s1.text"
                                action='write')

        # Prepare CK-12 text sentences corpus
        self.ck12text_sent_corpus = '%s/CK12/ck12_text_sentences.txt' % corpus_dir
        if not os.path.exists(self.ck12text_sent_corpus):
            textr = SentenceReader(min_chars_per_line=1, min_words_per_section=10)
            locdic = textr.read(dir='%s/CK12' % corpus_dir, 
                                outfile=self.ck12text_sent_corpus,
                                sentence_sep='. ',
                                action='write')

        # Prepare Utah OER corpus
        self.oer_corpus = '%s/UtahOER/oer_text.txt' % corpus_dir
        if not os.path.exists(self.oer_corpus):
            textr = TextReader(min_chars_per_line=1, min_words_per_section=10)
            locdic = textr.read(dir='%s/UtahOER' % corpus_dir, 
                                outfile=self.oer_corpus,
                                first_line_regexp='^.*Table of [cC]ontents*', 
                                action='write')

        # Prepare Saylor+OpenStax corpus
        self.saylor_corpus = '%s/Saylor/saylor_text.txt' % corpus_dir
        if not os.path.exists(self.saylor_corpus):
            textr = TextReader(min_chars_per_line=1, min_words_per_section=20)
            locdic = textr.read(dir='%s/Saylor' % corpus_dir, 
                                outfile=self.saylor_corpus,
                                first_line_regexp='^.*(CHAPTER|Chapter) 1.*', 
                                action='write')

        # Prepare AI2 data corpus
        self.ai2_corpus = '%s/AI2_data/ai2_corpus.txt' % corpus_dir
        if not os.path.exists(self.ai2_corpus):
            textr = SimpleLineReader(min_chars_per_line=1, min_words_per_section=2)
            locdic = textr.read(filenames=['%s/AI2_data/ai2_summary.txt' % corpus_dir], 
                                outfile=self.ai2_corpus,
                                action='write')

        # Prepare StudyStack corpus
        self.sstack_corpus = '%s/StudyStack/studystack_corpus.txt' % corpus_dir
        if not os.path.exists(self.sstack_corpus):
            textr = SimpleLineReader(min_chars_per_line=1, min_words_per_section=2)
            locdic = textr.read(filenames=['%s/StudyStack/sstack_data.text' % corpus_dir], 
                                outfile=self.sstack_corpus,
                                action='write')

        # Prepare StudyStack corpus #2 (small)
        self.sstack_corpus2 = '%s/StudyStack/studystack_corpus2.txt' % corpus_dir
        if not os.path.exists(self.sstack_corpus2):
            textr = SimpleLineReader(min_chars_per_line=1, min_words_per_section=2)
            locdic = textr.read(filenames=['%s/StudyStack/sstack_data2.text' % corpus_dir], 
                                outfile=self.sstack_corpus2,
                                action='write')

        # Prepare StudyStack corpus #3 (small+)
        self.sstack_corpus3 = '%s/StudyStack/studystack_corpus3.txt' % corpus_dir
        if not os.path.exists(self.sstack_corpus3):
            textr = SimpleLineReader(min_chars_per_line=1, min_words_per_section=2)
            locdic = textr.read(filenames=['%s/StudyStack/sstack_data3.text' % corpus_dir], 
                                outfile=self.sstack_corpus3,
                                action='write')

        # Prepare StudyStack corpus #4 (small-medium)
        self.sstack_corpus4 = '%s/StudyStack/studystack_corpus4.txt' % corpus_dir
        if not os.path.exists(self.sstack_corpus4):
            textr = SimpleLineReader(min_chars_per_line=1, min_words_per_section=2)
            locdic = textr.read(filenames=['%s/StudyStack/sstack_data4.text' % corpus_dir], 
                                outfile=self.sstack_corpus4,
                                action='write')

        # Prepare quizlet corpus
        self.quizlet_corpus = '%s/quizlet/quizlet_corpus.txt' % corpus_dir
        if not os.path.exists(self.quizlet_corpus):
            textr = SimpleLineReader(min_chars_per_line=1, min_words_per_section=2)
            locdic = textr.read(filenames=['%s/quizlet/quizlet_data.text' % corpus_dir], 
                                outfile=self.quizlet_corpus,
                                action='write')
        
        # Prepare SimpleWiki corpus
#         if False: # skip this one...
#             self.simplewiki_corpus = '%s/simplewiki/simplewiki_1.0000_0.0100_0_10_True_True_True_corpus.txt' % corpus_dir
#             if not os.path.exists(self.simplewiki_corpus):
#                 wkb = WikiCorpusBuilder(wiki_name='simplewiki', wiki_dir='%s/simplewiki'%corpus_dir, wiki_file='simplewiki-20151102-pages-articles.xml', debug_flag=False)
#                 wkb.read_categories(reread=False)
#                 wkb.read_pages_in_categories(target_categories=None, max_cat_depth=9999, 
#                                              important_categories=['Earth', 'Cellular respiration', 'DNA', 'Units of length', 'History of science', 
#                                                                    'Evolutionary biology', 'Nonmetals', 'Health', 'Charles Darwin'], reread=False)
#                 wkb.find_common_words(wiki_common_words_min_frac=1.0, wiki_uncommon_words_max_frac=0.01, use_wiki_stop_words=False, reread=False)
#                 self.simplewiki_corpus = wkb.create_corpus(train_b, valid_b, min_pos_words_in_page_name=0, min_pos_words_in_section=10, 
#                                                            only_first_section_per_page=False, use_all_pages_match_pos_word=True, use_all_pages_match_answer=True, 
#                                                            always_use_first_section=True, max_read_lines=9990000000, reread=False)
    
        # Prepare SimpleWiki corpus #2
        self.simplewiki_corpus2 = '%s/simplewiki/simplewiki_1.0000_0.0500_0_5_True_True_True_corpus.txt' % corpus_dir
        if not os.path.exists(self.simplewiki_corpus2):
            wkb = WikiCorpusBuilder(wiki_name='simplewiki', wiki_dir='%s/simplewiki'%corpus_dir, wiki_file='simplewiki-20151102-pages-articles.xml', debug_flag=False)
            wkb.read_categories(reread=False)
            wkb.read_pages_in_categories(target_categories=None, max_cat_depth=9999, 
                                         important_categories=['Earth', 'Cellular respiration', 'DNA', 'Units of length', 'History of science', 
                                                               'Evolutionary biology', 'Nonmetals', 'Health', 'Charles Darwin'], reread=False)
            wkb.find_common_words(wiki_common_words_min_frac=1.0, wiki_uncommon_words_max_frac=0.05, use_wiki_stop_words=False, reread=False)
            self.simplewiki_corpus2 = wkb.create_corpus(train_b, valid_b, min_pos_words_in_page_name=0, min_pos_words_in_section=5, 
                                                        only_first_section_per_page=False, use_all_pages_match_pos_word=True, use_all_pages_match_answer=True, 
                                                        always_use_first_section=True, max_read_lines=9990000000, reread=False)

        # Prepare SimpleWiki corpus #3
        self.simplewiki_corpus3 = '%s/simplewiki/simplewiki_1.0000_0.1000_0_3_True_True_False_corpus.txt' % corpus_dir
        if not os.path.exists(self.simplewiki_corpus3):
            wkb = WikiCorpusBuilder(wiki_name='simplewiki', wiki_dir='%s/simplewiki'%corpus_dir, wiki_file='simplewiki-20151102-pages-articles.xml', debug_flag=False)
            wkb.read_categories(reread=False)
            wkb.read_pages_in_categories(target_categories=None, max_cat_depth=9999, 
                                         important_categories=['Earth', 'Cellular respiration', 'DNA', 'Units of length', 'History of science', 
                                                               'Evolutionary biology', 'Nonmetals', 'Health', 'Charles Darwin'], reread=False)
            wkb.find_common_words(wiki_common_words_min_frac=1.0, wiki_uncommon_words_max_frac=0.1, use_wiki_stop_words=False, reread=False)
            self.simplewiki_corpus3 = wkb.create_corpus(train_b, valid_b, min_pos_words_in_page_name=0, min_pos_words_in_section=3, 
                                                        only_first_section_per_page=False, use_all_pages_match_pos_word=True, use_all_pages_match_answer=True, 
                                                        always_use_first_section=False, max_read_lines=9990000000, reread=False)
    
        # Prepare SimpleWiki corpus - page names
        self.simplewiki_corpus_pn = '%s/simplewiki/simplewiki_1.0000_0.0100_0_3_True_True_False_pn59342_corpus.txt' % corpus_dir
        if not os.path.exists(self.simplewiki_corpus_pn):
            wkb = WikiCorpusBuilder(wiki_name='simplewiki', wiki_dir='%s/simplewiki'%corpus_dir, wiki_file='simplewiki-20151102-pages-articles.xml', debug_flag=False)
            wkb.read_categories(reread=False)
            wkb.read_pages_in_categories(target_categories=None, max_cat_depth=9999, 
                                         important_categories=['Earth', 'Cellular respiration', 'DNA', 'Units of length', 'History of science', 
                                                               'Evolutionary biology', 'Nonmetals', 'Health', 'Charles Darwin'], reread=False) 
            wkb.find_common_words(wiki_common_words_min_frac=1.0, wiki_uncommon_words_max_frac=0.01, use_wiki_stop_words=False, reread=False)
            self.simplewiki_corpus_pn = wkb.create_corpus(train_b=train_b, valid_b=valid_b, min_pos_words_in_page_name=0, min_pos_words_in_section=3, 
                                                          only_first_section_per_page=False, use_all_pages_match_pos_word=True, use_all_pages_match_answer=True,
                                                          pages_to_use=self.word_sets, 
                                                          always_use_first_section=False, max_read_lines=9990000000, reread=False)
    
        # Prepare wikibooks corpus
        self.wikibooks_corpus = '%s/wikibooks/wikibooks_1.0000_0.0200_0_10_True_True_False_corpus.txt' % corpus_dir
        if not os.path.exists(self.wikibooks_corpus):
            wkb = WikiCorpusBuilder(wiki_name='wikibooks', wiki_dir='%s/wikibooks'%corpus_dir, wiki_file='enwikibooks-20151102-pages-articles.xml', debug_flag=False)
            wkb.read_categories(reread=False)
            wkb.read_pages_in_categories(target_categories=None, max_cat_depth=9999, important_categories=[], reread=False)
            wkb.find_common_words(wiki_common_words_min_frac=1.0, wiki_uncommon_words_max_frac=0.02, use_wiki_stop_words=False, reread=False)
            self.wikibooks_corpus = wkb.create_corpus(train_b, valid_b, min_pos_words_in_page_name=0, min_pos_words_in_section=10, 
                                                      only_first_section_per_page=False,  
                                                      use_all_pages_match_pos_word=True, use_all_pages_match_answer=True, 
                                                      always_use_first_section=False, max_read_lines=99900000000, reread=False)
    
        # Prepare wikibooks corpus #2
#         self.wikibooks_corpus2 = None
#         if prp_wiki_corpuses:
#             wkb = WikiCorpusBuilder(wiki_name='wikibooks', wiki_dir='%s/wikibooks'%corpus_dir, wiki_file='enwikibooks-20151102-pages-articles.xml',
#                                     debug_flag=False)
#             wkb.read_categories(reread=False)
#             wkb.read_pages_in_categories(target_categories=None, max_cat_depth=9999, important_categories=[], reread=False)
#             wkb.find_common_words(wiki_common_words_min_frac=1.0, wiki_uncommon_words_max_frac=0.1, use_wiki_stop_words=False, reread=False)
#             self.wikibooks_corpus2 = wkb.create_corpus(train_b, valid_b, min_pos_words_in_page_name=0, min_pos_words_in_section=3, 
#                                                        only_first_section_per_page=False,  
#                                                        use_all_pages_match_pos_word=True, use_all_pages_match_answer=True, 
#                                                        always_use_first_section=False, max_read_lines=99900000000, reread=False)
    
        wiki_target_categories = set([#'Nature','Medicine',
                                      'Biology','Chemistry','Physics','Astronomy','Earth',
                                      'Genetics', 'Geology', 'Health', 'Science', 'Anatomy', 'Physiology', 'Solar System',
                                      'Water', 'Meteorology',
                                      'Water in the United States', 'Agriculture in the United States', 'Environment of the United States',
                                      #'Water', 'Physical chemistry', 'Physical phenomena', 'Human homeostasis', 'Body fluids'
                                      ])
        wiki_important_categories = []
    
        # Prepare wiki corpus
#         self.wiki_corpus = None
#         if False and prp_wiki_corpuses:
#             wkb = WikiCorpusBuilder(wiki_name='wiki', wiki_dir='%s/wiki'%corpus_dir, wiki_file='enwiki-20150901-pages-articles.xml',
#                                     debug_flag=False)
#             wkb.read_categories(reread=False)
#             wkb.read_pages_in_categories(target_categories=wiki_target_categories, max_cat_depth=3, important_categories=wiki_important_categories, reread=False)
#             wkb.find_common_words(wiki_common_words_min_frac=1.0, wiki_uncommon_words_max_frac=0.02, use_wiki_stop_words=False, max_read_lines=50000000, reread=False)
#             self.wiki_corpus = wkb.create_corpus(train_b, valid_b, min_pos_words_in_page_name=0, min_pos_words_in_section=20, 
#                                                  only_first_section_per_page=False,  
#                                                  use_all_pages_match_pos_word=True, use_all_pages_match_answer=True, 
#                                                  always_use_first_section=False, max_read_lines=99900000000, reread=False)
#             gc.collect()
    
        # Prepare wiki corpus #2
#         self.wiki_corpus2 = None
#         if False and prp_wiki_corpuses:
#             wkb = WikiCorpusBuilder(wiki_name='wiki', wiki_dir='%s/wiki'%corpus_dir, wiki_file='enwiki-20150901-pages-articles.xml',
#                                     debug_flag=False)
#             wkb.read_categories(reread=False)
#             wkb.read_pages_in_categories(target_categories=wiki_target_categories, max_cat_depth=3, important_categories=wiki_important_categories, reread=False)
#             wkb.find_common_words(wiki_common_words_min_frac=1.0, wiki_uncommon_words_max_frac=0.01, use_wiki_stop_words=False, max_read_lines=50000000, reread=False)
#             self.wiki_corpus2 = wkb.create_corpus(train_b, valid_b, min_pos_words_in_page_name=0, min_pos_words_in_section=40, 
#                                                   only_first_section_per_page=False,  
#                                                   use_all_pages_match_pos_word=True, use_all_pages_match_answer=True, 
#                                                   always_use_first_section=False, max_read_lines=99900000000, reread=False)
    
        # Prepare wiki corpus #3 - only 1st section per page
        self.wiki_corpus3 = '%s/wiki/wiki_1.0000_0.0200_0_5_True_True_False_corpus.txt' % corpus_dir
        if not os.path.exists(self.wiki_corpus3):
            wkb = WikiCorpusBuilder(wiki_name='wiki', wiki_dir='%s/wiki'%corpus_dir, wiki_file='enwiki-20150901-pages-articles.xml', debug_flag=False)
            wkb.read_categories(reread=False)
            wkb.read_pages_in_categories(target_categories=wiki_target_categories, max_cat_depth=3, important_categories=wiki_important_categories, reread=False)
            wkb.find_common_words(wiki_common_words_min_frac=1.0, wiki_uncommon_words_max_frac=0.02, use_wiki_stop_words=False, max_read_lines=50000000, reread=False)
            self.wiki_corpus3 = wkb.create_corpus(train_b, valid_b, min_pos_words_in_page_name=0, min_pos_words_in_section=5, 
                                                  only_first_section_per_page=True,  
                                                  use_all_pages_match_pos_word=True, use_all_pages_match_answer=True, 
                                                  always_use_first_section=False, max_read_lines=99900000000, reread=False)

        # Prepare wiki corpus #4 - only rare words
#         self.wiki_corpus4 = None
#         if False and prp_wiki_corpuses:
#             wkb = WikiCorpusBuilder(wiki_name='wiki', wiki_dir='%s/wiki'%corpus_dir, wiki_file='enwiki-20150901-pages-articles.xml',
#                                     debug_flag=False)
#             wkb.read_categories(reread=False)
#             wkb.read_pages_in_categories(target_categories=wiki_target_categories, max_cat_depth=6, important_categories=wiki_important_categories, reread=False)
#             wkb.find_common_words(wiki_common_words_min_frac=0.1, wiki_uncommon_words_max_frac=0.0015, use_wiki_stop_words=False, max_read_lines=50000000, reread=False)
#             self.wiki_corpus4 = wkb.create_corpus(train_b, valid_b, min_pos_words_in_page_name=0, min_pos_words_in_section=10, 
#                                                   only_first_section_per_page=False,  
#                                                   use_all_pages_match_pos_word=True, use_all_pages_match_answer=True, 
#                                                   always_use_first_section=False, max_read_lines=99900000000, reread=False)

        # Prepare wiki corpus - page names
        self.wiki_corpus_pn = '%s/wiki/wiki_0.5000_0.1000_0_5_True_True_False_pn59342_corpus.txt' % corpus_dir
        if not os.path.exists(self.wiki_corpus_pn):
            wkb = WikiCorpusBuilder(wiki_name='wiki', wiki_dir='%s/wiki'%corpus_dir, wiki_file='enwiki-20150901-pages-articles.xml', debug_flag=False)
            wkb.read_categories(reread=False)
            wkb.read_pages_in_categories(target_categories=None, max_cat_depth=9999, important_categories=wiki_important_categories, reread=False) 
            wkb.find_common_words(wiki_common_words_min_frac=0.5, wiki_uncommon_words_max_frac=0.1, use_wiki_stop_words=False, max_read_lines=50000000, reread=False)
            self.wiki_corpus_pn = wkb.create_corpus(train_b, valid_b, min_pos_words_in_page_name=0, min_pos_words_in_section=5, 
                                                    only_first_section_per_page=False,  
                                                    use_all_pages_match_pos_word=True, use_all_pages_match_answer=True, 
                                                    pages_to_use=self.word_sets,
                                                    always_use_first_section=False, max_read_lines=99900000000, reread=False)

        wkb = None
        locdic = None
        gc.collect()
        
        # Prepare Lucene indexes
        self.lucene_dir1, self.lucene_parser1, self.lucene_corpus1 = None, None, None
        self.lucene_dir2, self.lucene_parser2, self.lucene_corpus2 = None, None, None
        self.lucene_dir3, self.lucene_parser3, self.lucene_corpus3 = None, None, None
        self.lucene_dir4, self.lucene_parser4, self.lucene_corpus4 = None, None, None
        self.lucene_dir5, self.lucene_parser5, self.lucene_corpus5 = None, None, None
        self.lucene_dir6, self.lucene_parser6, self.lucene_corpus6 = None, None, None
        self.lucene_dir7, self.lucene_parser7, self.lucene_corpus7 = None, None, None
        # This condition is here since I don't have PyLucene on my Windows system
        if (len(sys.argv) >= 3) and (sys.argv[1] == 'prep') and (int(sys.argv[2]) >= 21):
       
            self.lucene_dir1 = '%s/lucene_idx1' % corpus_dir
            self.lucene_parser1 = SimpleWordParser(word_func=EnglishStemmer().stem, split_words_regexp='[\-\+\*\/\,\;\:\(\)]', min_word_length=1)
            self.lucene_corpus1 = LuceneCorpus(index_dir=self.lucene_dir1, filenames=[self.sstack_corpus, self.quizlet_corpus], parser=self.lucene_parser1)
            if not os.path.exists(self.lucene_dir1):
                 self.lucene_corpus1.prp_index()
                 
            self.lucene_dir2 = '%s/lucene_idx2' % corpus_dir
            self.lucene_parser2 = SimpleWordParser(word_func=LancasterStemmer().stem, split_words_regexp='[\-\+\*\/\,\;\:\(\)]', min_word_length=1)
            self.lucene_corpus2 = LuceneCorpus(index_dir=self.lucene_dir2, filenames=[self.sstack_corpus3, self.quizlet_corpus, self.ck12text_corpus,
                                                                                      self.wiki_corpus_pn, self.simplewiki_corpus_pn], parser=self.lucene_parser2)
            if not os.path.exists(self.lucene_dir2):
                 self.lucene_corpus2.prp_index()
                 
            self.lucene_dir3 = '%s/lucene_idx3' % corpus_dir
            self.lucene_parser3 = SimpleWordParser(word_func=PorterStemmer().stem, split_words_regexp='[\-\+\*\/\,\;\:\(\)]', min_word_length=1)
            self.lucene_corpus3 = LuceneCorpus(index_dir=self.lucene_dir3, filenames=[self.sstack_corpus4, self.quizlet_corpus, self.oer_corpus, self.saylor_corpus,
                                                                                      self.ck12html_para_corpus, self.ai2_corpus], 
                                               parser=self.lucene_parser3, similarity=BM25Similarity())
            if not os.path.exists(self.lucene_dir3):
                 self.lucene_corpus3.prp_index()

            self.lucene_dir4 = '%s/lucene_idx4' % corpus_dir
            self.lucene_parser4 = SimpleWordParser(word_func=EnglishStemmer().stem, split_words_regexp='[\-\+\*\/\,\;\:\(\)]', min_word_length=1)
            self.lucene_corpus4 = LuceneCorpus(index_dir=self.lucene_dir4, filenames=[self.sstack_corpus, self.quizlet_corpus, self.ck12html_corpus], 
                                               parser=self.lucene_parser4, similarity=BM25Similarity())
            if not os.path.exists(self.lucene_dir4):
                 self.lucene_corpus4.prp_index()

            self.lucene_dir5 = '%s/lucene_idx5' % corpus_dir
            self.lucene_parser5 = SimpleWordParser(word_func=LancasterStemmer().stem, split_words_regexp='[\-\+\*\/\,\;\:\(\)]', min_word_length=1)
            self.lucene_corpus5 = LuceneCorpus(index_dir=self.lucene_dir5, filenames=[self.wiki_corpus3, self.simplewiki_corpus3], 
                                               parser=self.lucene_parser5, similarity=None)
            if not os.path.exists(self.lucene_dir5):
                 self.lucene_corpus5.prp_index()
 
            self.lucene_dir6 = '%s/lucene_idx6' % corpus_dir
            self.lucene_parser6 = SimpleWordParser(word_func=EnglishStemmer().stem, split_words_regexp='[\-\+\*\/\,\;\:\(\)]', min_word_length=1)
            self.lucene_corpus6 = LuceneCorpus(index_dir=self.lucene_dir6, filenames=[self.ck12text_corpus, self.saylor_corpus, self.oer_corpus, self.ai2_corpus], 
                                               parser=self.lucene_parser6, similarity=BM25Similarity())
            if not os.path.exists(self.lucene_dir6):
                 self.lucene_corpus6.prp_index()
 
            self.lucene_dir7 = '%s/lucene_idx7' % corpus_dir
            self.lucene_parser7 = SimpleWordParser(word_func=PorterStemmer().stem, split_words_regexp='[\-\+\*\/\,\;\:\(\)]', min_word_length=1)
            self.lucene_corpus7 = LuceneCorpus(index_dir=self.lucene_dir7, filenames=[self.sstack_corpus2, self.wiki_corpus_pn, self.simplewiki_corpus_pn,
                                                                                      self.ck12html_para_corpus, self.oer_corpus], 
                                               parser=self.lucene_parser7, similarity=None)
            if not os.path.exists(self.lucene_dir7):
                 self.lucene_corpus7.prp_index()
        
        print '-> Finished preparing corpuses'
    
    ALL_FEATURE_TYPES = {# "BASIC" features extracted from the questions and answers, w/o external corpus:
                         'BASIC': 0, 
                         # Features computed using my search functions:
                         'ck-hp_saylor.triplets.1': 1, 'ck-hp_saylor_oer.triplets.1': 2,  'qz_ck-ts.1': 3,  
                         'st2_qz_oer_ck-hp.1': 4, 'st2_qz_oer_ck-t.triplets.1': 5, 
                         'st2_qz_wk-pn_oer_ck-h.pairs.1': 6, 'st_qz.1': 7, 'st_qz.pairs2.1': 8, 'st_qz_ai.1': 9, 
                         'st_qz_saylor_ck-t.a1_vs_a2.1': 10, 'sw-pn_qz.1': 11,  
                         'sw-pn_ss_ck-t_ai.1': 12, 'sw2_ck-ts.1': 13, 'tr_st_qz.1': 14, 'tr_st_qz.2': 15, 'wk3_sw3.1': 16,
                         'wk-pn_sw-pn_wb.a1_vs_a2.1': 17, 'st_qz.triplets13.1': 18, 'st_qz.Z': 19, 'wk-pn_sw-pn.1': 20,
                         # Features computed using PyLucene:
                         'lucene.1': 21, 'lucene.2': 22, 'lucene.3': 23, 'lucene.4': 24, 'lucene.5': 25, 'lucene.6': 26, 'lucene.7': 27, 
                         }
        
    def prepare_features(self, dataf_q, dataf_b, train_df, aux_b, cache_dir, ftypes=None):
        '''
        Compute one or more features by running the relevant search function.
        aux_b - an additional binary data source with possibly same questions as in dataf_b, to save computations
        '''
        if ftypes is not None:
            assert (len(ftypes) > 0) and (set(ftypes).issubset(FeatureExtractor.ALL_FEATURE_TYPES.values())), \
                    'Feature types should be non-empty subset of:\n%s' % FeatureExtractor.ALL_FEATURE_TYPES 
        self.cache_dir = '%s/%s' % (self.base_dir, cache_dir)
        create_dirs([self.cache_dir])

        if self.print_level > 0:
            print '-> Preparing features, cache dir = %s' % cache_dir
        
        locdic = None
        stemmer1 = PorterStemmer()
        stem1 = stemmer1.stem
        stemmer2 = LancasterStemmer()
        stem2 = stemmer2.stem
        #This is probably a Krovetz Stemmer from the looks of it. Stems a word to look like a meaningful english word. 
        stemmer3 = EnglishStemmer()
        stem3 = stemmer3.stem

        tag_weights1 = {'NN':1.5,'NNP':1.5,'NNPS':1.5,'NNS':1.5, 'VB':1.3,'VBD':1.3,'VBG':1.3,'VBN':1.3,'VBP':1.3,'VBZ':1.3, 
                        'JJ':1.0,'JJR':1.0,'JJS':1.0, 'RB':1.0,'RBR':1.0,'RBS':1.0,'RP':1.0}
        tag_weight_func1 = lambda tag: tag_weights1.get(tag, 0.8)

        tag_weights2 = {'NN':2.0,'NNP':2.0,'NNPS':2.0,'NNS':2.0, 'VB':1.5,'VBD':1.5,'VBG':1.5,'VBN':1.5,'VBP':1.5,'VBZ':1.5, 
                        'JJ':1.0,'JJR':1.0,'JJS':1.0, 'RB':0.8,'RBR':0.8,'RBS':0.8,'RP':0.8}
        tag_weight_func2 = lambda tag: tag_weights2.get(tag, 0.5)

        swr = '[\-\+\*\/\,\;\:\(\)]' # split_words_regexp

        if 'correctAnswer' in dataf_q.columns:
            targets_q = dict(zip(dataf_q.index,dataf_q['correctAnswer']))
            targets_b = np.array(dataf_b['correct'])
        else:
            targets_q, targets_b = None, None
        
        train_b = dataf_b
       
        # ==================================================================================================================================
        # Compute funcs with various combinations of corpora, parsers, and score params
        # ==================================================================================================================================
        ds_funcs = {
                    # 1: 0.4448 
                    'ck-hp_saylor.triplets.1': {'corpora': [self.ck12html_para_corpus, self.saylor_corpus], 
                                                'parser': SimpleWordParser(word_func=stem3, tuples=[1,2,3], split_words_regexp=swr),  
                                                'num_words_qst': [None]+range(2,60), 'num_words_ans': [None]+range(2,40), 'score': 'weights', 'prob_type': 'tf-idf', 'tf_log_flag': True, 
                                                'score_params': {'coeffs': lambda n: 1.0/(1.0+np.arange(n)), 'calc_over_vs_under': True, 
                                                                 'minword1_coeffs': lambda mw,nw: ((1.0+mw)/(1.0+nw))**0.4, 'minword2_coeffs': lambda mw,nw: ((1.0+mw)/(1.0+nw))**0.7},
                                                'recalc': False, 'skip': False},
                    # 2: 0.4500 
                    'ck-hp_saylor_oer.triplets.1': {'corpora': [self.ck12html_para_corpus, self.saylor_corpus, self.oer_corpus], 
                                                    'parser': SimpleWordParser(word_func=stem2, tuples=[1,2,3], min_word_length=1),   
                                                    'num_words_qst': [None]+range(2,60), 'num_words_ans': [None]+range(2,40), 'score': 'weights', 'prob_type': 'tf-idf', 'tf_log_flag': True, 
                                                    'score_params': {'coeffs': lambda n: 1.0/(1.0+np.arange(n))**0.7, 'calc_over_vs_under': True, 
                                                                     'minword1_coeffs': lambda mw,nw: np.sqrt((1.0+mw)/(1.0+nw)), 'minword2_coeffs': lambda mw,nw: np.sqrt((1.0+mw)/(1.0+nw))},
                                                    'recalc': False, 'skip': False},
                    # 3: 0.4164
                    'qz_ck-ts.1': {'corpora': [self.quizlet_corpus, self.ck12text_sent_corpus], 
                                   'parser': NltkTokenParser(word_func=lambda word,tag: stem3(word), word_func_requires_tag=False,
                                                             tuples=[1], tag_weight_func=tag_weight_func1),  
                                   'num_words_qst': [None]+range(2,30), 'num_words_ans': [None]+range(2,20), 'score': 'hg', 'prob_type': 'tf-idf', 'tf_log_flag': False, 
                                   'score_params': {'minword1_coeffs': lambda mw,nw: np.sqrt((1.0+mw)/(1.0+nw)), 'minword2_coeffs': lambda mw,nw: np.sqrt((1.0+mw)/(1.0+nw))},
                                   'recalc': False, 'skip': False},
                    # 4: 0.4720
                    'st2_qz_oer_ck-hp.1': {'corpora': [self.sstack_corpus2, self.quizlet_corpus, self.oer_corpus, self.ck12html_para_corpus],
                                           'parser': SimpleWordParser(word_func=None, min_word_length=1),
                                           'num_words_qst': [None]+range(2,30), 'num_words_ans': [None]+range(2,20), 'score': 'weights', 'prob_type': 'tf-idf', 'tf_log_flag': False, 
                                           'score_params': {'norm': lambda w: w**2/(np.sum(w**2) + 0.0), 
                                                            'minword1_coeffs': lambda mw,nw: ((1.0+mw)/(1.0+nw))**1.4, 'minword2_coeffs': lambda mw,nw: ((1.0+mw)/(1.0+nw))},
                                           'recalc': False, 'skip': False},
                    # 5: 0.5256
                    'st2_qz_oer_ck-t.triplets.1': {'corpora': [self.sstack_corpus2, self.quizlet_corpus, self.oer_corpus, self.ck12text_corpus],
                                                   'parser': SimpleWordParser(word_func=stem3, tuples=[1,2,3], split_words_regexp=swr, min_word_length=1),  
                                                   'num_words_qst': [None]+range(2,60), 'num_words_ans': [None]+range(2,40), 'score': 'weights', 'prob_type': 'tf-idf', 'tf_log_flag': False, 
                                                   'score_params': {'coeffs': lambda n: 2.0/(2.0+np.arange(n))**0.7, 'calc_over_vs_under': True, 
                                                                    'minword1_coeffs': lambda mw,nw: ((1.0+mw)/(1.0+nw))**0.4, 'minword2_coeffs': lambda mw,nw: ((1.0+mw)/(1.0+nw))**0.6},
                                                   'recalc': False, 'skip': False},
                    # 6: 0.5388
                    'st2_qz_wk-pn_oer_ck-h.pairs.1': {#'corpora': [self.sstack_corpus2, self.quizlet_corpus, self.wiki_corpus_pn, self.oer_corpus, self.ck12html_corpus],
                                                      'corpora': [self.sstack_corpus4, self.quizlet_corpus, self.wiki_corpus_pn, self.oer_corpus, self.ck12html_corpus], 
                                                      'parser': SimpleWordParser(word_func=stem3, tuples=[1,2], split_words_regexp=swr, min_word_length=1),  
                                                      'num_words_qst': [None]+range(2,50), 'num_words_ans': [None]+range(2,30), 'score': 'weights', 'prob_type': 'tf-idf', 'tf_log_flag': False, 
                                                      'score_params': {'coeffs': lambda n: 1.0/(1.0+np.arange(n))**0.8, 'calc_over_vs_under': True, 
                                                                       'minword1_coeffs': lambda mw,nw: ((1.0+mw)/(1.0+nw))**0.3, 'minword2_coeffs': lambda mw,nw: ((1.0+mw)/(1.0+nw))**0.7},
                                                      'recalc': False, 'skip': False},
                    # 7: 0.5580
                    'st_qz.1': {'corpora': [self.sstack_corpus, self.quizlet_corpus], 
                                'parser': SimpleWordParser(word_func=stem3, tuples=[1]),  
                                'num_words_qst': [None]+range(2,30), 'num_words_ans': [None]+range(2,20), 'score': 'weights', 'prob_type': 'tf-idf', 'tf_log_flag': True, 
                                'score_params': {'coeffs': lambda n: 1.0/(1.0+np.arange(n)), 'calc_over_vs_under': True, 
                                                 'minword1_coeffs': lambda mw,nw: ((1.0+mw)/(1.0+nw)), 'minword2_coeffs': lambda mw,nw: ((1.0+mw)/(1.0+nw))},
                                'recalc': False, 'skip': False},
                    # 8: 0.3340
                    'st_qz.pairs2.1': {'corpora': [self.sstack_corpus, self.quizlet_corpus], 
                                       'parser': SimpleWordParser(word_func=stem3, tuples=[2], min_word_length=1),  
                                       'num_words_qst': [None]+range(2,30), 'num_words_ans': [None]+range(2,20), 'score': 'weights', 'prob_type': 'tf-idf', 'tf_log_flag': True, 
                                       'score_params': {'coeffs': lambda n: (2.0/(2.0+np.arange(n)))**1.4, 'calc_over_vs_under': True, 
                                                        'minword1_coeffs': lambda mw,nw: ((1.0+mw)/(1.0+nw))**0.8, 'minword2_coeffs': lambda mw,nw: ((1.0+mw)/(1.0+nw))**0.9},
                                       'recalc': False, 'skip': False},                    
                    # 9: 0.5184
                    'st_qz_ai.1': {'corpora': [self.sstack_corpus, self.quizlet_corpus, self.ai2_corpus], 
                                   'parser': NltkTokenParser(word_func=lambda word,tag: stem2(word), word_func_requires_tag=False,
                                                             tuples=[1], tag_weight_func=tag_weight_func1),
                                   'num_words_qst': [None]+range(2,30), 'num_words_ans': [None]+range(2,20), 'score': 'weights', 'prob_type': 'tf-idf', 'tf_log_flag': False, 
                                   'score_params': {'coeffs': lambda n: 1.0/(1.0+np.arange(n))**0.3, 'calc_over_vs_under': True,
                                                    'norm': lambda w: w**2/(np.sum(w**2) + 0.0), 
                                                    'minword1_coeffs': lambda mw,nw: ((1.0+mw)/(1.0+nw))**2, 'minword2_coeffs': lambda mw,nw: ((1.0+mw)/(1.0+nw))**1.5},
                                   'recalc': False, 'skip': False},                    
                    # 10: 0.3080
                    'st_qz_saylor_ck-t.a1_vs_a2.1': {'corpora': [self.sstack_corpus, self.quizlet_corpus, self.saylor_corpus, self.ck12text_corpus], 
                                                     'parser': SimpleWordParser(word_func=stem2, tuples=[1]),  
                                                     'num_words_qst': [None]+range(2,30), 'num_words_ans': [None]+range(2,20), 'score': 'weights', 'prob_type': 'tf-idf', 'tf_log_flag': True, 
                                                     'score_params': {'coeffs': lambda n: 1.0/(1.0+np.arange(n)), 'calc_over_vs_under': True, 
                                                                      'minword1_coeffs': lambda mw,nw: ((1.0+mw)/(1.0+nw)), 'minword2_coeffs': lambda mw,nw: ((1.0+mw)/(1.0+nw))},
                                                     'apairs': {'sim_scores_comb_weights': ([10, 3, 1], [1, 3, 10]), 'search_type': 'a1_vs_a2'},
                                                     'recalc': False, 'skip': False},
                    # 11: 0.4628
                    'sw-pn_qz.1': {'corpora': [self.simplewiki_corpus_pn, self.quizlet_corpus], 
                                   'parser': SimpleWordParser(word_func=stem3, tuples=[1], split_words_regexp=swr),  
                                   'num_words_qst': [None]+range(2,30), 'num_words_ans': [None]+range(2,20), 'score': 'weights', 'prob_type': 'tf-idf', 'tf_log_flag': True, 
                                   'score_params': {'coeffs': lambda n: 1.0/(1.0+np.arange(n))**2.0,  
                                                    'minword1_coeffs': lambda mw,nw: np.sqrt((1.0+mw)/(1.0+nw)), 'minword2_coeffs': lambda mw,nw: ((1.0+mw)/(1.0+nw))**0.8},
                                   'recalc': False, 'skip': False},
                    # 12: 0.4620 
                    'sw-pn_ss_ck-t_ai.1': {'corpora': [self.simplewiki_corpus_pn, self.sstack_corpus, self.ck12text_corpus, self.ai2_corpus], 
                                           'parser': SimpleWordParser(word_func=stem2, tuples=[1], split_words_regexp=swr),  
                                           'num_words_qst': [None]+range(2,30), 'num_words_ans': [None]+range(2,20), 'score': 'weights', 'prob_type': 'tf-idf', 'tf_log_flag': True, 
                                           'score_params': {'coeffs': lambda n: 1.0/(1.0+np.arange(n))**0.3,  
                                                            'minword1_coeffs': lambda mw,nw: ((1.0+mw)/(1.0+nw))**1.8, 'minword2_coeffs': lambda mw,nw: ((1.0+mw)/(1.0+nw))**1.5},
                                           'recalc': False, 'skip': False},
                    # 13: 0.4192
                    'sw2_ck-ts.1': {'corpora': [self.simplewiki_corpus2, self.ck12text_sent_corpus], 
                                    'parser': SimpleWordParser(word_func=stem2, tuples=[1], split_words_regexp=swr),  
                                    'num_words_qst': [None]+range(2,30), 'num_words_ans': [None]+range(2,20), 'score': 'weights', 'prob_type': 'tf-idf', 'tf_log_flag': False, 
                                    'score_params': {'minword1_coeffs': lambda mw,nw: ((1.0+mw)/(1.0+nw)), 'minword2_coeffs': lambda mw,nw: ((1.0+mw)/(1.0+nw))**1.8,
                                                     'norm': lambda w: w**2/(np.sum(w**2) + 0.0)},
                                    'recalc': False, 'skip': False},
                    # 14: 0.5468 
                    'tr_st_qz.1': {'corpora': [self.sstack_corpus, self.quizlet_corpus], 
                                   'parser': SimpleWordParser(word_func=stem1, tuples=[1], split_words_regexp=swr, min_word_length=1),  
                                   'num_words_qst': [None]+range(2,30), 'num_words_ans': [None]+range(2,15), 'score': 'weights', 'prob_type': 'tf-idf', 'tf_log_flag': True, 
                                   'score_params': {'coeffs': lambda n: np.sqrt(1.0/(1.0+np.arange(n))), 'calc_over_vs_under': True, 
                                                    'minword1_coeffs': lambda mw,nw: ((1.0+mw)/(1.0+nw))**1.5, 'minword2_coeffs': lambda mw,nw: ((1.0+mw)/(1.0+nw))**2},
                                   'recalc': False, 'skip': False, 'train': True},
                    # 15: 0.5088
                    'tr_st_qz.2': {'corpora': [self.sstack_corpus, self.quizlet_corpus], 
                                   'parser': NltkTokenParser(word_func=None, word_func_requires_tag=False, tuples=[1], tag_weight_func=tag_weight_func1),  
                                   'num_words_qst': [None]+range(2,30), 'num_words_ans': [None]+range(2,15), 'score': 'weights', 'prob_type': 'tf-idf', 'tf_log_flag': False, 
                                   'score_params': {'coeffs': lambda n: np.ones(n), 'calc_over_vs_under': True, 
                                                    'norm': lambda w: w**2/(np.sum(w**2) + 0.0),
                                                    'minword1_coeffs': lambda mw,nw: ((1.0+mw)/(1.0+nw))**0.75, 'minword2_coeffs': lambda mw,nw: ((1.0+mw)/(1.0+nw))},
                                   'recalc': False, 'skip': False, 'train': True},
                    # 16: 0.4028
                    'wk3_sw3.1': {'corpora': [self.wiki_corpus3, self.simplewiki_corpus3], 
                                  'parser': SimpleWordParser(word_func=stem1, tuples=[1], split_words_regexp=swr),  
                                  'num_words_qst': [None]+range(3,40), 'num_words_ans': [None]+range(2,20), 'score': 'weights', 'prob_type': 'tf-idf', 'tf_log_flag': True, 
                                  'score_params': {'coeffs': lambda n: (1.0/(1.0+np.arange(n)))**1.2},
                                  'recalc': False, 'skip': False},
                    # 17: over 0.2872, under 0.2908                              
                    'wk-pn_sw-pn_wb.a1_vs_a2.1': {'corpora': [self.wiki_corpus_pn, self.simplewiki_corpus_pn, self.wikibooks_corpus], 
                                                  'parser': SimpleWordParser(word_func=stem3, tuples=[1], split_words_regexp=swr),  
                                                  'num_words_qst': [None]+range(1,30), 'num_words_ans': [None]+range(1,20), 'score': 'weights', 'prob_type': 'tf-idf', 'tf_log_flag': False, 
                                                  'score_params': {'coeffs': lambda n: 1.0/(1.0+np.arange(n))**0.7, 'calc_over_vs_under': True, 
                                                                   'minword1_coeffs': lambda mw,nw: ((1.0+mw)/(1.0+nw))**1.5, 'minword2_coeffs': lambda mw,nw: ((1.0+mw)/(1.0+nw))**1.5},
                                                  'apairs': {'sim_scores_comb_weights': ([12, 2, 1], [1, 2, 12]), 'search_type': 'a1_vs_a2'},
                                                  'recalc': False, 'skip': False},
                    # 18: 0.5500 
                    'st_qz.triplets13.1': {'corpora': [self.sstack_corpus, self.quizlet_corpus], 
                                           'parser': SimpleWordParser(word_func=stem1, tuples=[1,3], min_word_length=1, split_words_regexp=swr),  
                                           'num_words_qst': [None]+range(2,60), 'num_words_ans': [None]+range(2,30), 'score': 'weights', 'prob_type': 'tf-idf', 'tf_log_flag': True, 
                                           'score_params': {'coeffs': lambda n: 3.0/(3.0+np.arange(n)), 'calc_over_vs_under': True, 
                                                            'minword1_coeffs': lambda mw,nw: ((3.0+mw)/(3.0+nw))**1.2, 'minword2_coeffs': lambda mw,nw: ((3.0+mw)/(3.0+nw))**1.5},
                                           'recalc': False, 'skip': False},
                    # 19: 0.45126
                    'st_qz.Z': {'corpora': [self.sstack_corpus, self.quizlet_corpus],
                                'parser': SimpleWordParser(word_func=stem3, split_words_regexp=swr),
                                'norm_scores': True,
                                'recalc': False, 'skip': False, 'zscore': True},
                    # 20: 0.3884
                    'wk-pn_sw-pn.1': {'corpora': [self.wiki_corpus_pn, self.simplewiki_corpus_pn], 
                                      'parser': SimpleWordParser(word_func=None, tuples=[1], split_words_regexp=swr),  
                                      'num_words_qst': [None]+range(2,40), 'num_words_ans': [None]+range(2,30), 'score': 'weights', 'prob_type': 'tf-idf', 'tf_log_flag': False, 
                                      'score_params': {'norm': lambda w: w**1.5/(np.sum(w**1.5) + 0.0)},
                                      'recalc': False, 'skip': False},
                    # 21: 0.5424
                    'lucene.1': {'lucene_corpus': self.lucene_corpus1,
                                 'parser': self.lucene_parser1,  
                                 'max_docs': 1000, 'weight_func': lambda n: 1.0/(1.0+np.arange(n))**1.5, 'score_func': None, 'norm_scores': True,
                                 'recalc': False, 'skip': False, 'lucene': True},
                    # 22: 0.5304
                    'lucene.2': {'lucene_corpus': self.lucene_corpus2,
                                 'parser': self.lucene_parser2,  
                                 'max_docs': 500, 'weight_func': lambda n: 1.0/(1.0+np.arange(n))**1.2, 'score_func': lambda s: s**1.5, 'norm_scores': True,
                                 'recalc': False, 'skip': False, 'lucene': True},
                    # 23: 0.5388 
                    'lucene.3': {'lucene_corpus': self.lucene_corpus3,
                                 'parser': self.lucene_parser3,  
                                 'max_docs': 3000, 'weight_func': lambda n: 1.0/(3.0+np.arange(n))**1.4, 'score_func': lambda s: (s/100.0)**5, 'norm_scores': True,
                                 'recalc': False, 'skip': False, 'lucene': True},
                    # 24: 0.5500 
                    'lucene.4': {'lucene_corpus': self.lucene_corpus4,
                                 'parser': self.lucene_parser4,  
                                 'max_docs': 2500, 'weight_func': lambda n: 1.0/(1.0+np.arange(n))**2.2, 'score_func': lambda s: (s/100.0)**4, 'norm_scores': True,
                                 'recalc': False, 'skip': False, 'lucene': True},
                    # 25: 0.4292 
                    'lucene.5': {'lucene_corpus': self.lucene_corpus5,
                                 'parser': self.lucene_parser5,  
                                 'max_docs': 750, 'weight_func': lambda n: 1.0/(2.0+np.arange(n))**1.6, 'score_func': lambda s: (s/10.0)**2.5, 'norm_scores': True,
                                 'recalc': False, 'skip': False, 'lucene': True},
                    # 26: 0.4672 
                    'lucene.6': {'lucene_corpus': self.lucene_corpus6,
                                 'parser': self.lucene_parser6,  
                                 'max_docs': 800, 'weight_func': lambda n: 1.0/(5.0+np.arange(n))**2, 'score_func': lambda s: (s/10.0)**3, 'norm_scores': True,
                                 'recalc': False, 'skip': False, 'lucene': True},
                    # 27: 0.4684  
                    'lucene.7': {'lucene_corpus': self.lucene_corpus7,
                                 'parser': self.lucene_parser7,  
                                 'max_docs': 250, 'weight_func': lambda n: 1.0/(10.0+np.arange(n)), 'score_func': lambda s: (s+2.0)**3.4, 'norm_scores': True,
                                 'recalc': False, 'skip': False, 'lucene': True},
                   }
        
        check_same_question = not set(train_b['ID']).isdisjoint(train_df['ID'])
        
        for fn,params in sorted(ds_funcs.iteritems()):
            if params['skip']: continue
            if (ftypes is not None) and (FeatureExtractor.ALL_FEATURE_TYPES[fn] not in ftypes): continue
            if params.has_key('zscore') or params.has_key('lucene'):
                func_name = fn
            else:
                func_name = ['%s_over'%fn, '%s_under'%fn]
            if self.print_level > 1:
                print 'Computing features: %s' % str(func_name)
            if params.has_key('corpora'):
                locdic = lambda: CorpusReader.build_locdic_from_outfile(filename=params['corpora'], parser=params['parser'], 
                                                                        min_word_docs_frac=0, max_word_docs_frac=1, min_word_count_frac=0, max_word_count_frac=1)
            else:
                locdic = None
            norm_scores = params['norm_scores'] if params.has_key('norm_scores') else self.norm_scores_default
            self.recalc = params['recalc']
            #print 'recalc = %s' % self.recalc
            if params.has_key('train'):
                assert params['train']
                self.add_answer_func(train_b, aux_b,
                                     func=AnswersTrainDoubleSearchFunc(train_df[train_df['correct']==1], check_same_question=check_same_question, 
                                                                       base_locdic=locdic, parser=params['parser'],
                                                                       num_words_qst=params['num_words_qst'], num_words_ans=params['num_words_ans'], 
                                                                       score=params['score'], score_params=params['score_params'], norm_scores=norm_scores,
                                                                       prob_type=params['prob_type'], tf_log_flag=params['tf_log_flag']),  
                                     name=func_name)
            elif params.has_key('train0'):
                assert params['train0'] and (locdic is None)
                self.add_answer_func(train_b, aux_b,
                                     func=AnswersTrainDoubleSearchFunc(train_df[train_df['correct']==0], check_same_question=check_same_question, 
                                                                       parser=params['parser'],
                                                                       num_words_qst=params['num_words_qst'], num_words_ans=params['num_words_ans'], 
                                                                       score=params['score'], score_params=params['score_params'], norm_scores=norm_scores,
                                                                       prob_type=params['prob_type'], tf_log_flag=params['tf_log_flag']),  
                                     name=func_name)
            elif params.has_key('apairs'): 
                self.add_answer_func(train_b, aux_b,
                                     func=AnswersPairsDoubleSearchFunc(locdic=locdic, parser=params['parser'],
                                                                       num_words_qst=params['num_words_qst'], num_words_ans=params['num_words_ans'], 
                                                                       score=params['score'], score_params=params['score_params'], norm_scores=norm_scores,
                                                                       prob_type=params['prob_type'], tf_log_flag=params['tf_log_flag'], 
                                                                       sim_scores_comb_weights=params['apairs']['sim_scores_comb_weights'], search_type=params['apairs']['search_type']),
                                     name=func_name) 

            elif params.has_key('zscore'): 
                self.add_answer_func(train_b, aux_b, 
                                     func=AnswersWordZscoreFunc(locdic=locdic, parser=params['parser'], norm_scores=norm_scores), 
                                     name=func_name)
            elif params.has_key('lucene'):     
                self.add_answer_func(train_b, aux_b,
                                     func=AnswersLuceneSearchFunc(lucene_corpus=params['lucene_corpus'], parser=params['parser'], 
                                                                  max_docs=params['max_docs'], weight_func=params['weight_func'], score_func=params['score_func'], 
                                                                  norm_scores=norm_scores),
                                     name=func_name)
            else:
                self.add_answer_func(train_b, aux_b,
                                     func=AnswersDoubleSearchFunc(locdic=locdic, parser=params['parser'],
                                                                  num_words_qst=params['num_words_qst'], num_words_ans=params['num_words_ans'], 
                                                                  score=params['score'], score_params=params['score_params'], norm_scores=norm_scores,
                                                                  prob_type=params['prob_type'], tf_log_flag=params['tf_log_flag']),  
                                     name=func_name) 
            if ((self.print_level > 1) or self.recalc) and (targets_q is not None):
                if params.has_key('zscore') or params.has_key('lucene'):
                    print ' AUC of %s: %.4f' % (func_name, calc_auc(targets_b, train_b[func_name], two_sides=True))
                    print ' Accuracy of %s: %.4f' % (func_name, calc_accuracy(targets_q, get_predictions_from_binary_dataf(train_b, func_name, direction='max')))
                else:
                    print ' AUC of %s: %.4f' % (func_name[0], calc_auc(targets_b, train_b[func_name[0]], two_sides=True))
                    print ' Accuracy of %s: %.4f' % (func_name[0], calc_accuracy(targets_q, get_predictions_from_binary_dataf(train_b, func_name[0], direction='max')))
                    print ' AUC of %s: %.4f' % (func_name[1], calc_auc(targets_b, train_b[func_name[1]], two_sides=True))
                    print ' Accuracy of %s: %.4f' % (func_name[1], calc_accuracy(targets_q, get_predictions_from_binary_dataf(train_b, func_name[1], direction='max')))
            gc.collect()
            self.recalc = False
            
        # ==================================================================================================================================
        # Compute the "basic" features, ie, those obtained from the questions+answers, without any external corpus
        # ==================================================================================================================================
        if (ftypes is None) or (FeatureExtractor.ALL_FEATURE_TYPES['BASIC'] in ftypes):
            simple_parser   = SimpleWordParser(word_func=None , ignore_special_words=False, min_word_length=1)
            pairs_parser    = SimpleWordParser(word_func=None , ignore_special_words=False, min_word_length=1, tuples=[1,2])
            stemmed_parser  = SimpleWordParser(word_func=stem1, ignore_special_words=True , min_word_length=1)
            stemmed_parserB = SimpleWordParser(word_func=stem3, ignore_special_words=True , min_word_length=2, split_words_regexp=swr)
            stemmed_parserC = SimpleWordParser(word_func=stem2, ignore_special_words=True , min_word_length=2, split_words_regexp=swr)
            stemmed_parser2 = SimpleWordParser(word_func=stem2, ignore_special_words=False, min_word_length=1)
            stemmed_parser3 = SimpleWordParser(word_func=stem3, ignore_special_words=False, min_word_length=1)
            stemmed_pairs_parser3 = SimpleWordParser(word_func=stem3, ignore_special_words=False, min_word_length=1, tuples=[1,2], split_words_regexp=swr)
        
            func_name = 'ans_in_qst'
            self.add_answer_func(train_b, aux_b, func=AnswersInQuestionFunc(), name=func_name)
            func_name = 'ans_in_qst_stem'
            self.add_answer_func(train_b, aux_b, func=AnswersInQuestionFunc(parser=stemmed_parser), name=func_name)
        
            func_name = 'ans_in_ans'
            self.add_answer_func(train_b, aux_b, func=AnswersInAnswersFunc(), name=func_name)
            func_name = 'ans_in_ans_stem'
            self.add_answer_func(train_b, aux_b, func=AnswersInAnswersFunc(parser=stemmed_parser), name=func_name)
                
            func_name = 'ans_count'
            self.add_answer_func(train_b, aux_b, func=AnswerCountFunc(train_df, check_same_question=check_same_question, count_type='count'), name=func_name) # 0.2532 (0.5313)
            func_name = 'ans_words_stem_count'
            self.add_answer_func(train_b, aux_b, func=AnswerCountFunc(train_df, check_same_question=check_same_question, count_type='count', parser=stemmed_parser, single_words=True), name=func_name)
            func_name = 'ans_words_stem_count_nonorm'
            self.add_answer_func(train_b, aux_b, func=AnswerCountFunc(train_df, check_same_question=check_same_question, count_type='count', parser=stemmed_parserC, single_words=True, norm_scores=False), name=func_name)
            func_name = 'ans_qst_words_stem_count'
            self.add_answer_func(train_b, aux_b, func=AnswerCountFunc(train_df, check_same_question=check_same_question, count_type='count', parser=stemmed_parserB, single_words=True, use_questions=True), name=func_name)
                
            func_name = 'ans_correct'
            self.add_answer_func(train_b, aux_b, func=AnswerCountFunc(train_df, check_same_question=check_same_question, count_type='correct'), name=func_name)
            func_name = 'ans_words_stem_correct'
            self.add_answer_func(train_b, aux_b, func=AnswerCountFunc(train_df, check_same_question=check_same_question, count_type='correct', parser=stemmed_parser, single_words=True), name=func_name)
            func_name = 'ans_words_stem_correct_nonorm'
            self.add_answer_func(train_b, aux_b, func=AnswerCountFunc(train_df, check_same_question=check_same_question, count_type='correct', parser=stemmed_parserB, single_words=True, norm_scores=False), name=func_name)
            func_name = 'ans_qst_words_stem_correct'
            self.add_answer_func(train_b, aux_b, func=AnswerCountFunc(train_df, check_same_question=check_same_question, count_type='correct', parser=stemmed_parser3, single_words=True, use_questions=True), name=func_name)
            func_name = 'ans_words_stem_pairs_correct'
            self.add_answer_func(train_b, aux_b, func=AnswerCountFunc(train_df, check_same_question=check_same_question, count_type='correct', parser=stemmed_pairs_parser3, single_words=True), name=func_name)
        
            func_name = 'ans_pval'
            self.add_answer_func(train_b, aux_b, func=AnswerCountFunc(train_df, check_same_question=check_same_question, count_type='pval'), name=func_name)
            func_name = 'ans_stem_pval'
            self.add_answer_func(train_b, aux_b, func=AnswerCountFunc(train_df, check_same_question=check_same_question, count_type='pval', parser=stemmed_parser, single_words=False), name=func_name)
            func_name = 'ans_words_stem_pval'
            self.add_answer_func(train_b, aux_b, func=AnswerCountFunc(train_df, check_same_question=check_same_question, count_type='pval', parser=stemmed_parser3, single_words=True), name=func_name)
            func_name = 'ans_words_pairs_zscore'
            self.add_answer_func(train_b, aux_b, func=AnswerCountFunc(train_df, check_same_question=check_same_question, count_type='zscore', parser=pairs_parser, single_words=True), name=func_name)
            func_name = 'ans_words_stem_pval'
            self.add_answer_func(train_b, aux_b, func=AnswerCountFunc(train_df, check_same_question=check_same_question, count_type='pval', parser=stemmed_parser3, single_words=True), name=func_name)
            func_name = 'ans_words_stem_zscore'
            self.add_answer_func(train_b, aux_b, func=AnswerCountFunc(train_df, check_same_question=check_same_question, count_type='zscore', parser=stemmed_parser2, single_words=True), name=func_name)

            func_name = 'ans_corr_vs_qst_count'
            self.add_answer_func(train_b, aux_b, func=AnswerCountFunc(train_df, check_same_question=check_same_question, count_type='ans_vs_qst', parser=stemmed_parser, single_words=True, norm_scores=False), name=func_name)
                
            func_name = 'ans_length'
            self.add_answer_func(train_b, aux_b, func=AnswersLengthFunc(log_flag=True ), name=func_name)
            func_name = 'ans_length_ratio'
            self.add_answer_func(train_b, aux_b, func=AnswersLengthFunc(log_flag=False), name=func_name)
            func_name = 'ans_num_words'
            self.add_answer_func(train_b, aux_b, func=AnswersNumWordsFunc(), name=func_name)
                
            func_name = 'is_BC'
            self.add_answer_func(train_b, aux_b, func=AnswersIsBCFunc(), name=func_name) 
            func_name = 'BCDA'
            self.add_answer_func(train_b, aux_b, func=AnswersBCDAFunc(), name=func_name) 
        
            func_name = 'is_numerical'
            self.add_answer_func(train_b, aux_b, func=AnswersIsNumericalFunc(), name=func_name)

        print '-> Finished preparing features (types: %s)' % ('all' if ftypes is None else ', '.join(['%s'%ft for ft in ftypes]))
        
    def _cache_filename(self, fname):
        return '%s/%s.pkl' % (self.cache_dir, fname)
    
    def _read_from_cache(self, fname):
        filename = self._cache_filename(fname)
        #print 'Loading from cache %s' % filename
        return load_from_pkl(filename)

    def _save_to_cache(self, fname, data):
        filename = self._cache_filename(fname)
        print 'Saving to cache %s' % filename
        return save_to_pkl(filename, data)

    def _is_in_cache(self, name):
        if self.cache_dir is None:
            return False
        exists = True
        if np.isscalar(name):
            exists = os.path.exists(self._cache_filename(name))
        else:
            for n in name:
                exists = exists and os.path.exists(self._cache_filename(n))
        return exists
        
    def add_answer_func(self, train_b, aux_b, func, name, question_ids=None): 
        '''
        Run a score function on each set of question and answers
        '''
        if (not self.recalc) and (self.cache_dir is not None) and (self._is_in_cache(name)):
            if np.isscalar(name):
                train_b[name] = self._read_from_cache(name)
            else:
                for n in name:
                    train_b[n] = self._read_from_cache(n)
            return
        
        if aux_b is None:
            aux_ids = set()
        else:
            aux_ids = set(aux_b['ID'])
            print 'Given %d aux IDs' % len(aux_ids)
        groups = train_b.groupby('ID').groups
        for i,(idx,inds) in enumerate(groups.iteritems()):
            assert len(set(train_b.irow(inds)['question']))==1
            if (question_ids is not None) and (idx not in question_ids): continue
            question = train_b.irow(inds[0])['question']
            is_dummy = train_b.irow(inds[0])['is_dummy']
            answers = np.array(train_b.irow(inds)['answer'])
            if 'correct' in train_b.columns:
                print '\n-----> #%d : correct = %s' % (i, ', '.join(['%d'%c for c in np.array(train_b.irow(inds)['correct'])]))
                sys.stdout.flush()
    #         print 'applying func to: %s' % str(np.array(train_b.irow(inds)['answer']))
    
            # Check if there's an identical question in aux_b
            if idx in aux_ids:
                same_qst = aux_b[aux_b['ID']==idx]
                assert question == np.unique(same_qst['question'])[0]
                assert set(answers) == set(same_qst['answer'])
                assert np.all(answers == np.array(same_qst['answer']))
                print 'Found same question in aux (ID %s):\n%s' % (idx, str(same_qst))
                vals = []
                for ai,ans in enumerate(answers): # for ans in answers:
                    if np.isscalar(name):
                        #vals.append(float(same_qst[same_qst['answer']==ans][name]))
                        vals.append(float(same_qst.irow(ai)[name]))
                    else:
                        #vals.append(np.array(same_qst[same_qst['answer']==ans][name]).flatten())
						vals.append(np.array(same_qst.irow(ai)[name]).flatten())
                print ' -> vals: %s' % vals
            else:
                # Check if it's a dummy question
                if is_dummy > 1:
                    # No need to waste time on dummy questions...
                    if np.isscalar(name):
                        vals = [-1] * len(inds)
                    else:
                        vals = [np.ones(len(name)) * (-1)] * len(inds) 
                else:
                    # Compute func
                    vals = func(question, answers)
            if question_ids is not None:
                print 'vals = %s' % str(vals)
            for val,ind in zip(vals, inds):
                if np.isscalar(val):
                    train_b.set_value(ind, name, val)
                else:
                    assert len(val)==len(name)
                    for v,n in zip(val,name):
                        train_b.set_value(ind, n, v)
                        
        if (self.cache_dir is not None) and (question_ids is None):
            if np.isscalar(name):
                self._save_to_cache(name, np.array(train_b[name]))
            else:
                for n in name:
                    self._save_to_cache(n, np.array(train_b[n]))


def add_qa_features(train):
    '''
    Add simple features computed from the questions and/or answers
    '''
    parser = SimpleWordParser()
    train['q_which']     = np.array([('which' in qst.lower().split(' ')) for qst in train['question']])
    train['q____']       = np.array([('___' in qst) for qst in train['question']])
    not_words_weights = {'NOT':1, 'EXCEPT':1, 'LEAST':1} #, 'not':0.5, 'except':0.5, 'least':0.5}
#     train['q_not']       = np.array([1*np.any([(w in ['NOT','EXCEPT','LEAST']) for w in qst.split(' ')]) for qst in train['question']])
    train['q_not']       = np.array([np.max([not_words_weights.get(w,0) for w in qst.split(' ')]) for qst in train['question']])
    train['q_num_words'] = np.array([len(parser.parse(qst)) for qst in train['question']])
    train['a_num_words'] = np.array([np.mean([len(parser.parse(ans)) for ans in anss]) for anss in np.array(train[['answerA','answerB','answerC','answerD']])])
    
def mark_dummy_questions(train):
    '''
    Mark dummy questions so that we won't waste time on them
    '''
    bad_matches = ['[^\(]*[\)]', '.*[\(][^\)]*$',
                   ]
    bad_searches = ['(Which|Where|How|What|Why) [^\?]*$',
                    '(The| the| a| an) (the|from|in|and|a|an|is|be|are|am|that|this|these|those|he|she|it|they|you|of|has|have|had|when|what|which|why|how|without|should|could|will|would|by) ',
                    ' that (and|or|be) ', 
                    ' (will|would|should|could|of) (will|would|should|can|could|must|of) ',
                    ' an [bcdfgjklmnpqrstvwxyz]',
                    ]
    
    #parser = SimpleWordParser()
    if False: # the "dummy" marking doesn't save a lot (only ~10% of questions), so let's skip it...
        dummies = []
        for qst in train['question']:
            dummy = 0
            words = re.split('[ ]+', qst) #parser.parse(qst)
            dummy += np.sum([words[i]==words[i+1] for i in range(len(words)-1)])
            dummy += np.sum([(re.match(br, qst) is not None) for br in bad_matches])
            dummy += np.sum([(re.search(br, qst) is not None) for br in bad_searches])
            dummies.append(dummy)
        train['is_dummy'] = np.array(dummies)
    train['is_dummy'] = np.zeros(len(train))
    #print 'Marked %d,%d dummy questions' % (np.sum(train['is_dummy'] > 0), np.sum(train['is_dummy'] > 1))


def convert_values(train, test, features, aux_test=None, target_col=None, method=None, min_num_vals=1, replace_flag=True, build_ids=None):
    if build_ids is None:
        build_data = train
    else:
        build_data = train.take(build_ids)
    global_mean = np.mean(build_data[target_col])
#     print 'Converting - global mean for %d records = %.4f' % (len(build_data), global_mean)
    if method=='linreg':
        #lr = linear_model.LinearRegression(normalize=False)
        lr = linear_model.LogisticRegression(penalty='l1', dual=False)
        lr.fit(build_data[features], build_data[target_col])
        print 'LR coeffs (on %d rows): %s' % (len(build_data), lr.coef_)
        new_feature = '%s_%s' % (','.join(features), str(method))
        for ds in [train, test, aux_test]:
            if ds is None: continue
            ds[new_feature] = lr.predict(ds[features])
        return [new_feature]

    converted_features = []
    for feature in features:
        tf = train[feature]
        test_f = [] if test is None else test[feature]
        aux_test_f = [] if aux_test is None else aux_test[feature]
        if np.any(np.isnan(tf)):
            # Replace nan by the smallest value -1
            allf = unique_with_nan(np.concatenate([tf, test_f, aux_test_f]))
            min_val1 = np.min([v for v in allf if not np.isnan(v)]) - 1
            tf         = [(min_val1 if np.isnan(v) else v) for v in tf]
            test_f     = [(min_val1 if np.isnan(v) else v) for v in test_f]
            aux_test_f = [(min_val1 if np.isnan(v) else v) for v in aux_test_f]
        vals = np.unique(np.concatenate([tf, test_f, aux_test_f]))
        if len(vals) < min_num_vals: continue
        tar_means = [np.mean(np.concatenate([build_data[build_data[feature]==val][target_col], [global_mean]])) for val in vals] # add [global_mean] as tikun and for values that don't occur in build data
        if method is None:
            cats = [ord(c)-ord('A') for c in vals]
        elif method=='label':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(tf) + list(test[feature]))
            cats = lbl.transform(vals)
            print 'feature: %s' % feature
            print '  vals: %s' % vals
            print '  cats: %s' % cats
        elif method=='order':
            cats = range(len(vals)) # new categories
            print 'tar means: %s' % ', '.join(['%.2f (%d)'%(t,np.sum(tf==v)) for v,t in zip(vals,tar_means)])
            vals = np.take(vals, np.argsort(tar_means))
        elif method=='mean':
            cats = tar_means
        elif method=='prob-z':
            M, N = len(build_data), np.sum(build_data[target_col])
            cats = []
            for val in vals:
                n, k = np.sum(np.asarray(build_data[feature])==val), np.sum(build_data[build_data[feature]==val][target_col])
                if n == 0: # no samples in build dataset -> give z=0
                    cats.append(0.0)
                else:
                    prob = hg_test(M, np.sum(np.asarray(build_data[feature])==val), N, np.sum(build_data[build_data[feature]==val][target_col]))
                    zprob = prob['dir'] * scipy.stats.norm.isf(prob['prob'])
                    #print '  val = %s -> M=%d, n=%d, N=%d, k=%d -> prob=%s -> z = %.2f' % (val, M, n, N, k, prob, zprob)
                    cats.append(zprob)
#             print 'vals = %s' % vals[:5]
#             print 'cats = %s' % cats[:5]
#             cats = [scipy.stats.ranksums(build_data[tf==val][target_col],build_data[tf!=val][target_col])[0] for val in vals]
#            cats = [scipy.stats.ranksums(train[tf==val][target_col],train[tf!=val][target_col])[0] for val in vals]
#             cats = [scipy.stats.ranksums(train[tf==val][target_col],train[target_col])[0] for val in vals]
        else:
            raise ValueError('Unknown method: %s' % method)
        val_to_cat = dict(zip(vals,cats))
#         print 'Converting feature %s (%d values): %s' % (feature, len(vals), sorted(val_to_cat, key=val_to_cat.get))
        if replace_flag:
            new_feature = feature
        else:
            new_feature = '%s_%s' % (feature, str(method))
        for ds in [train, test, aux_test]:
            if ds is None: continue
            ds[new_feature] = np.array(map(lambda v: np.nan if np.isnan(v) else val_to_cat[v], ds[feature]))
        converted_features.append(new_feature)
    return converted_features


             
#################################################################################################
# Data preparation/handling
#################################################################################################

MARK_ANSWER_ALL  = ' <ALL>'
MARK_ANSWER_BOTH = ' <BOTH>'
MARK_ANSWER_NONE = ' <NONE>'

def sub_complex_answers(train):
    '''
    Substitute complex answers like "Both A and B" by the contents of answers A and B,
    "All of the above" by the contents of all answers, and "None of the above" by "". 
    We also mark these substitutions for later use.
    '''
    print 'Substituting complex answers'
    all_re  = re.compile('\s*all of the above\s*')
    both_re = re.compile('\s*both ([a-d]) and ([a-d])[\.]?\s*')
    none_re = re.compile('\s*none of the above\s*')
    for ind,answers in zip(train.index, np.array(train[['answerA','answerB','answerC','answerD']])):
        for ansi,anst in zip(['A','B','C','D'], answers):
            new_ans = None
            all_m = re.match(all_re, anst.lower())
            if all_m is not None:
#                 assert ansi in ['D'], 'Strange... answer%s = %s' % (ansi,anst) # not true in validation set...
                new_ans = '%s and %s and %s%s' % (answers[0], answers[1], answers[2], MARK_ANSWER_ALL)
            else:
                both_m = re.match(both_re, anst.lower())
                if both_m is not None:
                    #assert ansi in ['C','D'], 'Strange... answer%s = %s' % (ansi,anst)
                    both1, both2 = both_m.groups()[0].upper(), both_m.groups()[1].upper()
                    #assert both1!=both2 and both1!=ansi and both2!=ansi
                    new_ans = '%s and %s%s' % (answers[ord(both1)-ord('A')], answers[ord(both2)-ord('A')], MARK_ANSWER_BOTH)
                else:
                    if re.match(none_re, anst.lower()) is not None:
    #                     assert ansi in ['C','D'], 'Strange... answer%s = %s' % (ansi,anst)
                        new_ans = '%s' % MARK_ANSWER_NONE
            if new_ans is not None:
#                 print ' replacing "%s" in #%d by: "%s"' % (anst, ind, new_ans)
                train.set_value(ind, 'answer%s'%ansi, new_ans)
    
def prp_binary_dataf(train):
    stemmer = PorterStemmer()
    parser = SimpleWordParser(word_func=stemmer.stem, min_word_length=1, tolower=True, ascii_conversion=True, ignore_special_words=False)
    indices, questions, answers, correct, ans_names, more_cols_vals = [], [], [], [], [], []
    is_all, is_both, is_none, keywords = [], [], [], []
    if 'correctAnswer' in train.columns:
        correct_answer = np.array(train['correctAnswer'])
    else:
        correct_answer = np.zeros(len(train))
    more_cols = [col for col in train.columns if col not in ['question', 'answerA', 'answerB', 'answerC', 'answerD', 'correctAnswer']]
    for idx,(qst,ansA,ansB,ansC,ansD),cor,mcols in zip(train.index, np.array(train[['question', 'answerA', 'answerB', 'answerC', 'answerD']]), 
                                                       correct_answer, np.array(train[more_cols])):
        for ia,(ic,ans) in enumerate(zip(['A','B','C','D'],[ansA, ansB, ansC, ansD])):
            indices.append(idx)
            questions.append(qst)
            a_ans, a_all, a_both, a_none, a_keywords = ans, 0, 0, 0, 0
            if ans.endswith(MARK_ANSWER_ALL):
                a_ans = ans[:-len(MARK_ANSWER_ALL)]
                a_all = 1
            elif ans.endswith(MARK_ANSWER_BOTH):
                a_ans = ans[:-len(MARK_ANSWER_BOTH)]
                a_both = 1
            elif ans.endswith(MARK_ANSWER_NONE):
                a_ans = ans[:-len(MARK_ANSWER_NONE)]
                a_none = 1
            else:
                words = parser.parse(ans)
                if 'both' in words:
                    a_both = 0.5
                if stemmer.stem('investigation') in words:
                    a_keywords = 1
            answers.append(a_ans)
            is_all.append(a_all)
            is_both.append(a_both)
            is_none.append(a_none)
            keywords.append(a_keywords)
            if cor==0:
                correct.append(0) # no 'correctAnswer' column -> set correct=0 for all answers
            else:
                correct.append(1 if ia==(ord(cor)-ord('A')) else 0)
            ans_names.append(ic)
            more_cols_vals.append(mcols)
    pdict = {'ID': indices, 'question': questions, 'answer': answers, 'correct': correct, 'ans_name': ans_names, 
             'is_all': is_all, 'is_both': is_both, 'is_none': is_none} #, 'ans_keywords': keywords}
    for icol,mcol in enumerate(more_cols):
        pdict[mcol] = np.array([vals[icol] for vals in more_cols_vals])
    return pd.DataFrame(pdict)

def get_predictions_from_binary_dataf(dataf, column, direction='max', scores=None):
    assert direction in ['max','min']
    assert (column is None) != (scores is None) # exactly one of them must be provided
    # Fast code, assuming dataf is ordered such that all 4 answers are consecutive and always appear in order of A,B,C,D
    if scores is None:
        scores = np.array(dataf[column])
    ind = 3 if direction=='max' else 0
    selected = scores.reshape((len(scores)/4,4)).argsort(axis=1)[:,ind]
    ids = dataf['ID'][::4]
    assert len(set(ids)) == len(scores)/4
    names = ['A','B','C','D']
    return dict([(id,names[sel]) for id,sel in zip(ids,selected)])
                  
    # Original, slow code:      
#     q_groupby = dataf.groupby('ID')
#     preds = {} #[''] * q_groupby.ngroups
#     factor = 1 if direction=='max' else -1
#     for idx,inds in q_groupby.groups.iteritems():
#         if scores is None:
#             vals = np.array(dataf.irow(inds)[column]) * factor
#         else:
#             vals = np.take(scores, inds) * factor
#         anss = np.array(dataf.irow(inds)['ans_name'])
#         preds[idx] = anss[np.argmax(vals)]
#     
#     return preds

def add_scores_from_binary_dataf(dataf, dataf_b, column):
    columns4 = ['%s%s'%(column,ans) for ans in ['A','B','C','D']]
    vals = []
    for i in range(len(dataf)):
        row, idx = dataf.irow(i), dataf.index[i]
        b_rows = dataf_b[dataf_b['ID']==idx]
        assert len(b_rows)==4 and np.all(b_rows['ans_name']==['A','B','C','D'])
        vals.append(np.array(b_rows[column]))
    vals = np.array(vals)
    for j,col in enumerate(columns4):
        dataf[col] = vals[:,j]
    return columns4

                 
#################################################################################################
# Functions for computing a value (feature) for each answer
#################################################################################################
                
class AnswersFunc(object):
    def __call__(self, question, answers):
        pass

#returns answer lengths normalized by the mean of the lengths of answers
#for answers = ["go through a life cycle", "form a food web", "act as a source of food", "affect other parts of the ecosystem"]    
#if log then returns array([ 0.04082202,  0.44628713,  0.04082202,  0.36464316], dtype=float32)
#else array([ 0.95999998,  0.63999999,  0.95999998,  1.44000006], dtype=float32)
class AnswersLengthFunc(AnswersFunc):
    def __init__(self, log_flag=True):
        self.log_flag = log_flag
    def __call__(self, question, answers):
        lens = np.array(map(len, answers), dtype=np.float32)
        assert np.sum(lens)>0
        if self.log_flag:
            return np.abs(np.log((1.0+lens) / (1.0+np.mean(lens))))
        else:
            print '%s' % ((1.0+lens) / (1.0+np.mean(lens)))
            return (1.0+lens) / (1.0+np.mean(lens))

"""
returns array([4, 3, 3, 3]) for 
answers=['go through a life cycle', 'form a food web', 'act as a source of food', 'affect other parts of the ecosystem']
with stop words ignored
"""
class AnswersNumWordsFunc(AnswersFunc):
    def __init__(self, parser=SimpleWordParser()):
        self.parser = parser
    def __call__(self, question, answers):
        num_words = []
        for ans in answers:
            num_words.append(len(self.parser.parse(ans)))
        return np.array(num_words)

class AnswersIsBCFunc(AnswersFunc):
    def __init__(self):
        pass
    def __call__(self, question, answers):
        assert len(answers)==4
        return np.array([0,1,1,0])

class AnswersBCDAFunc(AnswersFunc):
    # Counts in training set: A: 584, B: 672, C: 640, D: 604 
    def __init__(self):
        pass
    def __call__(self, question, answers):
        assert len(answers)==4
        return np.array([0,3,2,1])

"""
for answers = ['0', '100', 'twenty', '1,678'], returns array([0, 2, 2, 0])
order array will be [0,2,1,3] and 2nd and third elements have the middle value hence [0,2,2,0]
for answers=['go through a life cycle', 'form a food web', 'act as a source of food', 'affect other parts of the ecosystem'], 
returns array([ 1.,  1.,  1.,  1.])
"""
class AnswersIsNumericalFunc(AnswersFunc):
    NUMBER_STR = {'zero':0, 'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8, 'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'twenty':20}
    def __init__(self):
        pass
    def __call__(self, question, answers):
        nums = [self._is_numerical(ans) for ans in answers]
        if np.all([(num is not None) for num in nums]):
#             print 'answers: %s' % answers
#             print 'nums: %s' % nums
            if len(set(nums))==len(nums):
                order = np.argsort(nums)
#                 print ' -> %s' % np.array([(2 if (x>0 and x<len(answers)-1) else 0) for x in order])
                return np.array([(2 if (x>0 and x<len(answers)-1) else 0) for x in order])
            else: # not unique numbers
                return np.ones(len(answers))
        else:
            return np.ones(len(answers))
        
    def _is_numerical(self, answer):
        if len(answer)==0:
            return None
        # Find first word that is a number
        words = answer.split(' ')
        for word in words:
            word = word.replace(',','') # turn "2,000" to "2000"
            if re.match('^[0-9]+([\.][0-9]+)?$', word) is not None:
                return float(word)
            else:
                if AnswersIsNumericalFunc.NUMBER_STR.has_key(word):
                    return float(AnswersIsNumericalFunc.NUMBER_STR[word])
        return None # did not find a word that looks like a number
# old code:
#         first_word = answer.split(' ')[0].replace(',','')
#         if re.match('^[0-9]+([\.][0-9]+)?$', first_word) is not None:
#             return float(first_word)
#         else:
#             if AnswersIsNumericalFunc.NUMBER_STR.has_key(first_word):
#                 return float(AnswersIsNumericalFunc.NUMBER_STR[first_word])
        
"""
 fraction of the answer in question for each answer
 question: 'Frogs lay eggs that develop into tadpoles and then into adult frogs. This sequence of changes is an example of how living things'
 answers=['go through a life cycle', 'form a food web', 'act as a source of food', 'affect other parts of the ecosystem']
 return value: [ 0.,  0.,  0.,  0.]
"""
class AnswersInQuestionFunc(AnswersFunc):
    def __init__(self, parser=SimpleWordParser()):
        self.parser = parser
    def __call__(self, question, answers):
        q_words = set(self.parser.parse(question))
        in_fracs = []
        for ans in answers:
            a_words = set(self.parser.parse(ans))
            if len(a_words) == 0:
                in_fracs.append(0.0)
            else:
                in_fracs.append(len(q_words.intersection(a_words)) / (len(a_words) + 0.0))
        return np.array(in_fracs)

"""
for answers = 'go through a life cycle', 'form a food web', 'act as a source of food', 'affect other parts of the ecosystem'
returns array([ 0. ,  0.5,  0.5,  0. ])
only the second and third answers overlap with the other answers and equally overlap.
"""
class AnswersInAnswersFunc(AnswersFunc):
    def __init__(self, parser=SimpleWordParser()):
        self.parser = parser
    def __call__(self, question, answers):
        a_words = [set(self.parser.parse(ans)) for ans in answers]
        in_fracs = []
        for i,ans in enumerate(answers):
            w1 = a_words[i]
#             print 'ans %d: %s' % (i, w1)
            if len(w1) == 0:
                in_fracs.append(0.0)
            else:
                frac = 0.0
                for j in range(len(answers)):
                    if j==i: continue
#                     print '   with ans %d: %s -> %.3f' % (j, a_words[j], (len(w1.intersection(a_words[j])) / (len(w1) + 0.0)))
                    frac += (len(w1.intersection(a_words[j])) / (len(w1) + 0.0))
                in_fracs.append(frac / (len(answers) - 1.0))
#                 print '-> %.3f' % in_fracs[-1]
        in_fracs = np.array(in_fracs)
        if np.all(in_fracs == 0):
            return np.ones(len(answers)) * 0.25
        else:
            return in_fracs / np.sum(in_fracs)

"""
the returned answer slightly varies on if norm_scores is enabled
if ans_type is count, returns the sume of mean of counts of words in the answer (mentioned in answer before) and mean of counts (mentioned in question before if enabled); mean over the words in the answer
if ans_type is correct, returns the geometric mean of ratio of count of words in correct answers to any answer and question (if enabled)
if ans_type is ans_vs_qst, returns the geometric mean of ratios of number of times each word appeared in the corrcet answer to number times it appeared in a question
if ans_type is pval, then a hypergeometric probability is computed based on total number of answers in training data N*4, total number of corrcet answers in training data N (because np.sum(train['correct']) is over the column 'correct' which has values either 1 or 0), number of times the same word is mentioned in a correct answer, total number of mentions of the word. An intutive score is assigned based on if its an under-estimation or over-estimation, refer the relevant code.
if ans_type is zscore, geometric mean of zscore of hg prob. of each word is returned 
"""
class AnswerCountFunc(AnswersFunc):
    def __init__(self, train, check_same_question=True, count_type='count', parser=None, single_words=False, use_questions=False, norm_scores=True):
        self.train = train
        self.check_same_question = check_same_question
        self.count_type = count_type
        self.count_type in ['count','correct','pval','zscore','ans_vs_qst'] 
        self.parser = parser
        self.single_words = single_words
        if single_words:
            assert self.parser is not None
        self.use_questions = use_questions
        self.norm_scores = norm_scores
        self.ans_trans_f = self._get_answer_trans_func()
        self.ans_count, self.ans_correct = None, None
        self.qst_count, self.qst_correct = None, None
        self._parsed_words = None
    
    def _get_answer_trans_func(self):
        if self.parser is None:
            return lambda ans: [ans]
        else:
            if self.single_words:
                return lambda ans: self.parser.parse(ans)
            else:
                return lambda ans: [' '.join(self.parser.parse(ans))]

    def _parse_train(self, train):
        self._parsed_words = {}
        for txt in np.concatenate([np.unique(train['answer']), np.unique(train['question'])]):
            self._parsed_words[txt] = self.ans_trans_f(txt)
            
    def _calc_stats(self, train):
        self.ans_count, self.ans_correct = {}, {}
        self.qst_count, self.qst_correct = {}, {}
        for ans,corr in zip(train['answer'],train['correct']):
            words = self._parsed_words[ans] #self.ans_trans_f(ans)
            for word in words:
                if not self.ans_count.has_key(word):
                    self.ans_count[word] = 1
                    self.ans_correct[word] = corr
                else:
                    self.ans_count[word] += 1
                    self.ans_correct[word] += corr
        if self.use_questions or (self.count_type == 'ans_vs_qst'):
            for qst in train['question']:
                words = self._parsed_words[qst] #self.ans_trans_f(qst)
                for word in words:
                    if not self.qst_count.has_key(word):
                        self.qst_count[word] = 1
                        self.qst_correct[word] = 0.25
                    else:
                        self.qst_count[word] += 1
                        self.qst_correct[word] += 0.25
        
    def __call__(self, question, answers):
        if self._parsed_words is None:
            self._parse_train(self.train) # parse only upon the 1st call

        if self.check_same_question:
            train = self.train[self.train['question']!=question]
        else:
            train = self.train
        if self.check_same_question or (self.ans_count is None):
            self._calc_stats(train)
            
        assert len(self.ans_count) > 0 and len(self.ans_correct) > 0
        if self.use_questions:
            assert len(self.qst_count) > 0 and len(self.qst_correct) > 0

        t_answers = [self.ans_trans_f(ans) for ans in answers]
        print 'answers: %s' % answers
        print ' t_answers: %s' % t_answers
        if self.count_type == 'count':
#             for t_ans in t_answers:
#                 print '  t_ans=%s -> counts: %s' % (t_ans, [self.ans_count.get(ta,0) for ta in t_ans])
            counts = []
            for t_ans in t_answers:
                if len(t_ans) == 0:
                    counts.append(0)
                else:
                    if self.use_questions:
                        counts.append(np.mean([self.ans_count.get(ta,0) for ta in t_ans] + [self.qst_count.get(ta,0) for ta in t_ans]))
#                         print '    ans counts: %s' % ', '.join(['%d'%self.ans_count.get(ta,0) for ta in t_ans])
#                         print '    qst counts: %s' % ', '.join(['%d'%self.qst_count.get(ta,0) for ta in t_ans])
#                         print '    -> mean = %.2f' % counts[-1]
                    else:
                        counts.append(np.mean([self.ans_count.get(ta,0) for ta in t_ans]))
        elif self.count_type == 'correct':
            counts = []
            for t_ans in t_answers:
                corrs = []
                for ta in t_ans:
                    corr, count = 0.25, 1.0
                    if self.ans_count.has_key(ta):
                        count += self.ans_count[ta]
                        corr  += self.ans_correct[ta]
                    if self.use_questions and self.qst_count.has_key(ta):
                        count += self.qst_count[ta]
                        corr  += self.qst_correct[ta]
                    corrs.append(-np.log(corr / count))
#                     print '  ans count , corr = %d , %d' % (self.ans_count.get(ta,0), self.ans_correct.get(ta,0))
#                     print '  qst count , corr = %d , %d' % (self.qst_count.get(ta,0), self.qst_correct.get(ta,0))
#                     print ' -> corr = %.3f' % corrs[-1]
#                     if self.ans_count.has_key(ta):
#                         corrs.append(-np.log((self.ans_correct[ta]+0.25)/(self.ans_count[ta]+1.0)))
#                     else:
#                         corrs.append(-np.log(0.25))
                if len(corrs) == 0:
                    counts.append(0.25)
                else:
                    counts.append(np.exp(-np.mean(corrs)))
            assert not np.any([np.isnan(x) for x in counts])
#             counts = [np.exp(-np.mean([-np.log((ans_correct[ta]+0.25)/(self.ans_count[ta]+1.0) if self.ans_count.has_key(ta) else 0.25) for ta in t_ans])) for t_ans in t_answers]
#             for t_ans,cnt in zip(t_answers,counts):
#                 print '  t_ans=%s -> correct: %s , count: %s -> %.4f' % (t_ans, [self.ans_correct.get(ta,'?') for ta in t_ans], [self.ans_count.get(ta,'?') for ta in t_ans], cnt)
        elif self.count_type == 'ans_vs_qst':
            # Count the number of times the answer (or its words) appears as a correct answer vs. the number of times it appears in questions
            assert not self.use_questions, 'use_questions not supported for pval or zscore'
            counts = []
            for t_ans in t_answers:
                ratios = []
                for ta in t_ans:
                    ans_corr, qst_count = 0.25, 1.0 # Laplace
                    if self.ans_count.has_key(ta):
                        ans_corr  += self.ans_correct[ta]
                    if self.qst_count.has_key(ta):
                        qst_count += self.qst_count[ta]
                    ratios.append(-np.log(ans_corr / qst_count))
                    print '  ans count , corr = %d , %d' % (self.ans_count.get(ta,0), self.ans_correct.get(ta,0))
                    print '  qst count , corr = %d , %d' % (self.qst_count.get(ta,0), self.qst_correct.get(ta,0))
                    print ' -> ratio = %.3f' % ratios[-1]
                if len(ratios) == 0:
                    counts.append(0.0)
                else:
                    counts.append(np.exp(-np.mean(ratios)))
            assert not np.any([np.isnan(x) for x in counts])
        else: # pval or zscore
            assert not self.use_questions, 'use_questions not supported for pval or zscore'
            counts = []
            for t_ans in t_answers:
#                 print 't_ans: %s' % t_ans
                t_probs = []
                for ta in t_ans:
                    if self.ans_count.has_key(ta):
                        mult_test_corr = True if (self.count_type == 'pval') else False 
                        hg = hg_test(len(train), np.sum(train['correct']), self.ans_count[ta], self.ans_correct[ta], dir=None, mult_test_correct=mult_test_corr)
                        prob, dir = hg['prob'], hg['dir']
                        if self.count_type == 'pval':
                            # Set score according to dir and prob:
                            # dir=-1:  0.25 for prob=1, 0 for prob=0
                            # dir= 1:  0.25 for prob=1, 1 for prob=0
                            if dir == -1: 
                                pr = 0.25*(np.exp(prob)-np.exp(0))/(np.exp(1)-np.exp(0))
                            else:
                                pr = 1.0 + 0.75*(np.exp(prob)-np.exp(0))/(np.exp(0)-np.exp(1))
                        else: # 'zscore'
                            ##assert prob <= 0.55, 'Prob is too large? (%.5f)' % prob
                            prob = np.clip(prob, 1E-20, 0.5)
                            pr = (-dir) * scipy.stats.norm.isf(prob) # Z-score
                           
#                         if prob<110.1:
#                             print '  ta: %s' % ta
#                             print '  count %2d , correct %2d -> prob %.5f dir %2s -> %.5f' % (self.ans_count[ta], self.ans_correct[ta], prob, dir, pr)                    
                    else:
                        if self.count_type == 'pval':
                            pr = 0.25
                        else: # 'zscore'
                            pr = 0.0
                    if self.count_type == 'pval':
                        t_probs.append(-np.log(pr))
                    else: # 'zscore'
                        t_probs.append(pr)
                if len(t_probs) == 0:
                    counts.append(0.25)
                else:
                    if self.count_type == 'pval':
                        counts.append(np.exp(-np.mean(t_probs)))
                    else: # 'zscore'
                        counts.append(scipy.stats.norm.sf(np.sum(t_probs)/np.sqrt(len(t_probs))) ** 2) # take square so that Z=0 will become prob=0.25 (and not 0.5)
#                         print ' -> t_probs = %s --> adding %.5f' % (t_probs, counts[-1]) 
#         for t_ans in t_answers:
#             print '* (%3d) %s' % (self.ans_count.get(t_ans,0), t_ans)
#         print '-> counts: %s' % ', '.join(['%.3f'%c for c in counts])
        counts = np.array(counts)
        if self.norm_scores:
            if np.all(counts == 0):
                return np.ones(len(answers)) * 0.25
            return counts / (np.sum(counts) + 0.0)
        else:
            return counts


#unused
class AnswerPairCountFunc(AnswersFunc):
    def __init__(self, train, check_same_question=True):
        self.train = train
        self.check_same_question = check_same_question
    def __call__(self, question, answers):
        if self.check_same_question:
            train = self.train[self.train['question']!=question]
        else:
            train = self.train
        ans_count = {}
        for idx,inds in train.groupby('ID').groups.iteritems():
            print 'idx %s' % idx
            assert len(inds)==4
            for i in range(3):
                row_i = train.irow(inds[i])
                for j in range(i+1,4):
                    row_j = train.irow(inds[j])
                    pair = tuple(sorted([row_i['answer'], row_j['answer']]))
                    print ' pair: %s' % str(pair)
#         for ans in train['answer']:
#             if not ans_count.has_key(ans):
#                 ans_count[ans] = 1
#             else:
#                 ans_count[ans] += 1
#         counts = [ans_count.get(ans,0) for ans in answers]
#         for ans in answers:
#             print '* (%3d) %s' % (ans_count.get(ans,0), ans)
#         print '-> counts: %s' % ', '.join(['%.3f'%c for c in counts])
#         counts = np.array(counts)
#         if np.all(counts == 0):
#             return np.ones(len(answers)) * 0.25
#         return counts / (np.sum(counts) + 0.0)

#unused
class AnswersSameQuestionFunc(AnswersFunc):
    def __init__(self, train, use_stemmer=False):
        self.train = train
        if use_stemmer:
            self.stemmer = PorterStemmer()
            self.parser = SimpleWordParser(word_func=self.stemmer.stem)
        else:
            self.parser = SimpleWordParser(word_func=None)
    def __call__(self, question, answers):
        question = question.lower()
#        q_words = self.parser.parse(question)
        same_q = self.train[self.train['question']==question]
#         if 'naturally occurring' in question:
#             print 'question = %s' % question
#             print 'same_q: %s' % same_q
        if len(same_q) == 0:
            return np.array([0.25,0.25,0.25,0.25])
        
        print '==> Found same question: %s' % question
#         print '%s' % same_q
        print '-------------------------------------------------'
        scores = []
        for ia,answer in enumerate(answers):
            answer = answer.lower()
            same_qa = same_q[same_q['answer']==answer]
            if len(same_qa) > 0:
                assert len(same_qa)==1
#                 print 'same_qa: %s' % same_qa.irow(0)['correct']
                scores.append(same_qa.irow(0)['correct']) # 0 or 1
            else:
                print '-> Answer (%s) not found...' % (answer)
                scores.append(0.01)
#             a_words = self.parser.parse(answer)
#             print '  -> %s' % ' ; '.join(a_words)
#             probs = locdic.double_search(words1=q_words, words2=a_words, num_words1=[None,1,2,3,4,5,6,7], num_words2=[None,1,2,3,4,5]) # (3,5)(2) -> 0.3012,0.2444; (2,3,4,5)(1,2) -> 0.3216,0.2756; (1-7),(1-5) -> 0.3232,0.2816
#             print '  answer #%d: %.2e , %.2e' % (ia, probs[0], probs[1])
        print 'scores: %s' % scores
#         ref_score0, ref_score1 = np.max([s[0] for s in scores]), np.max([s[1] for s in scores])
#         scores = [(s[0]/ref_score0, s[1]/ref_score1) for s in scores]
#         print '-> scores: %s ; %s' % (', '.join(['%.2e'%s[0] for s in scores]), ', '.join(['%.2e'%s[1] for s in scores]))
        return np.array(scores)

"""
Collects and returns zscores fore each word. 
The zscore for each word is (p-mu)/sig where mu=num_q_docs*(num_of_docs_aword/total_num_of_docs), sig=n*p*q, p=(number of intersecting docs between qwords and aword) 
"""
class AnswersWordZscoreFunc(AnswersFunc):
    def __init__(self, locdic=None, parser=SimpleWordParser(), score='zscore', norm_scores=True):
        self.locdic = locdic
        self.parser = parser
        self.score = score
        assert self.score == 'zscore'
        self.norm_scores = norm_scores
        
    def __call__(self, question, answers):
        if not isinstance(self.locdic, LocationDictionary):
            self.locdic = self.locdic() # a generator
            assert isinstance(self.locdic, LocationDictionary)
        num_total_docs = self.locdic.get_num_docs() + 0.0
        STD_FACTOR = 300.0
        EPSILON = 1E-320
        print 'question = %s' % question
        q_words = np.unique(self.parser.parse(question, calc_weights=False))
        print '  -> words: %s' % ' ; '.join(q_words)
        q_word_ids = [self.locdic.word_ids[word] for word in q_words if self.locdic.word_ids.has_key(word)]
        print '  -> ids: %s' % str(q_word_ids)
        q_docs = []
        for wid in q_word_ids:
            q_docs.append(set(self.locdic.word_locations[wid].keys()))
        print '  -> # docs: %s' % ' ; '.join(['%d'%len(d) for d in q_docs])
        
        scores = []
        for ia,answer in enumerate(answers):            
            a_words = np.unique(self.parser.parse(answer, calc_weights=False))
            print '  -> words: %s' % ' ; '.join(a_words)
            a_word_ids = [self.locdic.word_ids[word] for word in a_words if self.locdic.word_ids.has_key(word)]
            print '  -> ids: %s' % str(a_word_ids)
            zscores = []
            for a_word in a_word_ids:
                a_docs = self.locdic.word_locations[a_word].keys()
                a_pr = len(a_docs) / num_total_docs
                a_pr = np.clip(a_pr, 1E-5, 1.0-(1E-5))
#                 print '  a_word (%.5f) = %s' % (a_pr, a_word)
                for q_word,qd in zip(q_word_ids,q_docs):
                    if len(qd)==0: continue # no documents containing the question word
                    if q_word==a_word: continue # same word - skip
                    num_intersect = len(qd.intersection(a_docs))
#                     print '   qdocs %5d , adocs %5d -> intersection %4d' % (len(qd), len(a_docs), num_intersect)
                    n_exp, n_std = len(qd) * a_pr, np.sqrt(len(qd) * a_pr * (1.0-a_pr))
                    z = (num_intersect - n_exp) / n_std
                    zscores.append(-z) 
#                     print '     -> z = %.3f' % z
                
            print '  answer #%d: zscores = %s => sum = %.2f , num = %d' % (ia, zscores, np.sum(zscores), len(zscores))
            if len(zscores)==0:
                scores.append(0.0)
            else:
                scores.append(scipy.stats.norm.sf(np.sum(zscores)/(STD_FACTOR*np.sqrt(len(zscores)))) ** 2) # take square so that Z=0 will become prob=0.25 (and not 0.5)
#             scores.append(-np.sum(zscores)/np.sqrt(len(zscores))) # take square so that Z=0 will become prob=0.25 (and not 0.5)
            print '---> score = %.2e' % scores[-1]
        if self.norm_scores:
            ref_score = np.sum(scores) + len(scores)*EPSILON
            scores = [(s+EPSILON)/ref_score for s in scores]
        print ' -> scores: %s' % (', '.join(['%.2e'%s for s in scores]))
        assert not np.any([np.isnan(s) for s in scores])
#         dkfjdkfj()
        return np.array(scores)

"""
Just returns an array populated with return values of double_search with parameters as question words and each of the answer words
"""
class AnswersDoubleSearchFunc(AnswersFunc):
    def __init__(self, locdic=None, parser=SimpleWordParser(), num_words_qst=[None,1,2,3,4,5], num_words_ans=[None,1,2,3],
                 score='hg', score_params=None, prob_type='tf-idf', tf_log_flag=True, norm_scores=True):
        self.locdic = locdic
#         if use_stemmer:
#             self.stemmer = PorterStemmer()
#             self.parser = SimpleWordParser(word_func=self.stemmer.stem)
#         else:
#             self.parser = SimpleWordParser(word_func=None)
        self.parser = parser
        self.num_words_qst = num_words_qst
        self.num_words_ans = num_words_ans
        self.score = score
        self.score_params = score_params
        self.prob_type= prob_type
        self.tf_log_flag = tf_log_flag
        self.norm_scores = norm_scores
        self.total_time = 0
        
    def __call__(self, question, answers):
        if not isinstance(self.locdic, LocationDictionary):
            self.locdic = self.locdic() # a generator
            assert isinstance(self.locdic, LocationDictionary)
        EPSILON = 1E-320
        print 'question = %s' % question
        q_words, q_weights = self.parser.parse(question, calc_weights=True)
        if (q_weights is None) or (len(q_weights) == 0):
            print '  -> %s' % ' ; '.join(q_words)
        else:
            print '  -> %s' % ' ; '.join(['%s (%.1f)'%(w,q_weights.get(w,-1)) for w in q_words])
#         print '-> %s' % ' ; '.join(q_words)
#         if len(q_weights) > 0:
#             print '     weights: %s' % q_weights
        scores = []
        for ia,answer in enumerate(answers):            
            a_words, a_weights = self.parser.parse(answer, calc_weights=True)
            if (a_weights is None) or (len(a_weights) == 0):
                print '  -> %s' % ' ; '.join(a_words)
            else:
                print '  -> %s' % ' ; '.join(['%s (%.1f)'%(w,a_weights.get(w,-1)) for w in a_words])
#             if len(a_weights) > 0:
#                 print '     weights: %s' % a_weights
            t1 = time.time()
            probs = self.locdic.double_search(words1=q_words, words2=a_words, words1_weights=q_weights, words2_weights=a_weights, 
                                              num_words1=self.num_words_qst, num_words2=self.num_words_ans,
                                              score=self.score, score_params=self.score_params, prob_type=self.prob_type, tf_log_flag=self.tf_log_flag)
            self.total_time += (time.time() - t1)
            print '  answer #%d: %.2e , %.2e' % (ia, probs[0], probs[1])
            scores.append(probs)
#         ref_score0, ref_score1 = np.max([s[0] for s in scores]), np.max([s[1] for s in scores])
        if self.norm_scores:
            ref_score0, ref_score1 = np.sum([s[0] for s in scores]) + len(scores)*EPSILON, np.sum([s[1] for s in scores]) + len(scores)*EPSILON
            scores = [((s[0]+EPSILON)/ref_score0, (s[1]+EPSILON)/ref_score1) for s in scores]
        print ' -> scores: %s ; %s' % (', '.join(['%.2e'%s[0] for s in scores]), ', '.join(['%.2e'%s[1] for s in scores]))
        print '    total time so far: %.2f' % self.total_time
        return np.array(scores)

"""
Similar to AnswersDoubleSearchFunc with the difference that this call routine makes sure the locdic is built and is ready to use.
"""
class AnswersTrainDoubleSearchFunc(AnswersDoubleSearchFunc):
    def __init__(self, train, base_locdic=None, check_same_question=True, use_questions=True, use_answers=True, min_words_per_qa=1, 
                 parser=SimpleWordParser(),
                 num_words_qst=[None,1,2,3,4,5], num_words_ans=[None,1,2,3],
                 score='hg', score_params=None, prob_type='tf-idf', tf_log_flag=True, norm_scores=True,
                 min_word_docs_frac=0, max_word_docs_frac=0.2, min_word_count_frac=0, max_word_count_frac=0.01):
        AnswersDoubleSearchFunc.__init__(self, locdic=None, parser=parser, num_words_qst=num_words_qst, num_words_ans=num_words_ans,
                                         score=score, score_params=score_params, prob_type=prob_type, tf_log_flag=tf_log_flag, norm_scores=norm_scores)
        self.train = train
        self.base_locdic = base_locdic
        self.use_questions = use_questions
        self.use_answers   = use_answers
        self.min_words_per_qa = min_words_per_qa
        self.min_word_docs_frac = min_word_docs_frac
        self.max_word_docs_frac = max_word_docs_frac
        self.min_word_count_frac = min_word_count_frac
        self.max_word_count_frac = max_word_count_frac
        self.check_same_question = check_same_question
        self.locdic = None
        
    def __call__(self, question, answers):
        if self.locdic is None:
            if self.check_same_question:
                ld_train = self.train[self.train['question']!=question]
            else:
                ld_train = self.train
            if (self.base_locdic is not None) and (not isinstance(self.base_locdic, LocationDictionary)):
                self.base_locdic = self.base_locdic() # generator
            self.locdic = build_training_location_dictionary(ld_train, parser=self.parser, 
                                                             use_questions=self.use_questions, use_answers=self.use_answers, min_words_per_qa=self.min_words_per_qa, 
                                                             base_locdic=self.base_locdic,
                                                             min_word_docs_frac=self.min_word_docs_frac, max_word_docs_frac=self.max_word_docs_frac, 
                                                             min_word_count_frac=self.min_word_count_frac, max_word_count_frac=self.max_word_count_frac)
#             print 'base locdic %d docs -> +train locdic %d docs' % (len(self.base_locdic.doc_ids), len(self.locdic.doc_ids))
            
        scores = AnswersDoubleSearchFunc.__call__(self, question, answers)
        if self.check_same_question:
            self.locdic = None # delete locdic so that we'll create a new one for the next question
            gc.collect()
        return scores
    
"""
Returns an array of length=length(answers) that reflects the best similarity score for answer at index i with other answers.
This uses the first component of double_search routine which captures similarity between words1 and words2.
Finally comb weights are typically used to isolate the best and least similar answer to an option.
"""
class AnswersPairsDoubleSearchFunc(AnswersFunc):
    '''
    Check pairs of answers
    '''
    def __init__(self, locdic=None, parser=SimpleWordParser(), num_words_qst=[None,1,2,3,4,5], num_words_ans=[None,1,2,3],
                 score='hg', score_params=None, prob_type='tf-idf', tf_log_flag=True, norm_scores=True,
                 sim_scores_comb_weights=([1,0,0], [0,0,1]), search_type='q_vs_a1a2'):
        assert search_type in ['q_vs_a1a2', 'a1_vs_a2']
        self.locdic = locdic
        self.parser = parser
        self.num_words_qst = num_words_qst
        self.num_words_ans = num_words_ans
        self.score = score
        self.score_params = score_params
        self.prob_type= prob_type
        self.tf_log_flag = tf_log_flag
        self.norm_scores = norm_scores
        self.sim_scores_comb_weights = sim_scores_comb_weights
        self.search_type = search_type
        
    def __call__(self, question, answers):
        if not isinstance(self.locdic, LocationDictionary):
            self.locdic = self.locdic() # a generator
            assert isinstance(self.locdic, LocationDictionary)        
        EPSILON = 1E-320
        print 'question = %s' % question
        q_words, q_weights = self.parser.parse(question, calc_weights=True)
        print '-> %s' % ' ; '.join(q_words)
        sim_scores = {}
        for ia1 in range(len(answers)-1):
            answer1 = answers[ia1]
            a_words1, a_weights1 = self.parser.parse(answer1, calc_weights=True)
            print '  -> (%d) %s' % (ia1, ' ; '.join(a_words1))
            for ia2 in range(ia1+1,len(answers)):
                answer2 = answers[ia2]
                a_words2, a_weights2 = self.parser.parse(answer2, calc_weights=True)
                print '  -> (%d) %s' % (ia2, ' ; '.join(a_words2))
                if self.search_type == 'q_vs_a1a2':
                    a_weights12 = a_weights1.copy()
                    for w2,wgt2 in a_weights2.iteritems():
                        a_weights12[w2] = np.max([wgt2, a_weights1.get(w2, -np.inf)])
                    probs = self.locdic.double_search(words1=q_words, words2=a_words1+a_words2, words1_weights=q_weights, words2_weights=a_weights12,
                                                      num_words1=self.num_words_qst, num_words2=self.num_words_ans,
                                                      score=self.score, score_params=self.score_params, prob_type=self.prob_type, tf_log_flag=self.tf_log_flag)
                elif self.search_type == 'a1_vs_a2':
                    probs = self.locdic.double_search(words1=a_words1, words2=a_words2, words1_weights=a_weights1, words2_weights=a_weights2, 
                                                      num_words1=self.num_words_ans, num_words2=self.num_words_ans,
                                                      score=self.score, score_params=self.score_params, prob_type=self.prob_type, tf_log_flag=self.tf_log_flag)
                print '  answers #%d, #%d: %.2e , %.2e' % (ia1, ia2, probs[0], probs[1])
                sim_scores[(ia1,ia2)] = probs[0]
        # The score of each answer is its best similarity to another answer
        scores = []
        for ia in range(len(answers)):
            sscrs = sorted([scr for (ia1,ia2),scr in sim_scores.iteritems() if (ia1==ia) or (ia2==ia)], reverse=True)
            scores.append([np.dot(sscrs, self.sim_scores_comb_weights[0]), np.dot(sscrs, self.sim_scores_comb_weights[1])])
#         print 'scores: %s' % scores
        if self.norm_scores:
            ref_score0, ref_score1 = np.sum([s[0] for s in scores]) + len(scores)*EPSILON, np.sum([s[1] for s in scores]) + len(scores)*EPSILON
            scores = [((s[0]+EPSILON)/ref_score0, (s[1]+EPSILON)/ref_score1) for s in scores]
        print ' -> scores: %s ; %s' % (', '.join(['%.2e'%s[0] for s in scores]), ', '.join(['%.2e'%s[1] for s in scores]))
        return np.array(scores)

"""
Quueries a Lucene index with words in question in conjunction with words in an answer
returns an array of scores as given by the Lucene.  
"""
class AnswersLuceneSearchFunc(AnswersFunc):
    def __init__(self, lucene_corpus, parser, max_docs, weight_func=lambda n: np.ones(n), score_func=None, norm_scores=True):
        self.lucene_corpus = lucene_corpus
        self.parser = parser
        self.max_docs = max_docs
        self.weight_func = weight_func
        if score_func is None:
            self.score_func = lambda s: s
        else:
            self.score_func = score_func
        self.norm_scores = norm_scores
        
    def __call__(self, question, answers):
        EPSILON = 1E-30
        print 'question = %s' % question
        if self.parser is None:
            q_words = question
        else:
            q_words = self.parser.parse(question, calc_weights=False)
            print '  -> %s' % ' ; '.join(q_words)
        scores = []
        for ia,answer in enumerate(answers):
            if self.parser is None:
                a_words = answer
                if len(a_words.strip()) > 0:
                    search_words = '(%s) AND (%s)' % (q_words, a_words)
                else:
                    search_words = q_words
            else:            
                a_words = self.parser.parse(answer, calc_weights=False)
                print '  -> %s' % ' ; '.join(a_words)
                search_words = q_words + a_words
            score = self.lucene_corpus.search(words=search_words, max_docs=self.max_docs, weight_func=self.weight_func, score_func=self.score_func)
            print '  answer #%d: %.2f' % (ia, score)
            scores.append(score)

        if self.norm_scores:
            ref_score0 = np.sum(scores) + len(scores)*EPSILON
            scores = np.array(scores)/ref_score0
        print ' -> scores: %s' % (', '.join(['%.2e'%s for s in scores]))
        return np.asarray(scores)    
    

#################################################################################################
# Location dictionary
#################################################################################################
                
class LocationDictionary(object):
    def __init__(self, save_locations=False, doc_name_weight=0, base_locdic=None):
        self.save_locations = save_locations
        self.doc_name_weight = doc_name_weight
        self.reset()
        self.set_search_word_filter()
        if base_locdic is not None:
            self.copy(base_locdic)
    
    def reset(self):
        self.doc_ids, self.doc_names, self.doc_name_words = {}, [], []
        self.word_ids = {}
        self.word_counts = []
        self.word_locations, self.word_doc_names = [], []
        self.doc_lengths, self.doc_name_lengths = [], []
        self.doc_unique_words, self.doc_name_unique_words = [], []
        self.total_count = 0.0
        self._cache = {}
        self._cache_keys = None
        self._search_cache = None
        self._tf_idf_cache = None

    def copy(self, base_locdic):
        self.doc_ids = base_locdic.doc_ids.copy()
        self.doc_names = list(base_locdic.doc_names)
        self.doc_name_words = list(base_locdic.doc_name_words) 
        self.word_ids = base_locdic.word_ids.copy()
        self.word_counts = list(base_locdic.word_counts)
        self.word_locations = [wl.copy() for wl in base_locdic.word_locations]
        self.word_doc_names = [wd.copy() for wd in base_locdic.word_doc_names]
        self.doc_lengths = list(base_locdic.doc_lengths)
        self.doc_name_lengths = list(base_locdic.doc_name_lengths)
        self.doc_unique_words = list(base_locdic.doc_unique_words)
        self.doc_name_unique_words = list(base_locdic.doc_name_unique_words)
        self.total_count = base_locdic.total_count
        
    def add_words(self, doc_name, doc_name_words, words):
#         print 'Adding words: %s' % ' ; '.join(words)
        if self.doc_ids.has_key(doc_name):
            assert self.doc_name_weight == 0
            assert 'Should not be here?!...'
        else:
            self.doc_ids[doc_name] = len(self.doc_ids)
            self.doc_names.append(doc_name)
            words = doc_name_words + words # use the doc's name as part of the doc's content (important also for search)
            self.doc_lengths.append(0) 
            self.doc_unique_words.append(0)
            if self.doc_name_weight != 0:
                self.doc_name_words.append(doc_name_words)
                self.doc_name_lengths.append(0)
                self.doc_name_unique_words.append(0)
        doc_id = self.doc_ids[doc_name]
        self.doc_lengths[doc_id] += len(words)
        if self.doc_name_weight != 0: 
            self.doc_name_lengths[doc_id] += len(doc_name_words)
        for iw,word in enumerate(words):
            if not self.word_ids.has_key(word):
                self.word_ids[word] = len(self.word_ids)
                self.word_counts.append(0)
                self.word_locations.append({})
                if self.doc_name_weight != 0:
                    self.word_doc_names.append({})
            word_id = self.word_ids[word]
            self.word_counts[word_id] += 1
            self.total_count += 1.0
            if self.save_locations:
                if not self.word_locations[word_id].has_key(doc_id):
                    self.word_locations[word_id][doc_id] = []
                    self.doc_unique_words[doc_id] += 1
                self.word_locations[word_id][doc_id].append(iw) 
            else:
                if not self.word_locations[word_id].has_key(doc_id):
                    self.word_locations[word_id][doc_id] = 0 # we save the number of occurrences, not the list of locations...
                    self.doc_unique_words[doc_id] += 1
                self.word_locations[word_id][doc_id] += 1
        if self.doc_name_weight != 0:
            for iw,word in enumerate(doc_name_words):
                word_id = self.word_ids[word] # should already be here, since doc name words are part of the doc's words
                if not self.word_doc_names[word_id].has_key(doc_id):
                    self.word_doc_names[word_id][doc_id] = 0 
                    self.doc_name_unique_words[doc_id] += 1
                self.word_doc_names[word_id][doc_id] += 1
             
    def get_word(self, word_id):
        return [word for word,wid in self.word_ids.iteritems() if wid==word_id][0]
     
    def get_num_docs(self):
        return len(self.doc_ids)
    
    def get_word_num_docs(self, word_id):
        return len(self.word_locations[word_id])

    def get_word_num_doc_names(self, word_id):
        return len(self.word_doc_names[word_id])
    
    def get_word_tf_idf(self, word_id, doc_id, tf_log_flag=False, doc_name_flag=False):
        if doc_name_flag:
            doc_len = self.doc_name_lengths[doc_id]
            word_count_in_doc = self.word_doc_names[word_id].get(doc_id, 0)
            num_doc_unique_words = self.doc_name_unique_words[doc_id]
            word_num_docs = self.get_word_num_doc_names(word_id)
            if doc_len==0 or word_count_in_doc==0: 
                return 0.0
        else:
#             if self._tf_idf_cache is None:
#                 self._tf_idf_cache_log_flag = tf_log_flag
#                 self._tf_idf_cache = {}
#             else:
#                 assert self._tf_idf_cache_log_flag == tf_log_flag
#             cache_key = (doc_id, word_id)
#             if self._tf_idf_cache.has_key(cache_key):
#                 return self._tf_idf_cache[cache_key]
            doc_len = self.doc_lengths[doc_id]
            word_count_in_doc = len(self.word_locations[word_id][doc_id]) if self.save_locations else self.word_locations[word_id][doc_id]
            num_doc_unique_words = self.doc_unique_words[doc_id]
            word_num_docs = self.get_word_num_docs(word_id)
        assert word_count_in_doc>=1 and doc_len>=1
        if tf_log_flag:
            tf = np.log(1.0 + word_count_in_doc + 0.5/num_doc_unique_words) / np.log(1.0 + doc_len + 0.5)
        else:
            tf = (word_count_in_doc + 0.5/num_doc_unique_words) / (doc_len + 0.5)
#         print '  word %s (%s) , doc %s (%s)' % (word_id, self.get_word(word_id), doc_id, self.doc_names[doc_id])
#         print '    TF  = (%d + 0.5/%d) / (%d + 0.5) = %.5f' % (word_count_in_doc, num_doc_unique_words, doc_len, tf)
        idf = np.log((self.get_num_docs() + 0.5) / (word_num_docs + 0.5))
#         print '    IDF = log((%d +0.5) / (%d + 0.5)) = %.5f' % (self.get_num_docs(), word_num_docs, idf)
#         print '  -> TF*IDF = %.6f' % (tf*idf)
#         if not doc_name_flag:
#             self._tf_idf_cache[cache_key] = tf*idf 
        return tf*idf
        
    def sort_words_by_num_docs(self):
        ''' 
        Sort the number of docs each word appears in (sorted in descending order)
        '''
        return sorted([(self.get_word_num_docs(wid),word) for word,wid in self.word_ids.iteritems()], reverse=True)
    
    def sort_words_by_count(self):
        '''
        Sort the word by their count (in descending order)
        '''
        words = sorted(self.word_ids.keys(), key=lambda w: self.word_ids[w])
        return np.take(words, np.argsort(self.word_counts)[::-1])
        
    def set_search_word_filter(self, min_word_docs_frac=0, max_word_docs_frac=0.1, min_word_count_frac=0, max_word_count_frac=0.01):
        '''
        Set the min/max fraction of documents that each search word may appear in
        '''
        assert min_word_docs_frac <= max_word_docs_frac and min_word_count_frac <= max_word_count_frac 
        self.min_word_docs_frac = min_word_docs_frac
        self.max_word_docs_frac = max_word_docs_frac
        self.min_word_count_frac = min_word_count_frac
        self.max_word_count_frac = max_word_count_frac

    def _check_cache(self):
        if self._cache_keys != (self.total_count, self.min_word_docs_frac, self.max_word_docs_frac, self.min_word_count_frac, self.max_word_count_frac):
            # (Re-)compute cache
            print '=> Computing cache'
            self._cache_keys = (self.total_count, self.min_word_docs_frac, self.max_word_docs_frac, self.min_word_count_frac, self.max_word_count_frac)
            self._cache = {}
            self._cache['is_word_filtered'] = [self._filter_word_for_search(wid) for wid in range(len(self.word_ids))]
#             self._cache['word_probs'] = []
#             n_docs = self.get_num_docs() + 0.0
#             for wid in range(len(self.word_ids)):
#                 if self._cache['is_word_filtered'][wid]:
#                     self._cache['word_probs'].append(0) # we don't want to select a filtered word
#                 else:
#                     self._cache['word_probs'].append(self.word_counts[wid])
#             sum_probs = np.sum(self._cache['word_probs'])
#             self._cache['word_probs'] = np.array(np.cumsum(self._cache['word_probs']), dtype=np.float32) / (sum_probs + 0.0)
#             print '   Total for probs = %d (%d filtered words: %s)' % (sum_probs, np.sum(self._cache['is_word_filtered']), 
#                                                                        [word for word,wid in self.word_ids.iteritems() if self._cache['is_word_filtered'][wid]])
#             print '   Total count = %d' % np.sum(self.word_counts)
            print '   Total %d filtered words: %s' % (np.sum(self._cache['is_word_filtered']), 
                                                      [word for word,wid in self.word_ids.iteritems() if self._cache['is_word_filtered'][wid]])
        
    def _filter_word_for_search(self, word_id):
        n_docs = self.get_num_docs() + 0.0
        n_words = self.total_count + 0.0
        return ((self.get_word_num_docs(word_id)/n_docs) < self.min_word_docs_frac or (self.get_word_num_docs(word_id)/n_docs) > self.max_word_docs_frac or
                (self.word_counts[word_id]/n_words) < self.min_word_count_frac or (self.word_counts[word_id]/n_words) > self.max_word_count_frac)
        
    def _is_word_filtered(self, word_id):
        return self._cache['is_word_filtered'][word_id]
    
    def _filter_words_for_search(self, word_ids):
        self._check_cache()
        return [wid for wid in word_ids if not self._is_word_filtered(wid)] 
      
    def _get_random_words(self, num_words):
        '''
        Return num_words word ids, randomly chosen from the set of un-filtered words (with repetitions)
        '''
        assert False, 'Not used'
        self._check_cache()
        rwords = []
        for i in range(num_words):
            rwords.append(np.searchsorted(self._cache['word_probs'], np.random.rand(), side='left'))
#             print ' random word: %4d - %s' % (rwords[-1], [word for word,wid in self.word_ids.iteritems() if wid==rwords[-1]])
        return rwords
    
    def search_docs(self, words, word_ids=None, words_weights=None, min_words_per_doc=None, max_words_distance=None, 
                    prob_type='tf-idf', tf_log_flag=True): 
        assert max_words_distance is None, 'max_words_distance not supported yet'
        assert prob_type in ['word-probs', 'tf-idf']
#         start_time = time.time()
#         print '-> Searching for words: %s' % str(words)
        if words_weights is None:
            words_weights = {} # default weight is 1
        word_ids_weights = dict([(self.word_ids[word], wgt) for word,wgt in words_weights.iteritems() if self.word_ids.has_key(word)])
#         print 'word_ids_weights: %s' % word_ids_weights
        
        if words is not None:
            assert word_ids is None
            words = np.unique(words)
            # Get the word ids to search for
            word_ids = [self.word_ids[word] for word in words if self.word_ids.has_key(word)]
#             print 'IDs: %s' % word_ids
        else:
            word_ids = np.unique(word_ids)
        word_ids = self._filter_words_for_search(word_ids)
        # Ignore words with weight=0
        word_ids = [wid for wid in word_ids if word_ids_weights.get(wid,1)!=0]
#         print 'words: %s' % words
#         print 'word_ids_weights: %s' % word_ids_weights
        # Normalize weights
        wmean = np.mean([word_ids_weights.get(wid,1.0) for wid in word_ids])
        for wid in word_ids:
            word_ids_weights[wid] = word_ids_weights.get(wid, 1.0) / wmean
#         print '-> word_ids weights: %s' % word_ids_weights
            
#         print '    init time = %.3f secs.' % (time.time()-start_time)

#         print 'Searching for filtered words: %s' % ', '.join(['%s (%s)'%(wid,self.get_word(wid)) for wid in word_ids])
        if len(word_ids)==0:
            return dict([((0 if mwpd is None else mwpd),([],[])) for mwpd in min_words_per_doc])
#         if min_words_per_doc is None:
#             min_words_per_doc = len(word_ids)
#         assert min_words_per_doc > 0
#         if min_words_per_doc > len(word_ids):
#             return dict([(mwpd,([],[])) for mwpd in min_words_per_doc])

        # Find all relevant docs
#         start_time = time.time()
        words_per_doc, words_per_doc_name = {}, {}
        for wid in word_ids:
            for did in self.word_locations[wid].keys():
#                 words_per_doc[did] = words_per_doc.get(did,set()).union([wid])
                if not words_per_doc.has_key(did):
                    words_per_doc[did] = []
                words_per_doc[did].append(wid)
                if self.doc_name_weight != 0: 
#                     words_per_doc_name[did] = words_per_doc_name.get(did,set()).union([wid])
                    if not words_per_doc_name.has_key(did):
                        words_per_doc_name[did] = []  
                    words_per_doc_name[did].append(wid)
#         print 'words_per_doc: %s' % words_per_doc
#         print 'words_per_doc_name: %s' % words_per_doc_name
#         print '    find time = %.3f secs.' % (time.time()-start_time)

#         start_time = time.time()

        min_words_per_doc2 = []
        for mwpd in min_words_per_doc:
            if mwpd is None:
                mwpd = len(word_ids)
            elif mwpd < 0:
                mwpd = np.max([1, len(word_ids) + mwpd])
            elif mwpd > len(word_ids):
                continue
            min_words_per_doc2.append(mwpd)
        if len(min_words_per_doc2) == 0:
            return {}
        min_words_per_doc2 = np.unique(min_words_per_doc2)[::-1] # sort in reverse order and remove duplicates
#         print 'min_words_per_doc  = %s' % min_words_per_doc
#         print 'min_words_per_doc2 = %s' % min_words_per_doc2
        
        docs_probs = {}
        prev_doc_probs = {}
        min_mwpd = min_words_per_doc2[-1]
        for mwpd in min_words_per_doc2:
#             print '-> mwpd = %d' % mwpd
            if mwpd == 0:
                docs_probs[mwpd] = ([], []) # dummy empty sets for searching with 0 min words
                continue
            assert (mwpd > 0) and (mwpd <= len(word_ids))
            docs = [doc for doc,ws in words_per_doc.iteritems() if len(ws)>=mwpd]
#             assert set(docs).issuperset(prev_doc_probs.keys())
#             print 'mwpd=%d, docs: %s' % (mwpd, docs)
            # Compute probability/score of each document
            n_docs = self.get_num_docs() + 0.0
            probs = []
            for doc in docs:
                if prev_doc_probs.has_key(doc):
                    prob = prev_doc_probs[doc]
#                     print ' using prob %.3f for doc %s' % (prob, doc)
                else:
                    if prob_type == 'word-probs':
                        prob = -np.sum([np.log(word_ids_weights[wid]*(self.get_word_num_docs(wid)+0.5)/(n_docs+0.5))/len(word_ids) for wid in words_per_doc[doc]])
                        if self.doc_name_weight != 0:
                            prob += self.doc_name_weight * -np.sum([np.log(word_ids_weights[wid]*(self.get_word_num_doc_names(wid)+0.5)/(n_docs+0.5))/len(word_ids) for wid in words_per_doc_names[doc]])
                    elif prob_type == 'tf-idf':
                        prob = np.sum([word_ids_weights[wid]*self.get_word_tf_idf(wid, doc, tf_log_flag=tf_log_flag, doc_name_flag=False)/len(word_ids) for wid in words_per_doc[doc]])
                        if self.doc_name_weight != 0:
                            prob += self.doc_name_weight * np.sum([word_ids_weights[wid]*self.get_word_tf_idf(wid, doc, tf_log_flag=tf_log_flag, doc_name_flag=True)/len(word_ids) for wid in words_per_doc[doc]])
                    else:
                        raise ValueError('Unknown prob_type')
    #                     print '  doc %s (%s): prob = %7.4f' % (doc, self.doc_names[doc], prob)
#                     print ' computed prob %.3f for doc %s' % (prob, doc)
                    if mwpd > min_mwpd: # no pointing in caching values we won't be using...
                        prev_doc_probs[doc] = prob
                assert prob>=0
                probs.append(prob)
            docs_probs[mwpd] = (docs, probs)

#         print '    probs time = %.3f secs.' % (time.time()-start_time)
        return docs_probs
            
    """
    This is typically called with words1 = words in question and words2 = words in answer
    when score is 'hg' -- returns the hypergeometric prob. scores in both directions with hg_test routine
    when score is 'counts' -- returns the number of docs in the intersection, union of words1 and words2
    when score is 'weights' -- returns the dot product of weights of the docs retrieved in the intersection of both question, answer and only answer but not in question with the coeffs set in the score_params map with 'coeffs' key. Note that words1_weight and words2_weight are reflected in the scores returned by the search_docs
    returns two stat related params where the first one is an absolute measure of how close the two word sets (words1, words2) are related and second one is a usually a number to normalize the first number for comparison
    """
    def double_search(self, words1, words2, num_words1=[None], num_words2=[None], words1_weights={}, words2_weights={}, 
                      score='hg', score_params=None, prob_type='tf-idf', tf_log_flag=True):
        assert score in ['counts','hg','weights']
#         print '--> double_search: [%s] , [%s]' % (words1, words2)
#         docs1 = [(min_words_per_doc1,self.search_docs(words1, min_words_per_doc=[min_words_per_doc1])[0][0]) for min_words_per_doc1 in num_words1]
#         docs2 = [(min_words_per_doc2,self.search_docs(words2, min_words_per_doc=[min_words_per_doc2])[0][0]) for min_words_per_doc2 in num_words2]
        
#         words1_weights = {}
#         for iw,word in enumerate(words1):
#             words1_weights[word] = words1_weights.get(word,0.0) + ((10.0+iw)**4)/(10.0**4)    # (10.0+iw)**4/(10.0**4): 0.4092
# #         wsum = np.sum(words1_weights.values())
# #         for word in words1_weights.keys():
# #              words1_weights[word] = words1_weights[word] / wsum
# #         print 'words1_weights: %s' % words1_weights

#         start_time = time.time()
            
        if (self._search_cache is not None) and (self._search_cache['words'] == words1) and \
           (self._search_cache['num_words'] == num_words1) and (self._search_cache['params'] == (score,prob_type,tf_log_flag)):
            docs1 = self._search_cache['docs']
        else:
            docs1 = self.search_docs(words1, words_weights=words1_weights, 
                                     min_words_per_doc=num_words1, prob_type=prob_type, tf_log_flag=tf_log_flag)
            self._search_cache = {'words': words1, 'num_words': num_words1, 'params': (score,prob_type,tf_log_flag), 'docs': docs1}
        docs2 = self.search_docs(words2, words_weights=words2_weights,
                                 min_words_per_doc=num_words2, prob_type=prob_type, tf_log_flag=tf_log_flag)
        
#         end_time = time.time()
#         print 'search_docs time = %.3f secs.' % (end_time-start_time)
        
#         print 'docs1: %s' % docs1
#         print 'docs2: %s' % docs2
#         num_random = 1
#         docs_rand = []
#         for ri in range(num_random):
#             rwords = self._get_random_words(len(words2))
#             docs_rand.append([(min_words_per_doc2,self.search_docs(words=None, word_ids=rwords, min_words_per_doc=min_words_per_doc2)[0]) for min_words_per_doc2 in num_words2])
# #             print 'docs_rand: %s' % str(docs_rand[-1])
#         wi_time = 0
#         start_time = time.time()
        
        best_score_over, best_score_under = 0, 0
        for mw1,(d1,p1) in docs1.iteritems():
            if len(d1)==0 and mw1>0: continue
#             print 'mw1=%s , d1: %s' % (str(mw1), str(d1)) 
            for mw2,(d2,p2) in docs2.iteritems():
#                 if words2==['water']:
#                 print 'mw2=%s , d2: %s' % (str(mw2), str(d2))
                if len(d2)==0: continue
                assert mw2 > 0 
                if mw1 == 0:
                    d12 = d2
                else:
                    d12 = set(d1).intersection(d2)
#                 if words2==['water']:       
#                     print 'min words %s,%s -> d1 %d , d2 %d -> d12 %d' % (mw1, mw2, len(d1), len(d2), len(d12))
                # Compute intersections for random words
#                 rand_intersections = []
#                 for ri,docsr in enumerate(docs_rand):
#                     for mwr,dr in docsr:
# #                         print 'docsr: %s , %s' % (str(mwr), dr)
#                         if mwr==mw2:
#                             d1r = set(d1).intersection(dr)
# #                             print '  rand %d: min words %s,%s -> d1 %d , dr %d -> d1r %d' % (ri, mw1, mw2, len(d1), len(dr), len(d1r))
#                             rand_intersections.append(len(d1r))
#                             break
                if score == 'hg':
                    assert len(d1) > 0
                    # Compute p-value for over/under-representation of intersection
                    score_over  = -np.log10(np.max([1E-320, hg_test(M=int(self.get_num_docs()), n=len(d1), N=len(d2), k=len(d12), dir= 1)['prob']]))
                    score_under = -np.log10(np.max([1E-320, hg_test(M=int(self.get_num_docs()), n=len(d1), N=len(d2), k=len(d12), dir=-1)['prob']]))
    #                 rand_rank_over  = 1.0 - scipy.stats.percentileofscore(rand_intersections, len(d12), kind='strict')/100.0
    #                 rand_rank_under = scipy.stats.percentileofscore(rand_intersections, len(d12), kind='weak')/100.0
    #                 print ' -> pvals: %.2f , %.2f ; rank = %.2f , %.2f' % (pval_over, pval_under, rand_rank_over, rand_rank_under)
                elif score == 'counts':
                    score_over  = len(d12)
                    score_under = len(d1)+len(d2)-len(d12)
                elif score == 'weights':
                    if (score_params is not None) and score_params.has_key('norm'):
                        p1, p2 = score_params['norm'](np.asarray(p1)), score_params['norm'](np.asarray(p2))
                    if mw1 == 0:
                        w1 = dict(zip(d2,np.ones(len(d2)))) # use score=1 for each document in d2 (since we didn't actually search for any words from words1) 
                    else:
                        w1 = dict(zip(d1,p1))
                    w2 = dict(zip(d2,p2))
                    if False:
                        print 'intersection weights:'
                        for did in d12:
                            print ' doc id %-7s: %7.4f * %7.4f = %7.4f  (%s)' % (did, w1[did], w2[did], w1[did]*w2[did], self.doc_names[did])
#                     t1 = time.time()
                    weights_intersection = sorted([w1[did]*w2[did] for did in d12], reverse=True)
#                     wi_time += (time.time() - t1)
                    
                    if len(weights_intersection)==0:
                        score_over = 0.0
                    else:
                        if (score_params is not None) and score_params.has_key('coeffs'):
                            coeffs = score_params['coeffs'](len(weights_intersection))
                        else:
                            coeffs = np.ones(len(weights_intersection))
                        ##score_over  = np.average(weights_intersection, weights=coeffs) #np.mean(weights_intersection[-10:])
#                         print 'weights intersection: %s' % ', '.join(['%7.4f'%x for x in weights_intersection])
#                         print 'coeffs              : %s' % ', '.join(['%7.4f'%x for x in coeffs])
                        score_over  = np.dot(weights_intersection, coeffs)
#                         print '-> score_over = %.4f' % score_over 

                    d2not1 = set(d2).difference(d1)
                    weights_c_intersection = sorted([w2[did] for did in d2not1], reverse=True)
                    if len(weights_c_intersection)==0:
                        score_under = 0.0
                    else:
                        if (score_params is not None) and score_params.has_key('coeffs'):
                            coeffs = score_params['coeffs'](len(weights_c_intersection))
                        else:
                            coeffs = np.ones(len(weights_c_intersection))
##                        score_under = np.average(weights_c_intersection, weights=coeffs) #np.mean(weights_intersection[:10])
                        score_under = np.dot(weights_c_intersection, coeffs) 
#                     print '   score over = %7.2f , under = %7.2f' % (score_over, score_under)

                    if (score_params is not None) and score_params.has_key('calc_over_vs_under') and (score_params['calc_over_vs_under']==True):
                        w1p, w2p = np.sum(w1.values()) / self.get_num_docs(), np.sum(w2.values()) / self.get_num_docs()
                        oexp = w1p * w2p * self.get_num_docs()
                        oscore = (np.sum(weights_intersection) - oexp) / np.sum([np.sqrt(oexp*(1.0-w1p)), np.sqrt(oexp*(1.0-w2p))])
                        #score_under = np.clip(oscore/100.0 + 0.5, 0.0, 1.0) # change from std's to scale 0...1
                        score_under = scipy.stats.norm.cdf(oscore)
#                         print 'w1p=%.3f, w2p=%.3f -> oexp=%.3f ; wi=%.3f -> score under = %.3f' % (w1p, w2p, oexp, np.sum(weights_intersection), score_under) 
#                         avg_d1 = np.mean(w1.values())
#                         print 'mw1=%d, mw2=%d: over = %.3f , under = %.3f (avg d1 = %.3f) -> over/under = %.3f' % (mw1, mw2, score_over, score_under, avg_d1, score_over / (score_over + avg_d1 * score_under))
#                         score_under = score_over / np.sqrt(score_over + avg_d1 * score_under) # w/o sqrt: 0.4152

                assert score_over>=0 and score_under>=0
#                 print ' mw1=%2d , mw2=%2d -> score over = %.3f' % (mw1, mw2, score_over)
                if (score_params is not None) and score_params.has_key('minword1_coeffs'):
                    mwc1 = score_params['minword1_coeffs'](mw1, np.max(docs1.keys()))
                    mwc2 = score_params['minword2_coeffs'](mw2, np.max(docs2.keys()))
#                     print '   adding %.3f * %.3f = %.3f' % (score_over, mwc1*mwc2, score_over*mwc1*mwc2)
                    best_score_over  += score_over  * mwc1 * mwc2
                    best_score_under += score_under * mwc1 * mwc2
                else:
                    best_score_over  = np.max([best_score_over , score_over])
                    best_score_under = np.max([best_score_under, score_under])
#         print '---> best scores: %.3f , %.3f' % (best_score_over, best_score_under)

#         end_time = time.time()
#         print 'compute score time = %.3f secs. (wi time = %.3f)' % (end_time-start_time, wi_time)

        return best_score_over, best_score_under
        
        
def build_training_location_dictionary(train, parser=SimpleWordParser(), 
                                       use_questions=True, use_answers=True, min_words_per_qa=1, base_locdic=None,
                                       min_word_docs_frac=0, max_word_docs_frac=0.2, min_word_count_frac=0, max_word_count_frac=0.01,
                                       ascii_conversion=True):
#     print '=> Building LocationDictionary for %d training samples' % len(train)
    parser.ascii_conversion = ascii_conversion
    locdic = LocationDictionary(doc_name_weight=0, base_locdic=base_locdic)
    locdic.set_search_word_filter(min_word_docs_frac=min_word_docs_frac, max_word_docs_frac=max_word_docs_frac,
                                  min_word_count_frac=min_word_count_frac, max_word_count_frac=max_word_count_frac)
    if use_answers:
        for i,(qid,qst,ans) in enumerate(np.array(train[['ID','question','answer']])):
            words = parser.parse(qst) if use_questions else []
            words += parser.parse(ans)
            if len(words) >= min_words_per_qa:
                locdic.add_words('train_%s_%d'%(qid,i), [], words)
    else:
        assert use_questions
        for qst,ids in train.groupby('question').groups.iteritems():
            words = parser.parse(qst)
            if len(words) >= min_words_per_qa:
                locdic.add_words('train_%s'%(train.irow(ids[0])['ID']), [], words)      
    return locdic

def build_files_location_dictionary(filenames, dirname, file_suffix, part_sep='\r\n\r\n', min_words_in_part=10, 
                                    parser=SimpleWordParser(),
                                    min_word_docs_frac=0, max_word_docs_frac=0.2, min_word_count_frac=0, max_word_count_frac=0.01,
                                    ascii_conversion=True):
    parser.ascii_conversion = ascii_conversion
    if filenames is None:
        filenames = ['%s/%s'%(dirname,fname) for fname in os.listdir(dirname) if fname.endswith(file_suffix)]
        assert len(filenames)>0
    locdic = LocationDictionary(doc_name_weight=0)
    locdic.set_search_word_filter(min_word_docs_frac=min_word_docs_frac, max_word_docs_frac=max_word_docs_frac,
                                  min_word_count_frac=min_word_count_frac, max_word_count_frac=max_word_count_frac)
    total_parts = 0
    for fname in filenames:
        with open (fname, 'rb') as myfile:
            text = myfile.read()#.replace('\x00', ' ') 
        parts = re.split(part_sep, text)
#         print '%s' % text
#         print 'Found %d parts: [0] %s ; [1] %s ; [2] %s' % (len(parts), parts[0][:50], parts[1][:50], parts[2][:50])
        print 'Found %d parts' % len(parts)
        for pai,part in enumerate(parts):
            if len(part)>0:
#                 print '-------------------------------------------------------------- 1'
#                 print 'part: <<<%s>>>' % part
                words = parser.parse(part)
#                 print '-------------------------------------------------------------- 3'
#                 print 'words: %s' % ' ; '.join(words)
#                 for word in words:
#                     for c in word:
#                         if (ord(c)<ord('a') or ord(c)>ord('z')) and (ord(c)<ord('0') or ord(c)>ord('9')) and ord(c)!=ord('.'):
#                             print 'word=%s [%s] ord %d' % (word, c, ord(c))
#                             jshdjshd()
                if len(words) >= min_words_in_part:
#                     print '+++++ Adding part with %d words:\n%s' % (len(words), part)
#                     print '===== Words:\n%s' % ' ; '.join(words)
                    locdic.add_words('%s_p%d'%(fname,pai), None, words)
                    total_parts += 1
#                 else:
#                     print '----- Skipping part with %d words:\n%s' % (len(words), part)
#             if pai>3: 
#                 skdjskdj()
    print 'Read total of %d parts from %d files' % (total_parts, len(filenames))
    return locdic


#################################################################################################
# Regressor
#################################################################################################

class BaseRegressor(object):
    CACHE_DIR = None
    
    def __init__(self, name, use_cache, calc_train_preds=False):
        self.name = name
        self._use_cache = use_cache
        self._calc_train_preds = calc_train_preds
        self.cache = {} # cache results
        self.set_weights(train_weights=None, test_weights=None, aux_test_weights=None)
    
    def run(self, train, test, aux_test=None, cache_key=None):
        train_preds, test_preds, aux_test_preds = None, None, None 
        if self._use_cache and (cache_key is not None):
            # Try to load cached results
            res = self._read_from_cache(cache_key)
            if res is not None:
                assert len(res)==6 
                train_preds, test_preds, aux_test_preds, train_weights, test_weights, aux_test_weights = res
                self.set_weights(train_weights, test_weights, aux_test_weights)
        
        if test_preds is None:
            # Actually run the regressor
            train_preds, test_preds, aux_test_preds = self._do_run(train, test, aux_test, cache_key)
            if self._use_cache and (cache_key is not None):
                # Save results to cache
                train_weights, test_weights, aux_test_weights = self.get_weights()
                self._save_to_cache(cache_key, [train_preds, test_preds, aux_test_preds, train_weights, test_weights, aux_test_weights])
        return train_preds, test_preds, aux_test_preds

    def set_weights(self, train_weights, test_weights, aux_test_weights):
        self.train_weights, self.test_weights, self.aux_test_weights = train_weights, test_weights, aux_test_weights
        
    def get_weights(self):
        return self.train_weights, self.test_weights, self.aux_test_weights
    
    def _prp_build_data(self, train, build_fraction, test_fraction, random_seed):
        assert build_fraction>=0 and test_fraction>=0 and build_fraction+test_fraction<=1.0
        np.random.seed(random_seed)
        shuffled_ids = range(len(train))
        np.random.shuffle(shuffled_ids)
        n_build = int(build_fraction * len(shuffled_ids))
        n_test  = int(test_fraction  * len(shuffled_ids))
        build_ids = shuffled_ids[:n_build]
        if n_test==0:
            train_ids = shuffled_ids[n_build:]
            test_ids  = []
        else:
            train_ids = shuffled_ids[n_build:-n_test]
            test_ids  = shuffled_ids[-n_test:]
        print 'Prepared %d/%d/%d build/train/test ids' % (len(build_ids), len(train_ids), len(test_ids))
        return build_ids, train_ids, test_ids
    
    def _cache_filename(self, cache_key):
        return '%s/%s_%s' % (BaseRegressor.CACHE_DIR, self.name, '_'.join(['%s'%str(c) for c in cache_key]))
        
    def _read_from_cache(self, cache_key):
        create_dirs([BaseRegressor.CACHE_DIR])
        filename = self._cache_filename(cache_key)
        #print 'Loading from cache %s' % filename
        return load_from_pkl(filename)

    def _save_to_cache(self, cache_key, data):
        create_dirs([BaseRegressor.CACHE_DIR])
        filename = self._cache_filename(cache_key)
        print 'Saving to cache %s' % filename
        return save_to_pkl(filename, data)
        
class Regressor(BaseRegressor):
    ALL_MEANS_FEATURE = 'all_means' # sum of all the "means" features
    def __init__(self, name, clf, features, target, build_fraction, random_seed, z_transform=True, use_cache=False, calc_train_preds=False,
                 predict_proba=True):
        self.clf = clf
        self.features = features
        if self.features is not None:
            self.features = np.array(features)
        self.target = target
        self.build_fraction = build_fraction 
        self._random_seed = random_seed
        self.z_transform = z_transform
        self.predict_proba = predict_proba
        BaseRegressor.__init__(self, name, use_cache, calc_train_preds)
    
    def _transform_preds(self, preds):
        print 'preds: %s' % preds[:5]
        if self.z_transform:
            assert self.predict_proba
            # Convert scores from p-values to Z-scores
            preds = -scipy.stats.norm.isf(preds)
        print 'preds: %s' % preds[:5]
        return preds

    def _do_run(self, train, test, aux_test, cache_key):
        if self.features is None:
            features = np.array([f for f in train.columns if f!=self.target])
            print 'Using all supplied features: %s' % features
        else:
            features = self.features 
        print 'Building %d features' % len(features)
        build_ids, train_ids, _ = self._prp_build_data(train, build_fraction=self.build_fraction, test_fraction=0.0, random_seed=self._random_seed)
        f_builder = FeatureBuilder(train, test, aux_test=aux_test, build_ids=build_ids, target_col=self.target)
        means_features = [f for f in features if f.endswith('_mean')]
        f_builder.build_means([f[:-5] for f in means_features])
        lin_features = [f for f in features if f.endswith('_linreg')]
        f_builder.build_linreg([f[:-7] for f in lin_features])
        if Regressor.ALL_MEANS_FEATURE in features:
            for ds in [train, test, aux_test]:
                if ds is not None:
                    ds[Regressor.ALL_MEANS_FEATURE] = np.sum(ds[means_features], axis=1)
            print 'train all_means: %s' % train[Regressor.ALL_MEANS_FEATURE][:5]
        print 'Running regressor %s on %d,%d,%d train,test,aux_test samples (%d features)' % (self.name, len(train), len(test), 0 if aux_test is None else len(aux_test), len(features))
        #print '%s' % train[features][:5]
        sys.stdout.flush()
        if hasattr(self.clf, 'set_eval_dataf'):
            print 'Setting eval dataf'
            self.clf.set_eval_dataf(test[features], test[self.target])
        self.clf.fit(train.take(train_ids)[features], train.take(train_ids)[self.target])
        train_preds, test_preds, aux_test_preds = None, None, None
        if self.predict_proba:
            clf_pred_func = lambda x: self.clf.predict_proba(x)[:,1]
        else:
            clf_pred_func = lambda x: self.clf.predict(x)
        if self._calc_train_preds:
            if len(train)==0:
                train_preds = []
            else:
                train_preds = self._transform_preds(clf_pred_func(train[features]))
        if len(test)==0:
            test_preds = []
        else:
            test_preds = self._transform_preds(clf_pred_func(test[features]))
        if aux_test is not None:
            aux_test_preds = self._transform_preds(clf_pred_func(aux_test[features]))
        return train_preds, test_preds, aux_test_preds
        
class TwoPhaseRegressor(BaseRegressor):
    def __init__(self, name, phase1_regressors, phase2_regressor, phase1_fraction, phase2_features, opt_func, random_seed, use_cache=False):
        '''
        phase1_regressors - a list of regressors for phase #1
        phase2_regressor - a regressor for phase #2
        '''
        self.phase1_regressors = phase1_regressors
        self.phase2_regressor  = phase2_regressor
        self.phase1_fraction   = phase1_fraction 
        self.phase2_features   = phase2_features
        self.opt_func = opt_func
        self._random_seed = random_seed
        BaseRegressor.__init__(self, name, use_cache)
    
    def _do_run(self, train, test, aux_test, cache_key):
        assert aux_test is None
        phase1_ids, phase2_ids, _ = self._prp_build_data(train, build_fraction=self.phase1_fraction, test_fraction=0, random_seed=self._random_seed)
        print 'Prepared %d,%d ids for phase1,phase2' % (len(phase1_ids), len(phase2_ids))
        phase1_train, phase1_test = train.take(phase1_ids), train.take(phase2_ids)
        train_preds, test_preds = None, None
        phase1_preds1 = {self.phase2_regressor.target: np.array(phase1_test[self.phase2_regressor.target])}
        phase1_preds2 = {self.phase2_regressor.target: np.array(test[self.phase2_regressor.target])}
        for feature in self.phase2_features:
            phase1_preds1[feature] = np.array(phase1_test[feature])
            phase1_preds2[feature] = np.array(test[feature])
        for regr in self.phase1_regressors:
            r_train1, r_test1, r_aux_test1 = regr.run(train=phase1_train, test=phase1_test, aux_test=test, cache_key=cache_key)
            assert not phase1_preds1.has_key(regr.name)
            phase1_preds1[regr.name] = r_test1
            phase1_preds2[regr.name] = r_aux_test1
#         print 'phase1_preds1 (%d): %s' % (len(phase1_preds1), phase1_preds1)
#         print 'phase1_preds2 (%d): %s' % (len(phase1_preds2), phase1_preds2)
        for regr in self.phase1_regressors:
            print ' phase 1: %-10s : gini = %.5f -> %.5f ; test: %.5f' % (regr.name, self.opt_func(phase1_test[regr.target], phase1_preds1[regr.name]), 
                                                                          self.opt_func(phase1_test[self.phase2_regressor.target], phase1_preds1[regr.name]),
                                                                          self.opt_func(test[self.phase2_regressor.target], phase1_preds2[regr.name]))

        phase2_train, phase2_test = pd.DataFrame(phase1_preds1), pd.DataFrame(phase1_preds2)
        r_train2, r_test2, r_aux_test2 = self.phase2_regressor.run(train=phase2_train, test=phase2_test, aux_test=None, cache_key=cache_key)
        train_preds = None
        #TODO: need to save r_train1 from phase1 regressors, feed them as aux_test to phase 2, and put the results like this:
        #train_preds.put(phase1_ids, results of auz_test as above)
        #train_preds.put(phase2_ids, r_train2)

        return train_preds, r_test2, None
        
class MultiRegressor(BaseRegressor):
    def __init__(self, name, regressors, target, pred_trans=None, coeffs=None, use_cache=False, use_linreg=0, opt_func=None):
        self.name = name
        self.regressors = regressors
        self.target = target
        self.use_linreg = use_linreg
        self.opt_func = opt_func
        if pred_trans is None:
            self.pred_trans = [None]*len(self.regressors)
        else:
            self.pred_trans = pred_trans
        if coeffs is None:
            self.coeffs = np.ones(len(self.regressors))
        else:
            self.coeffs = np.array(coeffs)
        self.lr_coeffs = {}
        BaseRegressor.__init__(self, name, use_cache)
    
    def _transform_preds(self, preds, i_regr):
        if (preds is not None) and (self.pred_trans[i_regr] is not None):
            preds = self.pred_trans[i_regr](preds)
        return preds
        
    def _do_run(self, train, test, aux_test, cache_key):
        print 'Running multi-regressor %s on %d,%d,%d train,test,aux_test samples (%d regressors)' % (self.name, len(train), len(test), 0 if aux_test is None else len(aux_test), len(self.regressors))
        train_preds, test_preds, aux_test_preds = None, None, None
        train_weights, test_weights, aux_test_weights = None, None, None
        regr_train_preds, regr_test_preds = {}, {}
        for i_regr,(regr,coeff) in enumerate(zip(self.regressors, self.coeffs)):
            r_train, r_test, r_aux_test = regr.run(train, test, aux_test, cache_key)
#             print '    targets:     %s' % ', '.join(['%.3f'%v for v in train['Target'][:10]])
#             print '%-12s: test preds: %s' % (regr.name, ', '.join(['%.4f'%v for v in r_test[:10]]))
            gc.collect()
            if self.opt_func is not None:
                print ' Regressor %-12s: score = %.5f' % (regr.name, self.opt_func(test[self.target], r_test))
            r_train_weights, r_test_weights, r_aux_test_weights = regr.get_weights()
#             r_train    = self._transform_preds(r_train   , i_regr)
            r_test     = self._transform_preds(r_test    , i_regr)
            r_aux_test = self._transform_preds(r_aux_test, i_regr)
#             print ' -> train preds: %s' % ', '.join(['%.3f'%v for v in r_train[:10]])
#             if r_train_weights is None:
#                 r_train_weights = np.ones(len(r_train))
#             else:
#                 r_train_weights = r_train_weights / np.max(r_train_weights)
            if r_test_weights is None:
                r_test_weights = np.ones(len(r_test))
            else:
                r_test_weights = r_test_weights / np.max(r_test_weights)
            if aux_test is not None:
                if r_aux_test_weights is None:
                    r_aux_test_weights = np.ones(len(r_aux_test))
                else:
                    r_aux_test_weights = r_aux_test_weights / np.max(r_aux_test_weights)
            #print 'regressor test_weights: min=%.3f , mean=%.5f , max=%.3f' % (np.min(r_test_weights), np.mean(r_test_weights), np.max(r_test_weights))
#             r_train_total = coeff * r_train_weights * r_train 
            r_test_total  = coeff * r_test_weights  * r_test
            if aux_test is not None:
                r_aux_test_total = coeff * r_aux_test_weights  * r_aux_test
            else:
                r_aux_test_total = None
            if test_preds is None:
#                 train_preds = r_train_total
                test_preds  = r_test_total
                aux_test_preds = r_aux_test_total
#                 train_weights = coeff * r_train_weights
                test_weights  = coeff * r_test_weights
                if aux_test is not None:
                    aux_test_weights = coeff * r_aux_test_weights
            else: 
#                 train_preds += r_train_total
                test_preds  += r_test_total
#                 train_weights += coeff * r_train_weights
                test_weights  += coeff * r_test_weights
                if aux_test is not None:
                    aux_test_preds += r_aux_test_total
                    aux_test_weights += coeff * r_aux_test_weights
#             regr_train_preds[regr.name] = r_train_weights * r_train
            regr_test_preds [regr.name] = r_test_weights  * r_test

#         print 'total test_weights: min=%.3f , mean=%.5f , max=%.3f' % (np.min(test_weights), np.mean(test_weights), np.max(test_weights))
#         train_preds = train_preds / train_weights
        test_preds  = test_preds  / test_weights
        if aux_test is not None:
            aux_test_preds = aux_test_preds / aux_test_weights

        if self.use_linreg > 0:
            lr = linear_model.LinearRegression(normalize=False)
#             lr = linear_model.LarsCV(fit_intercept=True, verbose=False, max_iter=500, normalize=False, precompute='auto', cv=None, max_n_alphas=1000)
            regr_train_df, regr_test_df = pd.DataFrame(regr_train_preds), pd.DataFrame(regr_test_preds)
            regrs = sorted(regr_test_preds.keys())
            assert regrs==sorted(regr_train_preds.keys())
            found = False
            while not found:
                lr.fit(regr_test_df[regrs], np.asarray(test[self.target]))
                min_coeff = np.min(lr.coef_)
                if ((self.use_linreg == 1) and (min_coeff <= 0)) or (min_coeff < 0.00001*np.max(lr.coef_)):
                    drop_regr = regrs[np.argmin(lr.coef_)]
                    #print ' dropping %-10s (coeff = %.5f)' % (drop_regr, min_coeff)
                    regrs = [regr for regr in regrs if regr!=drop_regr] 
                else:
#                     print ' -> found!'
                    found = True
            for regr,coef in zip(regrs,lr.coef_):
                if self.lr_coeffs.has_key(regr):
                    self.lr_coeffs[regr].append(coef)
                else:
                    self.lr_coeffs[regr] = [coef]
            print 'Given  coeffs: %s' % ', '.join(['%7.3f'%c for c in self.coeffs])
            print 'Given  regrs : %s' % ', '.join(['%7s'%r.name for r in self.regressors])
            print 'LinReg coeffs: %s' % ', '.join(['%7.3f'%c for c in lr.coef_])
            print 'LinReg regrs : %s' % ', '.join(['%7s'%r for r in regrs])
            regr_test_out = lr.predict(regr_test_df[regrs])
#             for name,preds in sorted(regr_test_preds.iteritems()):
#                 print '%-10s: %s' % (name, ', '.join(['%.3f'%v for v in preds[:10]]))
            print 'Target    : %s' % (', '.join(['%.3f'%v for v in test[self.target][:10]]))
            print 'Test preds: %s' % (', '.join(['%.3f'%v for v in test_preds[:10]]))
            print 'LR preds  : %s' % (', '.join(['%.3f'%v for v in regr_test_out[:10]]))
#             print 'Test preds gini = %.5f' % opt_func(test[self.target], test_preds)
#             print 'LR   preds gini = %.5f' % opt_func(test[self.target], regr_test_out)
            test_preds = regr_test_out
        
#         print 'test_preds: %s' % ', '.join(['%.4f'%x for x in test_preds[:10]])
        return train_preds, test_preds, aux_test_preds

        
#################################################################################################
# FeatureBuilder
#################################################################################################

class FeatureBuilder(object):
    def __init__(self, train, test, aux_test, build_ids, target_col):
        self.train = train
        self.test = test
        self.aux_test = aux_test
        self.build_ids = build_ids
#         print 'Build ids: %s %s ...' % (build_ids[0], build_ids[1])
#         print ' train[build_ids[0]]: %s' % train.take([build_ids[0]])
        self.target_col = target_col
        
    def build_means(self, features):
        return convert_values(self.train, self.test, aux_test=self.aux_test, features=features, 
                              target_col=self.target_col, method='mean', min_num_vals=2, replace_flag=False, build_ids=self.build_ids)

    def build_linreg(self, features):
        built_features = []
        for feature in features:
            cols = feature.split(',')
            print 'linear feature of %d columns: %s' % (len(cols), cols)
            built_features.append(convert_values(self.train, self.test, aux_test=self.aux_test, features=cols,
                                                 target_col=self.target_col, method='linreg', min_num_vals=1, replace_flag=False, build_ids=self.build_ids))
        return built_features
        
        
#################################################################################################
# Trainer
#################################################################################################

class Trainer(object):
    
    def __init__(self, dataf, target_col, opt_func, groupby_col):
        self.dataf = dataf
        self.target = target_col
        assert self.target in self.dataf.columns
        self.opt_func = opt_func
        self.groupby_col = groupby_col
        assert self.groupby_col in self.dataf.columns

    def prp_train_test_data(self, test_fraction, build_fraction, max_train=None, max_test=None, max_build=None, random_seed=12345, shuffle_flag=True):
        '''
        Partition data into training and test sets
        test_fraction - fraction of data to serve as test set 
        '''
        np.random.seed(random_seed)
        groupby_vals = np.unique(self.dataf[self.groupby_col])
        shuffled_ids = range(len(groupby_vals))
        if shuffle_flag:
            # Choose random training & test sets
            np.random.shuffle(shuffled_ids)
#         print '%d shuffled_ids: %s' % (len(shuffled_ids), shuffled_ids[:7])
        n_test  = int(test_fraction*len(shuffled_ids))
        n_build = int(build_fraction*len(shuffled_ids))
        test_vals  = np.take(groupby_vals, shuffled_ids[:n_test])
        build_vals = np.take(groupby_vals, shuffled_ids[n_test:(n_test+n_build)])
        train_vals = np.take(groupby_vals, shuffled_ids[(n_test+n_build):])
        test_vals  = set(test_vals)
        build_vals = set(build_vals) 
        train_vals = set(train_vals) 
        self.test_ids = [id for id in range(len(self.dataf)) if self.dataf.irow(id)[self.groupby_col] in test_vals] #shuffled_ids[:n_test]
        if max_test is not None:
            self.test_ids = self.test_ids[:max_test]
        self.build_ids = [id for id in range(len(self.dataf)) if self.dataf.irow(id)[self.groupby_col] in build_vals] #shuffled_ids[n_test:(n_test+n_build)]
        if max_build is not None:
            self.build_ids = self.build_ids[:max_build]
        self.train_ids = [id for id in range(len(self.dataf)) if self.dataf.irow(id)[self.groupby_col] in train_vals] #shuffled_ids[(n_test+n_build):]
        if max_train is not None:
            self.train_ids = self.train_ids[:max_train]
        print 'There are %d/%d/%d build/train/test IDs' % (len(self.build_ids), len(self.train_ids), len(self.test_ids))
#         print 'build: %s' % sorted(self.build_ids)[:12]
#         print 'train: %s' % sorted(self.train_ids)[:12]
#         print 'test : %s' % sorted(self.test_ids )[:12]

    def fit_model(self, regressor, dataf=None, train_ids=None, test_ids=None, target=None,   
                  print_flag=False, full_output=False, test_dataf=None, cache_key=None, **kwargs):
        '''
        Fit model to training data and predict test samples
        '''
        if dataf is None:
            dataf = self.dataf
        if target is None:
            target = self.target
        train_dataf = dataf if train_ids is None else dataf.take(train_ids)
        if test_dataf is None:
            test_dataf = dataf if test_ids is None else dataf.take(test_ids)
        if print_flag:
            print 'Fitting model on %d/%d train/test samples' % (len(train_dataf), len(test_dataf))
            sys.stdout.flush()

        # Fit model to training data and get predictions for test data
#         print 'train_dataf: %s' % train_dataf
        train_preds, test_preds, aux_test_preds = regressor.run(train_dataf, test_dataf, cache_key=cache_key)
#         clf.fit(train_dataf[features], train_dataf[target])
#         test_preds = clf.predict(test_dataf[features])
#         print 'preds = %s' % test_preds
        if target in test_dataf.columns:
            score = self.opt_func(np.asarray(test_dataf[target]), np.asarray(test_preds))
            if print_flag:
                print ' model score is %.5f' % score
        else:
            score = -1

        train_dataf, test_dataf = None, None
        gc.collect() # save memory?
                
        if full_output:
            return score, test_preds
        else:
            return score

    def forward(self, method='knn', dataf=None, train_ids=None, test_ids=None, features=None, max_num_features=200, fwd_opt_func=None,
                cv_fold=5, min_cv_improvement=0.001, direction='forward', start_features=None, print_flag=True, max_rows=None, **kwargs):
        '''
        Run forward/backward algorithm to build a good model
        Returns the set of features found, and the model's CV score
        '''
        assert direction in ['forward','backward']
        if dataf is None:
            dataf = self.dataf
        if features is None:
            features = self.features
        if len(features)==0:
            return [], -1
        if train_ids is None:
            train_ids = self.train_ids
        if test_ids is None:
            test_ids = self.test_ids        
        if max_rows is not None:
            dataf = dataf[:max_rows]
            train_ids = [i for i in train_ids if i<max_rows]
            test_ids  = [i for i in test_ids  if i<max_rows]
        if start_features is not None:
            best_features = start_features
        else:
            if direction=='forward':
                best_features = []
                best_cv_score = 999999.9
            else: # 'backward'
                best_features = self.features # use all features as starting point
        dir_sign = 1 if direction=='forward' else -1
        print 'Running %s alg on %d,%d train,test samples' % (direction, len(train_ids), len(test_ids))
        print 'Initial features: %s' % str(best_features)
#old:        best_cv_score = self.cross_validation(method, dataf[best_features+[self.target]], train_ids, kfold=cv_fold, **kwargs)
        cv_opt_score, cv_preds, cv_test_ids = self.cross_validation(method, dataf[best_features+[self.target]], train_ids, kfold=cv_fold, full_output=True, **kwargs)
        if fwd_opt_func is None:
            best_cv_score = cv_opt_score
        else:
            cv_all_preds = np.ones(len(train_ids)) * (-1)
            np.put(cv_all_preds, cv_test_ids, cv_preds)
            assert np.all(cv_all_preds != -1)
            best_cv_score = fwd_opt_func(cv_opt_score, cv_all_preds)
        print 'Initial CV score: %.5f' % best_cv_score
        while (direction=='forward'  and len(best_features)<np.min([max_num_features, len(features)])) or\
              (direction=='backward' and len(best_features)>0):
            cv_scores = {}
            for feature in sorted(features):
                if print_flag:
                    print '   Trying to %s feature %s' % ('add' if direction=='forward' else 'remove', feature)
                if (direction=='forward' ) and (feature in best_features): continue
                if (direction=='backward') and (feature not in best_features): continue
                if direction=='forward':
                    new_features = best_features + [feature]
                else: # 'backward'
                    new_features = [f for f in best_features if f!=feature]
#old:                 cv_scores[feature] = self.cross_validation(method, dataf[new_features+[self.target]], train_ids, kfold=cv_fold, **kwargs)
                cv_opt_score, cv_preds, cv_test_ids = self.cross_validation(method, dataf[new_features+[self.target]], train_ids, kfold=cv_fold, full_output=True, **kwargs)
                print ' CV opt func score = %.4f' % cv_opt_score
                if fwd_opt_func is None:
                    cv_scores[feature] = cv_opt_score
                else:
                    cv_all_preds = np.ones(len(train_ids)) * (-1)
                    np.put(cv_all_preds, cv_test_ids, cv_preds)
                    assert np.all(cv_all_preds != -1)
                    cv_scores[feature] = fwd_opt_func(cv_opt_score, cv_all_preds)
                if print_flag:
                    print '   CV score for feature %s = %.5f' % (feature, cv_scores[feature])
                    
            cv_feature = sorted(cv_scores, key=lambda k: cv_scores[k])[-1]
            cv_score = cv_scores[cv_feature]
            if print_flag:
                print 'scores: %s ... %s' % (sorted(cv_scores.keys())[:3], sorted(cv_scores.keys())[-3:]) 
                print '-> cv_feature = %s , score =%.4f' % (cv_feature, cv_score)
#             print 'cv_score = %f' % cv_score
#             print 'best_cv_score + min_cv_improvement = %f' % (best_cv_score + min_cv_improvement)
            if cv_score > best_cv_score + min_cv_improvement: #cv_score < best_cv_score - dir_sign * min_cv_improvement:
#                 print ' ************* improved ***********'
                if print_flag:
                    print '---> %s %s -> best score = %.4f' % ('adding' if direction=='forward' else 'removing', cv_feature, cv_score)
                if direction=='forward':
                    best_features += [cv_feature]
                else: # 'backward'
                    best_features = [f for f in best_features if f!=cv_feature]
                best_cv_score = cv_score
#                 print '======> best_cv_score = %f' % best_cv_score
                if print_flag:
                    print ' Current model has %d features: %s' % (len(best_features), best_features)
            else:
                break
        if print_flag:
            print '-----> best features (%d, score %.4f): %s' % (len(best_features), best_cv_score, best_features)
            sys.stdout.flush()
        return best_features, best_cv_score

    def cross_validation(self, method, dataf, train_ids, kfold=5, num_folds=None, full_output=False, random_seed=374157, target=None, **kwargs):
        '''
        Run cross-validation to compute score per k-fold of data
        Returns the CV score, which is the average score of all k-folds
        '''
        np.random.seed(random_seed)
        train_ids = np.array(train_ids)
        val_to_ids = {}
        for tid in train_ids:
            val = self.dataf.irow(tid)[self.groupby_col]
            if val_to_ids.has_key(val):
                val_to_ids[val].append(tid)
            else:
                val_to_ids[val] = [tid]
#         np.random.shuffle(train_ids)
        train_groupby_vals = np.array(val_to_ids.keys())
        print '%d train_groupby_vals: %s' % (len(train_groupby_vals), train_groupby_vals[:12])
        np.random.shuffle(train_groupby_vals)
        print '%d train_groupby_vals: %s' % (len(train_groupby_vals), train_groupby_vals[:12])
        ldata = len(train_groupby_vals) #len(train_ids)
        assert ldata>=kfold and kfold>1
        print 'CV - %d-fold (%d train groupby vals, seed %s)' % (kfold, ldata, random_seed)
        sys.stdout.flush()
        cv_scores, cv_preds, cv_test_ids = [], None, None
        for ifold in range(kfold):
            fold_from, fold_to = int(ldata/(kfold+0.0)*ifold), int(ldata/(kfold+0.0)*(ifold+1)) 
#             inds_cv_train = np.take(train_ids, range(0, fold_from) + range(fold_to, ldata))
#             inds_cv_test  = np.take(train_ids, range(fold_from, fold_to))
            inds_cv_train = np.concatenate([val_to_ids[train_groupby_vals[i]] for i in range(0, fold_from)+range(fold_to, ldata)]) 
            inds_cv_test  = np.concatenate([val_to_ids[train_groupby_vals[i]] for i in range(fold_from,fold_to)]) 
            print 'CV training on %s...' % inds_cv_train[:12]
            print 'CV testing  on %s...' % inds_cv_test [:12]
            cv_score, preds = self.fit_model(method, dataf, train_ids=inds_cv_train, test_ids=inds_cv_test, target=target,  
                                             full_output=True, cache_key=(myint32(hash(tuple(train_ids))),random_seed,kfold,ifold), **kwargs)
            cv_scores.append(cv_score)
            print ' CV fold #%d/%d: score = %.5f' % (ifold, kfold, cv_score)
            sys.stdout.flush()
            if cv_preds is None:
                cv_preds = preds
                cv_test_ids = inds_cv_test
            else:
                cv_preds = np.concatenate([cv_preds, preds])
                cv_test_ids = np.concatenate([cv_test_ids, inds_cv_test])
            if (num_folds is not None) and (len(cv_scores)>=num_folds): break
        if full_output:
            return np.mean(cv_scores), cv_preds, cv_test_ids
        else:
            return np.mean(cv_scores)


def myint32(x):
    '''
    Helper function to convert long to int32 (we use it to convert hash() to int32, since on some machines it gives long) 
    '''
    x = (x & 0xFFFFFFFF)
    if x>0xFFFFFFFF:
        raise OverflowError
    if x>0x7FFFFFFF:
        x=int(0x100000000-x)
        if x<2147483648:
#             print '-x'
            return -x
        else:
            return -2147483648
    return x



#################################################################################################
# WordNormalizer - "normalize" words using NLTK hypernyms
#################################################################################################
        
class WordNormalizer(object):
    def __init__(self, hyp_level=5, stemmer=None, add_flag=False):
        self.hyp_level = hyp_level
        self.stemmer = stemmer
        self.add_flag = add_flag
        self._cache = {}
        
    def _find_hypernym(self, word, tag):
        assert tag in [wn.NOUN, wn.VERB, wn.ADJ, wn.ADV]
        hs = wn.synsets(word, pos=tag)
        hyps = []
        while len(hs) > 0:
            hyps.append(hs[0])
            hs = hyps[-1].hypernyms()
        if len(hyps)==0:
            if self.stemmer is None:
                return word
            else:
                return self.stemmer.stem(word)
        elif len(hyps) < self.hyp_level:
            return hyps[0].name()
        else:
            return hyps[-self.hyp_level].name()
        
    def normalize(self, word, tag):
        if self.stemmer is None:
            sword = word 
        else:
            sword = self.stemmer.stem(word)
        if tag is None:
            return sword
        wt = (word, tag)
        if not self._cache.has_key(wt):
            self._cache[wt] = self._find_hypernym(word, tag)
        if self.add_flag:
            return [sword, self._cache[wt]]
        else:
            return self._cache[wt]
        

#################################################################################################
# get_regressor() returns our final model - a regressor that combines multiple features
# using several Ensemble models
#################################################################################################

def get_regressor(target_col, coeffs=None):
    '''
    Return a Regressor - our model!
    '''
    # GB parameters
    gb_params0 = {'n_estimators':220, 'max_depth':5, 'min_samples_leaf':7 , 'subsample':0.9 , 'max_features':0.82, 'learning_rate':0.03}  
    gb_params1 = {'n_estimators':70 , 'max_depth':6, 'min_samples_leaf':7 , 'subsample':0.95, 'max_features':0.9 , 'learning_rate':0.08} 
    gb_params2 = {'n_estimators':95 , 'max_depth':6, 'min_samples_leaf':5 , 'subsample':0.9 , 'max_features':0.93, 'learning_rate':0.05} 
    gb_params3 = {'n_estimators':80 , 'max_depth':5, 'min_samples_leaf':20, 'subsample':0.91, 'max_features':0.92, 'learning_rate':0.07} 
    gb_params4 = {'n_estimators':50 , 'max_depth':5, 'min_samples_leaf':8 , 'subsample':0.97, 'max_features':0.85, 'learning_rate':0.1 , 'loss':'exponential'}
    gb_params5 = {'n_estimators':150, 'max_depth':7, 'min_samples_leaf':20, 'subsample':0.95, 'max_features':0.78, 'learning_rate':0.04}

    # Features for the GB models
    features0_02  = ['tr_st_qz.1_over', 'st_qz_ai.1_over', #'st_qz_saylor_ck-h.1_over', 
                     'ans_in_qst_stem', #'st2_qz_oer.1_under', 
                     'st_qz.1_over', #'ans_words_stem_correct', 
                     'wk-pn_sw-pn_wb.a1_vs_a2.1_over']
    features0_02 += ['st2_qz_oer_ck-t.triplets.1_under', 
                     'ans_length_ratio', 'ck-hp_saylor.triplets.1_over']
    features0_02 += ['lucene.2', 'lucene.4']
    features0_02 += ['is_numerical']
    features0_02 += ['tr_st_qz.2_over', 'st2_qz_wk-pn_oer_ck-h.pairs.1_over', 'wk3_sw3.1_under', 'wk-pn_sw-pn_wb.a1_vs_a2.1_over']
    features0_05  = ['tr_st_qz.1_over', 'st_qz_ai.1_over', #'st_qz_saylor_ck-h.1_over', 
                     'ans_in_qst_stem', 'ans_num_words', #'ans_words_stem_pval', 
                     'st2_qz_wk-pn_oer_ck-h.pairs.1_under', 
                     #'st_qz.pairs.1_over', 
                     'qz_ck-ts.1_over', 'ans_length_ratio', 'wk-pn_sw-pn_wb.a1_vs_a2.1_under',
                     'ans_stem_pval', 'st2_qz_oer_ck-t.triplets.1_over'
                     ] #, 'wk3_wb.1_under'] 
    features0_05 += ['ans_words_stem_zscore','ans_words_stem_pairs_correct','st_qz.triplets13.1_over']
    features0_05 += ['lucene.5','lucene.6','lucene.7'] 
    features0_09  = ['tr_st_qz.1_over', 'st_qz_ai.1_under', 'st2_qz_oer_ck-hp.1_over', 'ans_words_stem_pval', 'a_num_words', 'st_qz.pairs2.1_over', 
                     'st2_qz_oer_ck-t.triplets.1_under', 
                     'ans_in_qst_stem', 'sw-pn_ss_ck-t_ai.1_under', 'tr_st_qz.2_over',
                     'ans_length_ratio', 'st2_qz_wk-pn_oer_ck-h.pairs.1_under'] + ['ans_words_stem_zscore','ans_qst_words_stem_count']
    features0_09 += ['lucene.1','lucene.2','lucene.3','lucene.4']                 

    features1_01  = ['st_qz.1_over', 'ans_words_stem_correct', 'ans_in_qst_stem', 'st_qz_ai.1_under', 'ck-hp_saylor.triplets.1_over', 
                     'sw-pn_qz.1_over', 'sw-pn_ss_ck-t_ai.1_under', 'st2_qz_oer_ck-t.triplets.1_under'] 
    features1_01 += ['st2_qz_wk-pn_oer_ck-h.pairs.1_under', 
                     'st2_qz_oer_ck-hp.1_over', 'a_num_words'] 
    features1_01 += ['ans_words_stem_zscore', 'ans_words_stem_correct_nonorm']
    features1_01 += ['lucene.1','lucene.2','lucene.4']
    features1_05  = ['st_qz.1_over', 'st2_qz_wk-pn_oer_ck-h.pairs.1_under', 
                     'a_num_words', 'ans_words_stem_correct', 'st_qz_ai.1_over', 
                     'ans_in_qst_stem', 'st2_qz_oer_ck-t.triplets.1_over', 
                     'ans_correct', 'wk-pn_sw-pn_wb.a1_vs_a2.1_over', 'is_none'] 
    features1_05 += ['st_qz.triplets13.1_over'] 
    features1_05 += ['ans_qst_words_stem_correct'] 
    features1_05 += ['lucene.2','lucene.4']
    features1_09  = ['st_qz.1_over', 'st_qz_ai.1_under', 'st2_qz_oer_ck-t.triplets.1_over', 
                     'ans_words_stem_pval', 'a_num_words', 'tr_st_qz.2_over', 
                     'ans_in_qst_stem', 'ans_length_ratio', 'st_qz_saylor_ck-t.a1_vs_a2.1_over', 'ans_words_stem_correct', 'ck-hp_saylor_oer.triplets.1_over'] 
    features1_09 += ['qz_ck-ts.1_over', 'st2_qz_wk-pn_oer_ck-h.pairs.1_under', 
                     'st_qz.pairs2.1_over',
                     'ans_words_stem_zscore'] + ['ans_words_pairs_zscore'] + ['ans_corr_vs_qst_count']
    features1_09 += ['lucene.1','lucene.2','lucene.3','lucene.4']              
    
    features2_01  = ['st_qz.1_over', 'ans_words_stem_correct', 'wk3_sw3.1_under', 'ck-hp_saylor.triplets.1_over', 'st_qz_ai.1_under', 'ans_stem_pval', 
                     'ans_in_qst_stem', 'is_none'] 
    features2_01 += ['st2_qz_wk-pn_oer_ck-h.pairs.1_under', 
                     'sw2_ck-ts.1_over', 'sw-pn_ss_ck-t_ai.1_under', 'tr_st_qz.2_under'] 
    features2_01 += ['ans_words_pairs_zscore']
    features2_01 += ['lucene.1','lucene.4']            
    features2_05  = ['st_qz.1_over', 'st2_qz_wk-pn_oer_ck-h.pairs.1_under', 
                     'a_num_words', 'ans_words_stem_correct', 'st_qz_ai.1_over', 
                     'ans_in_qst', 'st2_qz_oer_ck-t.triplets.1_over', 
                     'tr_st_qz.2_under'] 
    features2_05 += ['wk-pn_sw-pn.1_over'] 
    features2_05 += ['ans_words_stem_zscore', 'ans_qst_words_stem_count']
    features2_05 += ['lucene.2','lucene.4']
    features2_09  = ['st_qz.1_over', 'st_qz_ai.1_under', 'ans_words_stem_pval', 'st2_qz_oer_ck-t.triplets.1_over', 
                     'a_num_words', 'tr_st_qz.2_over', 
                     'ans_length_ratio', 'ans_in_qst_stem', 'st2_qz_oer_ck-t.triplets.1_under', 
                     'st_qz_saylor_ck-t.a1_vs_a2.1_over', 
                     'st_qz.pairs2.1_over', 'qz_ck-ts.1_over', 'sw2_ck-ts.1_over'] 
    features2_09 += ['ans_words_pairs_zscore', 'ans_words_stem_pairs_correct'] 
    features2_09 += ['st_qz.Z']
    features2_09 += ['lucene.2','lucene.4']

    features3a_09  = ['st_qz_ai.1_over', 'st2_qz_oer_ck-t.triplets.1_over', 
                      'tr_st_qz.1_under', 'a_num_words', 'ans_words_stem_correct', 'st_qz.pairs2.1_over',
                      'st2_qz_wk-pn_oer_ck-h.pairs.1_under', 
                      'ans_in_qst'] #, 'st2_sw-pn_saylor_ck-t.1_over']
    features3a_09 += ['tr_st_qz.2_under', 'st_qz_saylor_ck-t.a1_vs_a2.1_under', 'st_qz.triplets13.1_over']
    features3a_09 += ['ans_qst_words_stem_count', 'ans_qst_words_stem_correct']
    features3a_09 += ['lucene.2','lucene.4']
    features3b_09  = ['st2_qz_wk-pn_oer_ck-h.pairs.1_over', 
                      'st_qz.1_over', 'st_qz_ai.1_under', 'ans_words_stem_pval', 'ans_num_words', 'tr_st_qz.2_over',
                      'ans_in_qst_stem', 'ans_length_ratio', 'st_qz_saylor_ck-t.a1_vs_a2.1_over'] 
    features3b_09 += ['ck-hp_saylor_oer.triplets.1_under']
    features3b_09 += ['ans_words_stem_pairs_correct', 'is_both','is_none','is_all','is_numerical']
    features3b_09 += ['lucene.1','lucene.2','lucene.3']
    features3m = ['lucene.1','lucene.2','lucene.3','lucene.4','lucene.5','lucene.6','lucene.7',
                  'ans_words_stem_pairs_correct', 'ans_in_qst_stem', 'ans_words_stem_correct', 
                  'ans_words_stem_pval','ans_words_stem_zscore', 'ans_length_ratio','ans_num_words',
                  'q_num_words', 'a_num_words', 
                  'st_qz.triplets13.1_over', 'st2_qz_oer_ck-t.triplets.1_over']
    
    features4_02 = ['st_qz.1_over', 'st2_qz_wk-pn_oer_ck-h.pairs.1_under', 
                    'ans_words_stem_correct', 'a_num_words', 'ans_in_qst_stem', 
                    'st_qz_ai.1_over', 'ck-hp_saylor.triplets.1_under']
    features4_02 += ['lucene.2','lucene.4']
    features4_02 += ['lucene.6']    
    features4_05 = ['st_qz.triplets13.1_over', 'st_qz.triplets13.1_under', 'st_qz_ai.1_under', #'sw-pn_qz_ck-h_saylor.1_over', 
                    'ans_words_stem_correct', 'st2_qz_wk-pn_oer_ck-h.pairs.1_under', 
                    'a_num_words', 
                    'st2_qz_oer_ck-t.triplets.1_over', #'wk3_wb.1_under', 
                    'ans_in_qst_stem'] #, 'sw_wb.1_under']
    features4_05 += ['lucene.1','lucene.3']
    features4_05 += ['lucene.5','lucene.7'] 
    features4_08 = ['st_qz.1_over', 'st_qz_ai.1_under', 'st2_qz_wk-pn_oer_ck-h.pairs.1_under', 
                    'ans_words_stem_pval', 'a_num_words', 
                    'st2_qz_oer_ck-t.triplets.1_over', #'wk3_wb.1_under', 
                    'ans_in_qst_stem', 'st_qz_saylor_ck-t.a1_vs_a2.1_over'] #, 'st_qz.pairs.1_over']
    features4_08 += ['lucene.1','lucene.2','lucene.4'] + ['is_both', 'is_none', 'is_all']
    features4_095 = ['st_qz.1_over', 'st_qz_ai.1_under', 'ans_words_pairs_zscore', 'st2_qz_wk-pn_oer_ck-h.pairs.1_under', 
                     'st2_qz_oer_ck-t.triplets.1_over', 
                     'a_num_words', 'st_qz_ai.1_over', 'ans_words_stem_correct', 'ans_in_qst_stem', 'st_qz_saylor_ck-t.a1_vs_a2.1_under', 'tr_st_qz.2_over'] # CV score 0.8010
    features4_095 += ['lucene.1','lucene.3','lucene.4']
    features4_095 += ['lucene.5','lucene.6'] 

    features5a = ['is_both', 'is_none', 'is_all', 'is_numerical', 'BCDA', 'q_not', # 'is_special', 
                  'a_num_words', 'q_num_words', 'ans_in_qst_stem', 'ans_in_ans_stem',
                  'ans_count', 'ans_words_stem_count', 'ans_qst_words_stem_count',
                  'ans_words_stem_correct', 'ans_qst_words_stem_correct', 'ans_words_stem_pairs_correct', 
                  'ans_stem_pval', 'ans_words_pairs_zscore', 'ans_words_stem_pval', 'ans_words_stem_zscore',
                  'ans_length_ratio', 'ans_num_words',
                  'st_qz.1_under',
                  'ck-hp_saylor.triplets.1_under', 'qz_ck-ts.1_under', 'st2_qz_oer_ck-hp.1_over',
                  'st2_qz_wk-pn_oer_ck-h.pairs.1_over', 
                  'st_qz.pairs2.1_under',
                  'st_qz_saylor_ck-t.a1_vs_a2.1_under', 'sw-pn_qz.1_over',
                  'sw-pn_ss_ck-t_ai.1_over', 'sw2_ck-ts.1_under', 'wk3_sw3.1_over',
                  'tr_st_qz.1_over',  
                  'wk-pn_sw-pn_wb.a1_vs_a2.1_over', 'st_qz.triplets13.1_over', 'st_qz.Z', 'wk-pn_sw-pn.1_over']
    features5b = ['is_both', 'is_none', 'is_all', 'is_numerical', ##'BCDA', 'q_not', # 'is_special', 
                  #'a_num_words', 'q_num_words', 'ans_in_qst_stem', 'ans_in_ans_stem',
                  'ans_words_stem_count_nonorm', 'ans_words_stem_correct_nonorm', 'ans_corr_vs_qst_count',
                  'ans_count', 'ans_words_stem_count', ##'ans_qst_words_stem_count',
                  'ans_words_stem_correct', ##'ans_qst_words_stem_correct', 'ans_words_stem_pairs_correct', 
                  'ans_stem_pval', 'ans_words_pairs_zscore', 'ans_words_stem_pval', 'ans_words_stem_zscore',
                  'ans_length', 'ans_num_words',
                  'st_qz.1_over',
                  'ck-hp_saylor.triplets.1_over', 'qz_ck-ts.1_over', 'st2_qz_oer_ck-hp.1_over', 'st2_qz_oer_ck-hp.1_under',
                  'st2_qz_wk-pn_oer_ck-h.pairs.1_over', 'st2_qz_wk-pn_oer_ck-h.pairs.1_under', 
                  'st_qz.pairs2.1_under',
                  'st_qz_saylor_ck-t.a1_vs_a2.1_under', 'sw-pn_qz.1_over', 'sw-pn_qz.1_under',
                  'sw-pn_ss_ck-t_ai.1_over', 'sw2_ck-ts.1_over', 'wk3_sw3.1_under',
                  ##'tr_st_qz.1_under',  
                  'wk-pn_sw-pn_wb.a1_vs_a2.1_under', 'st_qz.triplets13.1_under', ##'st_qz.Z', 
                  'wk-pn_sw-pn.1_over']
    features5c = ['is_both', 'is_none', 'is_all', 'is_numerical', 'BCDA', 'q_not', # 'is_special', 
                  'a_num_words', 'q_num_words', 'ans_in_qst_stem', 'ans_in_ans_stem',
                  'ans_count', 'ans_words_stem_count', 'ans_qst_words_stem_count',
                  'ans_words_stem_correct', 'ans_qst_words_stem_correct', 'ans_words_stem_pairs_correct', 
                  'ans_stem_pval', #'ans_words_pairs_zscore', 
                  'ans_words_stem_pval', #'ans_words_stem_zscore',
                  'ans_length', 'ans_length_ratio', #'ans_num_words',
                  ##'st_qz.1_over', 'st_qz.1_under',
                  'st_qz_ai.1_over', 'st_qz_ai.1_under',
                  'ck-hp_saylor.triplets.1_over', 'ck-hp_saylor_oer.triplets.1_under', 
                  'qz_ck-ts.1_over', 'qz_ck-ts.1_under', 'st2_qz_oer_ck-hp.1_over', 
                  'st2_qz_oer_ck-t.triplets.1_over', 'st2_qz_oer_ck-t.triplets.1_under',
                  'st2_qz_wk-pn_oer_ck-h.pairs.1_over', 
                  'st_qz.pairs2.1_over', 'st_qz.pairs2.1_under',
                  'st_qz_saylor_ck-t.a1_vs_a2.1_over', 'sw-pn_qz.1_over', 'sw-pn_qz.1_under',
                  'sw-pn_ss_ck-t_ai.1_over', 'sw2_ck-ts.1_under', 'tr_st_qz.2_over', 'wk3_sw3.1_over',
                  ##'tr_st_qz.1_over',  
                  'wk-pn_sw-pn_wb.a1_vs_a2.1_over', 'wk-pn_sw-pn_wb.a1_vs_a2.1_under', 
                  #'st_qz.triplets13.1_over', 
                  'st_qz.Z', 'wk-pn_sw-pn.1_over', 'wk-pn_sw-pn.1_under']
    features5c += ['lucene.1','lucene.2']
    features5d = [##'is_both', 'is_none', 'is_all', 'is_numerical', 'BCDA', 'q_not', # 'is_special', 
                  #'a_num_words', 'q_num_words', 'ans_in_qst_stem', 'ans_in_ans_stem',
                  'ans_count', 'ans_words_stem_count', 'ans_words_stem_count_nonorm', 'ans_qst_words_stem_count',
                  'ans_correct', 'ans_words_stem_correct', 'ans_words_stem_correct_nonorm', 'ans_qst_words_stem_correct', 'ans_words_stem_pairs_correct',
                  'ans_pval', 'ans_stem_pval', 'ans_words_pairs_zscore', 
                  'ans_words_stem_pval', 'ans_words_stem_zscore', 'ans_corr_vs_qst_count', 
                  'ans_length', 'ans_length_ratio', 'ans_num_words',
                  'st_qz.1_over', 'st_qz_ai.1_over', 'st_qz_ai.1_under',
                  'ck-hp_saylor.triplets.1_over',  
                  'qz_ck-ts.1_under', 'st2_qz_oer_ck-hp.1_under', 
                  'st2_qz_oer_ck-t.triplets.1_over',
                  'st2_qz_wk-pn_oer_ck-h.pairs.1_under', 
                  'st_qz.pairs2.1_over', 
                  'st_qz_saylor_ck-t.a1_vs_a2.1_under', 'sw-pn_qz.1_over', 
                  'sw-pn_ss_ck-t_ai.1_under', 'sw2_ck-ts.1_over', 'sw2_ck-ts.1_under',
                  'tr_st_qz.1_over', 'tr_st_qz.2_under', 'wk3_sw3.1_under',
                  'wk-pn_sw-pn_wb.a1_vs_a2.1_over', 'st_qz.triplets13.1_over', 'wk-pn_sw-pn.1_under']
    features5d += ['lucene.3','lucene.4']
    features5e = [##'is_both', 'is_none', 'is_all', 'is_numerical', 'BCDA', 'q_not', # 'is_special', 
                  #'a_num_words', 'q_num_words', 'ans_in_qst_stem', 'ans_in_ans_stem',
                  'ans_count', 'ans_words_stem_count', #'ans_words_stem_count_nonorm', 
                    'ans_qst_words_stem_count',
                  'ans_correct', 'ans_words_stem_correct', #'ans_words_stem_correct_nonorm', 
                    'ans_qst_words_stem_correct', 'ans_words_stem_pairs_correct',
                  'ans_pval', 'ans_stem_pval', 'ans_words_pairs_zscore', 
                    'ans_words_stem_pval', 'ans_words_stem_zscore', 'ans_corr_vs_qst_count', 
                  'ans_length', 'ans_length_ratio', 'ans_num_words',
                  'ck-hp_saylor.triplets.1_over', 'ck-hp_saylor_oer.triplets.1_over',  'qz_ck-ts.1_over',  
                  'st2_qz_oer_ck-hp.1_over', 'st2_qz_oer_ck-t.triplets.1_over', 
                  'st2_qz_wk-pn_oer_ck-h.pairs.1_over', 
                  'st_qz.1_over', 'st_qz.pairs2.1_over', 'st_qz_ai.1_over', 
                  'st_qz_saylor_ck-t.a1_vs_a2.1_over', 'sw-pn_qz.1_over',  
                  'sw-pn_ss_ck-t_ai.1_over', 'sw2_ck-ts.1_over', 'tr_st_qz.1_over', 'tr_st_qz.2_over', 'wk3_sw3.1_over',
                  'wk-pn_sw-pn_wb.a1_vs_a2.1_over', 'st_qz.triplets13.1_over', 'st_qz.Z', 'wk-pn_sw-pn.1_over']
    features5f = ['is_both', 'is_none', 'is_all', 'is_numerical', 'BCDA', 'q_not', 'q____', # 'is_special', 
                  'a_num_words', 'q_num_words', 'ans_in_qst_stem', 'ans_in_ans_stem',
                  #'ans_count', 
                  'ans_words_stem_count', #'ans_words_stem_count_nonorm', 
                  #  'ans_qst_words_stem_count',
                  #'ans_correct', 
                  'ans_words_stem_correct', #'ans_words_stem_correct_nonorm', 
                  #  'ans_qst_words_stem_correct', 
                  'ans_words_stem_pairs_correct',
                  #'ans_pval', 'ans_stem_pval', 
                  'ans_words_pairs_zscore', 'ans_words_stem_pval', 'ans_words_stem_zscore', #'ans_corr_vs_qst_count', 
                  'ans_length', 'ans_length_ratio', 'ans_num_words',
                  'ck-hp_saylor.triplets.1_over', 'ck-hp_saylor_oer.triplets.1_over',  'qz_ck-ts.1_over',  
                  'st2_qz_oer_ck-hp.1_over', 'st2_qz_oer_ck-t.triplets.1_over', 
                  'st2_qz_wk-pn_oer_ck-h.pairs.1_over', 
                  #'st_qz.1_under', 
                  'st_qz.pairs2.1_over', 
                  #'st_qz_ai.1_under', 
                  'st_qz_saylor_ck-t.a1_vs_a2.1_over', 'sw-pn_qz.1_over',  
                  'sw-pn_ss_ck-t_ai.1_over', 'sw2_ck-ts.1_over', 
                  #'tr_st_qz.1_over', 
                  'tr_st_qz.2_over', 'wk3_sw3.1_over',
                  'wk-pn_sw-pn_wb.a1_vs_a2.1_over', 'st_qz.triplets13.1_over', 'st_qz.Z', 'wk-pn_sw-pn.1_over',
                  'lucene.1','lucene.2','lucene.3','lucene.4']
             
    # GB regressors
    use_cache = False

    regressor0_02 = Regressor('GB0_02', ensemble.GradientBoostingClassifier(**gb_params0), 
                              features0_02, target_col, use_cache=use_cache, build_fraction=0.0, z_transform=False, random_seed=99002202) 
    regressor0_05 = Regressor('GB0_05', ensemble.GradientBoostingClassifier(**gb_params0), 
                              features0_05, target_col, use_cache=use_cache, build_fraction=0.0, z_transform=False, random_seed=99002205) 
    regressor0_09 = Regressor('GB0_09', ensemble.GradientBoostingClassifier(**gb_params0), 
                              features0_09, target_col, use_cache=use_cache, build_fraction=0.0, z_transform=False, random_seed=99002209)

    regressor1_01 = Regressor('GB1_01', ensemble.GradientBoostingClassifier(**gb_params1), 
                              features1_01, target_col, use_cache=use_cache, build_fraction=0.0, z_transform=False, random_seed=99000201) 
    regressor1_05 = Regressor('GB1_05', ensemble.GradientBoostingClassifier(**gb_params1), 
                              features1_05, target_col, use_cache=use_cache, build_fraction=0.0, z_transform=False, random_seed=99000205)
    regressor1_09 = Regressor('GB1_09', ensemble.GradientBoostingClassifier(**gb_params1), 
                              features1_09, target_col, use_cache=use_cache, build_fraction=0.0, z_transform=False, random_seed=99000209)
            
    regressor2_01 = Regressor('GB2_01', ensemble.GradientBoostingClassifier(**gb_params2), 
                              features2_01, target_col, use_cache=use_cache, build_fraction=0.0, z_transform=False, random_seed=99001201) 
    regressor2_05 = Regressor('GB2_05', ensemble.GradientBoostingClassifier(**gb_params2), 
                              features2_05, target_col, use_cache=use_cache, build_fraction=0.0, z_transform=False, random_seed=99001205)
    regressor2_09 = Regressor('GB2_09', ensemble.GradientBoostingClassifier(**gb_params2), 
                              features2_09, target_col, use_cache=use_cache, build_fraction=0.0, z_transform=False, random_seed=99001209)

    regressor3a_09 = Regressor('GB3a_09', ensemble.GradientBoostingClassifier(**gb_params3), 
                               features3a_09, target_col, use_cache=use_cache, build_fraction=0.0, z_transform=False, random_seed=99003219) 
    regressor3b_09 = Regressor('GB3b_09', ensemble.GradientBoostingClassifier(**gb_params3), 
                               features3b_09, target_col, use_cache=use_cache, build_fraction=0.0, z_transform=False, random_seed=99003229)
    regressor3m    = Regressor('GB3m', ensemble.GradientBoostingClassifier(**gb_params3), 
                               features3m, target_col, use_cache=use_cache, build_fraction=0.0, z_transform=False, random_seed=99003239)

    regressor4_02 = Regressor('GB4_02', ensemble.GradientBoostingClassifier(**gb_params4), 
                              features4_02, target_col, use_cache=use_cache, build_fraction=0.0, z_transform=False, random_seed=99004202)
    regressor4_05 = Regressor('GB4_05', ensemble.GradientBoostingClassifier(**gb_params4), 
                              features4_05, target_col, use_cache=use_cache, build_fraction=0.0, z_transform=False, random_seed=99004205)
    regressor4_08 = Regressor('GB4_08', ensemble.GradientBoostingClassifier(**gb_params4), 
                              features4_08, target_col, use_cache=use_cache, build_fraction=0.0, z_transform=False, random_seed=99004208)
    regressor4_095 = Regressor('GB4_095', ensemble.GradientBoostingClassifier(**gb_params4), 
                               features4_095, target_col, use_cache=use_cache, build_fraction=0.0, z_transform=False, random_seed=990042095)

    regressor5a = Regressor('GB5a', ensemble.GradientBoostingClassifier(**gb_params5), 
                            features5a, target_col, use_cache=use_cache, build_fraction=0.0, z_transform=False, random_seed=99005201)
    regressor5b = Regressor('GB5b', ensemble.GradientBoostingClassifier(**gb_params5), 
                            features5b, target_col, use_cache=use_cache, build_fraction=0.0, z_transform=False, random_seed=99005202)
    regressor5c = Regressor('GB5c', ensemble.GradientBoostingClassifier(**gb_params5), 
                            features5c, target_col, use_cache=use_cache, build_fraction=0.0, z_transform=False, random_seed=99005203)
    regressor5d = Regressor('GB5d', ensemble.GradientBoostingClassifier(**gb_params5), 
                            features5d, target_col, use_cache=use_cache, build_fraction=0.0, z_transform=False, random_seed=99005204)
    regressor5e = Regressor('GB5e', ensemble.GradientBoostingClassifier(**gb_params5), 
                            features5e, target_col, use_cache=use_cache, build_fraction=0.0, z_transform=False, random_seed=99005205)
    regressor5f = Regressor('GB5f', ensemble.GradientBoostingClassifier(**gb_params5), 
                            features5f, target_col, use_cache=use_cache, build_fraction=0.0, z_transform=False, random_seed=99005206)
    
    # Combine the individual regressors to a single model (weighted average)
    if coeffs is None:
        coeffs = [1 , 19,23,22 , 16,16,16 ,  14,14,7 , 16,14,48,7,1]
    regressor012345 =  MultiRegressor('MRall', [regressor0_02, #regressor0_05, regressor0_09,
                                                regressor1_01, regressor1_05, regressor1_09,   
                                                regressor2_01, regressor2_05, regressor2_09,
                                                regressor3a_09, regressor3b_09, regressor3m,
                                                #regressor4_02, regressor4_05, regressor4_08, regressor4_095,    
                                                regressor5a, regressor5b, regressor5c, regressor5d, #regressor5e, 
                                                regressor5f,
                                               ],
                                     coeffs=coeffs, target=target_col)
    
    return regressor012345

            
# ================================================================================================================================
# Main
# ================================================================================================================================

if __name__ == "__main1__":
    import sys
    import json    

    # Check command-line arguments
    if len(sys.argv)>3 or len(sys.argv)<2 or (sys.argv[1] not in ['prep','run']) or (sys.argv[1]=='run' and len(sys.argv)>2):
        print 'Usage:'
        print ' "python ai2_Cardal.py prep [feature: 0-%d]" - prepare feature, or all features (if none supplied)' % np.max(FeatureExtractor.ALL_FEATURE_TYPES.values())
        print ' "python ai2_Cardal.py run" - run training and prediction phases, create submission file'
        exit()
     
    # Read params from SETTINGS.json
    with open('SETTINGS.json') as f:
        json_params = json.load(f)
 
    print '===> Kaggle AI2 competition: Cardal solution'
    print '     %s' % ', '.join(['%s=%s'%(k,v) for k,v in sorted(json_params.iteritems())])

    base_dir       = json_params['BASE_DIR']
    input_dir      = '%s/%s' % (base_dir, json_params['INPUT_DIR'])
    corpus_dir     = '%s/%s' % (base_dir, json_params['CORPUS_DIR'])
    submission_dir = '%s/%s' % (base_dir, json_params['SUBMISSION_DIR'])
    
    ftypes = None
    if sys.argv[1] == 'prep':
        exec_type = 'prep'
        if len(sys.argv) > 2:
            ftypes = [int(sys.argv[2])]
    else: 
        exec_type = 'run'
    
    # Make sure all directories exist
    print '\n--> Making sure all directories exist'
    if not os.path.exists(base_dir):
        raise RuntimeError('Base directory (%s) does not exist' % base_dir)
    if not os.path.exists(input_dir):
        raise RuntimeError('Input directory (%s) does not exist' % input_dir)
    if not os.path.exists(corpus_dir):
        raise RuntimeError('Corpus directory (%s) does not exist' % corpus_dir)
    create_dirs([submission_dir])

    train_file      = json_params['TRAINING_FILE']
    validation_file = json_params['VALIDATION_FILE']
    test_file       = json_params['TESTING_FILE']

    # Initializations
    np.random.seed(15102015)
    
    SpecialWords.read_nltk_stopwords()
    BaseRegressor.CACHE_DIR = '%s/regressor_cache' % base_dir

    # ------------------------------------------------------------------------------------------------
    # Read input files
    # ------------------------------------------------------------------------------------------------
    print '\n--> Reading input files'
    
    train_q = read_input_file(input_dir, filename=train_file, sep='\t' if train_file.endswith('.tsv') else ',', max_rows=1000000)
    print 'Read %d train questions' % len(train_q) 
    sub_complex_answers(train_q)
    add_qa_features(train_q)
    mark_dummy_questions(train_q)
    assert np.sum(train_q['is_dummy'])==0
    train_b = prp_binary_dataf(train_q)
    print 'Prepared %d binary train samples' % len(train_b)

    valid_q = read_input_file(input_dir, filename=validation_file, sep='\t' if validation_file.endswith('.tsv') else ',', max_rows=1000000) 
    print 'Read %d validation questions' % len(valid_q)
    sub_complex_answers(valid_q)
    add_qa_features(valid_q)
    mark_dummy_questions(valid_q)
    valid_b = prp_binary_dataf(valid_q)
    print 'Prepared %d binary validation samples' % len(valid_b)

    test_q = read_input_file(input_dir, filename=test_file, sep='\t' if test_file.endswith('.tsv') else ',', max_rows=1000000) 
    print 'Read %d test questions' % len(test_q)
    sub_complex_answers(test_q)
    add_qa_features(test_q)
    mark_dummy_questions(test_q)
    test_b = prp_binary_dataf(test_q)
    print 'Prepared %d binary test samples' % len(test_b)
    sys.stdout.flush()
        
    assert len(set(train_q['ID']).union(valid_q['ID'])) == (len(train_q['ID']) + len(valid_q['ID']))

    # Correct answers of training questions    
    target_col = 'correct'
    targets_q = dict(zip(train_q.index,train_q['correctAnswer']))
    targets_b = np.array(train_b[target_col])

    # ------------------------------------------------------------------------------------------------
    # Compute/read features
    # ------------------------------------------------------------------------------------------------
    norm_scores_default = False
    data_pkl_file = None
    #data_pkl_file = '%s/all_data.pkl' % base_dir

    if (data_pkl_file is None) or (not os.path.exists(data_pkl_file)):
        print '\n--> Computing/reading features'
        fext = FeatureExtractor(base_dir=base_dir, recalc=False, norm_scores_default=norm_scores_default, print_level=2)
    
        fext.prepare_word_sets(corpus_dir=corpus_dir, train_b=train_b, valid_b=None, test_b=None)
        
        fext.prepare_corpuses(corpus_dir=corpus_dir, train_b=train_b, valid_b=valid_b, prp_wiki_corpuses=True)
        sys.stdout.flush()
        
        fext.prepare_features(dataf_q=train_q, dataf_b=train_b, train_df=train_b, aux_b=None   , cache_dir='funcs_train', ftypes=ftypes)
    
        fext.prepare_features(dataf_q=valid_q, dataf_b=valid_b, train_df=train_b, aux_b=None   , cache_dir='funcs_valid', ftypes=ftypes)
    
        fext.prepare_features(dataf_q=test_q , dataf_b=test_b , train_df=train_b, aux_b=valid_b, cache_dir='funcs_test' , ftypes=ftypes)
        
        if exec_type == 'prep':
            print '==> Done prep'
            exit()
        sys.stdout.flush()    
        if data_pkl_file is not None:
            save_to_pkl(data_pkl_file, (train_q, train_b, valid_q, valid_b, test_q, test_b))
    else:
        (train_q, train_b, valid_q, valid_b, test_q, test_b) = load_from_pkl(data_pkl_file)

    if not norm_scores_default:
        print '\n--> Normalizing features'
        for feature in train_b.columns:
            if feature.endswith('_over') or feature.endswith('_under'):
                #print '    Normalizing feature %s' % feature
                optf = lambda x: 1.0 - calc_auc(train_b['correct'], normalize_scores(train_b[feature], num_scores_per_row=4, pow=x[0], laplace=np.abs(x[1])))
                best_params = [1,0.00000001] #[1, 0.000001] #if feature.endswith('_over') else [1, 1e-7] #[1.0, 1e-100] 
                best_auc = calc_auc(train_b['correct'], normalize_scores(train_b[feature], num_scores_per_row=4, pow=best_params[0], laplace=best_params[1]))
                #print '      Best  params: %5.2f , %7.1e ; AUC = %.5f' % (best_params[0], best_params[1], best_auc)
                train_b[feature] = normalize_scores(train_b[feature], num_scores_per_row=4, pow=best_params[0], laplace=best_params[1])
                valid_b[feature] = normalize_scores(valid_b[feature], num_scores_per_row=4, pow=best_params[0], laplace=best_params[1])
                test_b [feature] = normalize_scores(test_b [feature], num_scores_per_row=4, pow=best_params[0], laplace=best_params[1])


#     for f in sorted(valid_b.columns):
#         vals = np.array(valid_b[f])
#         print '%-20s' % f
#         print '  1 : %s' % str(vals[0:4]) 
#         print '  2 : %s' % str(vals[4:8]) 
#         print ' 24 : %s' % str(vals[(23*4):(23*4+4)]) 
#         print ' 29 : %s' % str(vals[(28*4):(28*4+4)]) 
#     dkjfdkjfdkjf()

    # ------------------------------------------------------------------------------------------------
    # Obtain model, check it on training set using cross validation
    # ------------------------------------------------------------------------------------------------
    print '\n--> Checking model on training set'

    opt_func = lambda labels,preds: calc_auc(labels, preds, two_sides=True)

    # Get the model
    regressor = get_regressor(target_col=target_col)

    if 'correctAnswer' in valid_q.columns:
        print '===== Using train+valid as train... ====='
        orig_train_q, orig_train_b, orig_targets_q, orig_targets_b = train_q, train_b, targets_q, targets_b
        train_q = pd.concat([train_q, valid_q])
        train_b = pd.concat([train_b, valid_b])
        targets_q = dict(zip(train_q.index,train_q['correctAnswer']))
        targets_b = np.array(train_b['correct'])

    # Cross-validation on training data
    cv_kfold = 5  
    num_cv_iters = 1
    iter_scores, iter_acc_scores = [], []
    for it in range(num_cv_iters):
        print '\n-> CV iter %d\n' % it
        sys.stdout.flush()
        trainer = Trainer(train_b, target_col, opt_func=opt_func, groupby_col='ID')
        trainer.prp_train_test_data(test_fraction=0, build_fraction=0, random_seed=97531 + 17*it, shuffle_flag=True)
        cv_score, cv_preds, cv_test_ids = trainer.cross_validation(regressor, trainer.dataf, trainer.train_ids, kfold=cv_kfold, full_output=True, random_seed=135790 + 25*it)
        print '   CV score = %.4f' % cv_score
        all_preds = np.ones(len(train_b)) * (-1)
        np.put(all_preds, cv_test_ids, cv_preds)
        assert np.all(all_preds != -1)
        target_preds = get_predictions_from_binary_dataf(train_b, column=None, scores=all_preds, direction='max')
        cv_acc = calc_accuracy(targets_q, target_preds)
        print '   CV accuracy = %.5f' % cv_acc
        iter_scores.append(cv_score)
        iter_acc_scores.append(cv_acc)

    if num_cv_iters > 0:
        print 'CV scores   (mean = %.5f): %s' % (np.mean(iter_scores    ), ', '.join(['%.5f'%s for s in iter_scores]))
        print 'CV accuracy (mean = %.5f): %s' % (np.mean(iter_acc_scores), ', '.join(['%.5f'%s for s in iter_acc_scores]))


    # ------------------------------------------------------------------------------------------------
    # Apply the model on the validation and testing sets, build submission files
    # ------------------------------------------------------------------------------------------------
    if 'correctAnswer' in valid_q.columns:
        print '===== Using orig train+valid as train... ====='
        train_q, train_b, targets_q, targets_b = orig_train_q, orig_train_b, orig_targets_q, orig_targets_b
        
    trainer = Trainer(train_b, target_col, opt_func=opt_func, groupby_col='ID')
    trainer.prp_train_test_data(test_fraction=0, build_fraction=0, random_seed=975317, shuffle_flag=True)

    for di,(data_name,data_q,data_b) in enumerate([('validation', valid_q, valid_b), ('test', test_q, test_b)]):
        print '\n--> Applying model on %s data' % data_name
        
        if 'correctAnswer' in data_q.columns:
            data_targets = np.array(data_b[target_col])
            data_targets_names = dict(zip(data_q.index,data_q['correctAnswer']))
        else:
            data_targets, data_targets_names = None, None

        data_score, data_preds = trainer.fit_model(regressor, dataf=trainer.dataf, test_dataf=data_b, print_flag=False, full_output=True, cache_key=(di,di,di,di))
        if data_targets_names is not None:
            print '%s score    = %.5f' % (data_name, data_score)
        data_target_preds = get_predictions_from_binary_dataf(data_b, column=None, scores=data_preds, direction='max')
        print '%s predictions (%d):' % (data_name, len(data_q))
        for qi in range(np.min([30,len(data_q)])):
            id, scores = np.array(data_q['ID'])[qi], data_preds[(4*qi):(4*qi+4)]
            assert np.all([np.array(data_b['ID'])[j]==id for j in range(4*qi,4*qi+4)])  
            print '  %s: %s (%s)' % (id, data_target_preds[id], ', '.join(['%.2e'%s for s in scores]))
        if data_targets_names is not None:
            data_acc = calc_accuracy(data_targets_names, data_target_preds)
            print '%s accuracy = %.5f' % (data_name, data_acc)

        if True:
            submission_filename = '%s/ai2_cardal_%s_%4d%02d%02d_%02d%02d.csv' % (submission_dir, data_name, time.localtime()[0], time.localtime()[1], time.localtime()[2], time.localtime()[3], time.localtime()[4])                
            save_submission(submission_filename, data_target_preds)
    
    print 'Done.'

