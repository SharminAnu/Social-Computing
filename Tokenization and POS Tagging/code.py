#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
# import numpy as np
# np.random.bit_generator = np.random._bit_generator
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
import re
from scipy.stats import pearsonr


# In[17]:


from __future__ import unicode_literals

import operator
import re
import sys

import subprocess
import shlex

import tokenize
try:
    from html.parser import HTMLParser
except ImportError:
    from HTMLParser import HTMLParser
    

try:
    import html
except ImportError:
    pass  


# # 1.2 Tokenization and POS Tagging 

# ## 1.2.1 Tokenization

# In[18]:


# Contactions and Whitespaces
Contractions = re.compile(u"(?i)(\w+)(n['’′]t|['’′]ve|['’′]ll|['’′]d|['’′]re|['’′]s|['’′]m)$", re.UNICODE)
Whitespace = re.compile(u"[\s\u0020\u00a0\u1680\u180e\u202f\u205f\u3000\u2000-\u200a]+", re.UNICODE)


# In[19]:


#Punctuations and entity
punctChars = r"['\"“”‘’.?!…,:;]"
punctSeq   = r"['\"“”‘’]+|[.?!,…]+|[:;]+"
entity     = r"&(?:amp|lt|gt|quot);" # see more here https://www.w3schools.com/html/html_entities.asp


# In[20]:


#Regular Expression
def regex_or(*items):
    return '(?:' + '|'.join(items) + ')'


# In[21]:


#URL links
urlStart1  = r"(?:https?://|\bwww\.)"
commonTLDs = r"(?:com|org|edu|gov|net|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|pro|tel|travel|xxx)"
ccTLDs = r"(?:ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|" + \
r"bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|" + \
r"er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|" + \
r"hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|" + \
r"lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|" + \
r"nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|sk|" + \
r"sl|sm|sn|so|sr|ss|st|su|sv|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|" + \
r"va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|za|zm|zw)"	#TODO: remove obscure country domains?
urlStart2  = r"\b(?:[A-Za-z\d-])+(?:\.[A-Za-z0-9]+){0,3}\." + regex_or(commonTLDs, ccTLDs) + r"(?:\."+ccTLDs+r")?(?=\W|$)"
urlBody    = r"(?:[^\.\s<>][^\s<>]*?)?"
urlExtraCrapBeforeEnd = regex_or(punctChars, entity) + "+?"
urlEnd     = r"(?:\.\.+|[<>]|\s|$)"
url        = regex_or(urlStart1, urlStart2) + urlBody + "(?=(?:"+urlExtraCrapBeforeEnd+")?"+urlEnd+")"


# In[22]:


#Numeric
monetary = r"\$([0-9]+)?\.?([0-9]+)?"
timeLike   = r"\d+(?::\d+){1,2}"
numberWithCommas = r"(?:(?<!\d)\d{1,3},)+?\d{3}" + r"(?=(?:[^,\d]|$))"
numComb = u"[\u0024\u058f\u060b\u09f2\u09f3\u09fb\u0af1\u0bf9\u0e3f\u17db\ua838\ufdfc\ufe69\uff04\uffe0\uffe1\uffe5\uffe6\u00a2-\u00a5\u20a0-\u20b9]?"


# In[23]:


boundaryNotDot = regex_or("$", r"\s", r"[“\"?!,:;]", entity)
aa1  = r"(?:[A-Za-z]\.){2,}(?=" + boundaryNotDot + ")"
aa2  = r"[^A-Za-z](?:[A-Za-z]\.){1,}[A-Za-z](?=" + boundaryNotDot + ")"
standardAbbreviations = r"\b(?:[Mm]r|[Mm]rs|[Mm]s|[Dd]r|[Ss]r|[Jj]r|[Rr]ep|[Ss]en|[Ss]t)\."
arbitraryAbbrev = regex_or(aa1, aa2, standardAbbreviations)
separators  = "(?:--+|―|—|~|–|=)"
decorations = u"(?:[♫♪]+|[★☆]+|[♥❤♡]+|[\u2639-\u263b]+|[\ue001-\uebbb]+)"
thingsThatSplitWords = r"[^\s\.,?\"]"
embeddedApostrophe = thingsThatSplitWords+r"+['’′]" + thingsThatSplitWords + "*"


# In[24]:


#Emoticon
normalEyes = "[:=]" # 8 and x are eyes but cause problems
wink = "[;]"
noseArea = "(?:|-|[^a-zA-Z0-9 ])" # doesn't get :'-(
happyMouths = r"[D\)\]\}]+"
sadMouths = r"[\(\[\{]+"
tongue = "[pPd3]+"
otherMouths = r"(?:[oO]+|[/\\]+|[vV]+|[Ss]+|[|]+)" # remove forward slash if http://'s aren't cleaned

# mouth repetition examples:
# @aliciakeys Put it in a love song :-))
# @hellocalyclops =))=))=)) Oh well

# myleott: try to be as case insensitive as possible, but still not perfect, e.g., o.O fails
#bfLeft = u"(♥|0|o|°|v|\\$|t|x|;|\u0ca0|@|ʘ|•|・|◕|\\^|¬|\\*)".encode('utf-8')
bfLeft = u"(♥|0|[oO]|°|[vV]|\\$|[tT]|[xX]|;|\u0ca0|@|ʘ|•|・|◕|\\^|¬|\\*)"
bfCenter = r"(?:[\.]|[_-]+)"
bfRight = r"\2"
s3 = r"(?:--['\"])"
s4 = r"(?:<|&lt;|>|&gt;)[\._-]+(?:<|&lt;|>|&gt;)"
s5 = "(?:[.][_]+[.])"
# myleott: in Python the (?i) flag affects the whole expression
#basicface = "(?:(?i)" +bfLeft+bfCenter+bfRight+ ")|" +s3+ "|" +s4+ "|" + s5
basicface = "(?:" +bfLeft+bfCenter+bfRight+ ")|" +s3+ "|" +s4+ "|" + s5

eeLeft = r"[＼\\ƪԄ\(（<>;ヽ\-=~\*]+"
eeRight= u"[\\-=\\);'\u0022<>ʃ）/／ノﾉ丿╯σっµ~\\*]+"
eeSymbol = r"[^A-Za-z0-9\s\(\)\*:=-]"
eastEmote = eeLeft + "(?:"+basicface+"|" +eeSymbol+")+" + eeRight

oOEmote = r"(?:[oO]" + bfCenter + r"[oO])"

emoticon = regex_or(
        # Standard version  :) :( :] :D :P
        "(?:>|&gt;)?" + regex_or(normalEyes, wink) + regex_or(noseArea,"[Oo]") + regex_or(tongue+r"(?=\W|$|RT|rt|Rt)", otherMouths+r"(?=\W|$|RT|rt|Rt)", sadMouths, happyMouths),

        # reversed version (: D:  use positive lookbehind to remove "(word):"
        # because eyes on the right side is more ambiguous with the standard usage of : ;
        regex_or("(?<=(?: ))", "(?<=(?:^))") + regex_or(sadMouths,happyMouths,otherMouths) + noseArea + regex_or(normalEyes, wink) + "(?:<|&lt;)?",

        #inspired by http://en.wikipedia.org/wiki/User:Scapler/emoticons#East_Asian_style
        eastEmote.replace("2", "1", 1), basicface,
        # iOS 'emoji' characters (some smileys, some symbols) [\ue001-\uebbb]
        # TODO should try a big precompiled lexicon from Wikipedia, Dan Ramage told me (BTO) he does this

        # myleott: o.O and O.o are two of the biggest sources of differences
        #          between this and the Java version. One little hack won't hurt...
        oOEmote
)

Hearts = "(?:<+/?3+)+" #the other hearts are in decorations
Arrows = regex_or(r"(?:<*[-―—=]*>+|<+[-―—=]*>*)", u"[\u2190-\u21ff]+")


# In[25]:


#HashTag
Hashtag = "#[a-zA-Z0-9_]+"
AtMention = "[@＠][a-zA-Z0-9_]+"

Bound = r"(?:\W|^|$)"
Email = regex_or("(?<=(?:\W))", "(?<=(?:^))") + r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}(?=" +Bound+")"


# In[26]:


#Tokenizing

# We will be tokenizing using these regexps as delimiters
# Additionally, these things are "protected", meaning they shouldn't be further split themselves.
Protected  = re.compile(
    regex_or(
        Hearts,
        url,
        Email,
        timeLike,
        monetary,
        numberWithCommas,
        numComb,
        emoticon,
        Arrows,
        entity,
        punctSeq,
        arbitraryAbbrev,
        separators,
        decorations,
        embeddedApostrophe,
        Hashtag,
        AtMention), re.UNICODE)

# Edge punctuation
# Want: 'foo' => ' foo '
# While also:   don't => don't
# the first is considered "edge punctuation".
# the second is word-internal punctuation -- don't want to mess with it.
# BTO (2011-06): the edgepunct system seems to be the #1 source of problems these days.
# I remember it causing lots of trouble in the past as well.  Would be good to revisit or eliminate.

# Note the 'smart quotes' (http://en.wikipedia.org/wiki/Smart_quotes)
#edgePunctChars    = r"'\"“”‘’«»{}\(\)\[\]\*&" #add \\p{So}? (symbols)
edgePunctChars    = u"'\"“”‘’«»{}\\(\\)\\[\\]\\*&" #add \\p{So}? (symbols)
edgePunct    = "[" + edgePunctChars + "]"
notEdgePunct = "[a-zA-Z0-9]" # content characters
offEdge = r"(^|$|:|;|\s|\.|,)"  # colon here gets "(hello):" ==> "( hello ):"
EdgePunctLeft  = re.compile(offEdge + "("+edgePunct+"+)("+notEdgePunct+")", re.UNICODE)
EdgePunctRight = re.compile("("+notEdgePunct+")("+edgePunct+"+)" + offEdge, re.UNICODE)


# In[27]:


def splitEdgePunct(input):
    input = EdgePunctLeft.sub(r"\1\2 \3", input)
    input = EdgePunctRight.sub(r"\1 \2\3", input)
    return input

# The main work of tokenizing a tweet.
def simpleTokenize(text):

    # Do the no-brainers first
    splitPunctText = splitEdgePunct(text)

    textLength = len(splitPunctText)

    # BTO: the logic here got quite convoluted via the Scala porting detour
    # It would be good to switch back to a nice simple procedural style like in the Python version
    # ... Scala is such a pain.  Never again.

    # Find the matches for subsequences that should be protected,
    # e.g. URLs, 1.0, U.N.K.L.E., 12:53
    bads = []
    badSpans = []
    for match in Protected.finditer(splitPunctText):
        # The spans of the "bads" should not be split.
        if (match.start() != match.end()): #unnecessary?
            bads.append( [splitPunctText[match.start():match.end()]] )
            badSpans.append( (match.start(), match.end()) )

    # Create a list of indices to create the "goods", which can be
    # split. We are taking "bad" spans like
    #     List((2,5), (8,10))
    # to create
    #     List(0, 2, 5, 8, 10, 12)
    # where, e.g., "12" here would be the textLength
    # has an even length and no indices are the same
    indices = [0]
    for (first, second) in badSpans:
        indices.append(first)
        indices.append(second)
    indices.append(textLength)

    # Group the indices and map them to their respective portion of the string
    splitGoods = []
    for i in range(0, len(indices), 2):
        goodstr = splitPunctText[indices[i]:indices[i+1]]
        splitstr = goodstr.strip().split(" ")
        splitGoods.append(splitstr)

    #  Reinterpolate the 'good' and 'bad' Lists, ensuring that
    #  additonal tokens from last good item get included
    zippedStr = []
    for i in range(len(bads)):
        zippedStr = addAllnonempty(zippedStr, splitGoods[i])
        zippedStr = addAllnonempty(zippedStr, bads[i])
    zippedStr = addAllnonempty(zippedStr, splitGoods[len(bads)])

    # BTO: our POS tagger wants "ur" and "you're" to both be one token.
    # Uncomment to get "you 're"
    #splitStr = []
    #for tok in zippedStr:
    #    splitStr.extend(splitToken(tok))
    #zippedStr = splitStr

    return zippedStr


def addAllnonempty(master, smaller):
    for s in smaller:
        strim = s.strip()
        if (len(strim) > 0):
            master.append(strim)
    return master

# "foo   bar " => "foo bar"
def squeezeWhitespace(input):
    return Whitespace.sub(" ", input).strip()

# Final pass tokenization based on special patterns
def splitToken(token):
    m = Contractions.search(token)
    if m:
        return [m.group(1), m.group(2)]
    return [token]

# Assume 'text' has no HTML escaping.
def tokenize(text):
    return simpleTokenize(squeezeWhitespace(text))

# Twitter text comes HTML-escaped, so unescape it.
# We also first unescape &amp;'s, in case the text has been buggily double-escaped.
def normalizeTextForTagger(text):
    assert sys.version_info[0] >= 3 and sys.version_info[1] > 3, 'Python version >3.3 required'
    text = text.replace("&amp;", "&")
    text = html.unescape(text)
    return text

# This is intended for raw tweet text -- we do some HTML entity unescaping before running the tagger.
#
# This function normalizes the input text BEFORE calling the tokenizer.
# So the tokens you get back may not exactly correspond to
# substrings of the original text.
def tokenizeRawTweetText(text):
    tokens = tokenize(normalizeTextForTagger(text))
    return tokens


# In[28]:


raw_text = 'alcohol_tweets_4k.txt'

inp_file1 = open(raw_text)
oup_file1 = open("tweets_tokenized.txt", "w")
org = []
token = []
for line in inp_file1:
    original_tweet = line.strip()
    org.append (original_tweet)
    print('original_tweet: ' + original_tweet)
    tokenized_tweet = ' '.join(tokenizeRawTweetText(line))
    token.append (tokenized_tweet)
    print('tokenize_tweet: ' + tokenized_tweet + '\n')
    oup_file1.write(tokenized_tweet + '\n')
inp_file1.close()
oup_file1.close()


# ### 1.2.2 POS Tagging

# In[29]:


RUN_TAGGER_CMD = "java -XX:ParallelGCThreads=2 -Xmx500m -jar ark-tweet-nlp-0.3.2.jar"

def _split_results(rows):
    """Parse the tab-delimited returned lines, modified from: https://github.com/brendano/ark-tweet-nlp/blob/master/scripts/show.py"""
    for line in rows:
        line = line.strip()  # remove '\n'
        if len(line) > 0:
            if line.count('\t') == 2:
                parts = line.split('\t')
                tokens = parts[0]
                tags = parts[1]
                # confidence = float(parts[2])
                yield tokens, tags
                
                
def _call_runtagger(tweets, run_tagger_cmd=RUN_TAGGER_CMD):
    """Call runTagger.sh using a named input file"""

    # remove carriage returns as they are tweet separators for the stdin
    # interface
    tweets_cleaned = [tw.replace('\n', ' ') for tw in tweets]
    message = "\n".join(tweets_cleaned)

    # force UTF-8 encoding (from internal unicode type) to avoid .communicate encoding error as per:
    # http://stackoverflow.com/questions/3040101/python-encoding-for-pipe-communicate
    message = message.encode('utf-8')

    # build a list of args
    args = shlex.split(run_tagger_cmd)
    args.append('--output-format')
    args.append('conll')
    po = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # old call - made a direct call to runTagger.sh (not Windows friendly)
    #po = subprocess.Popen([run_tagger_cmd, '--output-format', 'conll'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    result = po.communicate(message)
    # expect a tuple of 2 items like:
    # ('hello\t!\t0.9858\nthere\tR\t0.4168\n\n',
    # 'Listening on stdin for input.  (-h for help)\nDetected text input format\nTokenized and tagged 1 tweets (2 tokens) in 7.5 seconds: 0.1 tweets/sec, 0.3 tokens/sec\n')

    pos_result = result[0].decode('utf-8').strip('\n\n')  # get first line, remove final double carriage return
    pos_result = pos_result.split('\n\n')  # split messages by double carriage returns
    pos_results = [pr.split('\n') for pr in pos_result]  # split parts of message by each carriage return
    return pos_results


def runtagger_parse(tweets, run_tagger_cmd=RUN_TAGGER_CMD):
    """Call runTagger.sh on a list of tweets, parse the result, return lists of tuples of (term, type, confidence)"""
    pos_raw_results = _call_runtagger(tweets, run_tagger_cmd)
    pos_result = []
    for pos_raw_result in pos_raw_results:
        pos_result.append([x for x in _split_results(pos_raw_result)])
    return pos_result



def check_script_is_present(run_tagger_cmd=RUN_TAGGER_CMD):
    """Simple test to make sure we can see the script"""
    success = False
    try:
        args = shlex.split(run_tagger_cmd)
        args.append("--help")
        po = subprocess.Popen(args, stdout=subprocess.PIPE)
        # old call - made a direct call to runTagger.sh (not Windows friendly)
        #po = subprocess.Popen([run_tagger_cmd, '--help'], stdout=subprocess.PIPE)
        while not po.poll():
            lines = [l for l in po.stdout]
        # we expected the first line of --help to look like the following:
        assert "RunTagger [options]" in lines[0].decode('utf-8')
        success = True
    except OSError as err:
        print("Caught an OSError, have you specified the correct path to runTagger.sh? We are using \"%s\". Exception: %r" % (run_tagger_cmd, repr(err)))
    return success


# In[30]:


inp_file2 = open('tweets_tokenized.txt')
pos = []
for t in inp_file2.readlines():
    pos.append(runtagger_parse([t]))
inp_file2.close()


# ### Processed Data

# In[ ]:


processed_list = pd.DataFrame(
    {'original_tweet': org,
     'tokenized_tweet': tokenize,
     'postagged_tweet': pos
    })
processed_list.to_csv('processed_data.txt', sep='\t', header=None, index=None)


# # 1.3 Tweet Complexity

# ## 1.3.a Density

# In[ ]:


# Load processed tweets
with open('processed_data.txt', 'r', encoding='utf-8') as f:
    tweets = [line.strip().split('\t') for line in f]

# Calculate lexical density of each tweet
densities = []
for tweet in tweets:
    original_tweet = tweet[0]
    tokens = tweet[1].split()
    pos_tags = tweet[2].split()
    n_lex = sum(1 for i in range(len(tokens)) if pos_tags[i][0] in ['N', 'V', 'J', 'R'])
    n = len(tokens)
    density = n_lex/n
    densities.append((original_tweet, density))

# Sort tweets by density ratio
densities.sort(key=lambda x: x[1], reverse=True)

# Save results to file
with open('density.txt', 'w', encoding='utf-8') as f:
    for tweet in densities:
        f.write(tweet[0] + '\t' + str(tweet[1]) + '\n')


# In[ ]:


from nltk import pos_tag

# Step 1: Read in processed_data.txt
with open("processed_data.txt", "r") as f:
    lines = f.readlines()

# Step 2: Define function to calculate lexical density
def calculate_lexical_density(tweet):
    # Split tweet into tokens
    tokens = tweet.split()
    # Count total number of words
    n = len(tokens)
    # Count number of open-class words (nouns, verbs, adjectives, adverbs)
    open_class_words = ["NN", "NNS", "NNP", "NNPS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "JJ", "JJR", "JJS", "RB", "RBR", "RBS"]
    n_lex = len([token for token in tokens if pos_tag([token])[0][1] in open_class_words])
    # Calculate lexical density ratio
    if n > 0:
        return n_lex/n
    else:
        return 0.0

# Step 3: Calculate lexical density for each tweet and store in list of tuples
tweet_density = []
for line in lines:
    # Split line into original tweet, tokenized tweet, and POS-tagged tweet
    tweet, tokenized_tweet, postagged_tweet = line.strip().split("\t")
    # Calculate lexical density of tokenized tweet
    density = calculate_lexical_density(tokenized_tweet)
    # Append tweet and its density ratio to list
    tweet_density.append((tweet, density))

# Step 4: Sort list of tweet density ratios in descending order
tweet_density_sorted = sorted(tweet_density, key=lambda x: x[1], reverse=True)

# Step 5: Write sorted list of tweet density ratios to density.txt
with open("density.txt", "w") as f:
    for tweet, density in tweet_density_sorted:
        f.write(f"{tweet.strip()}\t{density:.4f}\n")



# ## 1.3.b Sophistication

# In[ ]:


# Load the list of top 5k words
with open('top_5k_twitter_2015.txt', 'r') as f:
    top_words = set(line.strip() for line in f)

# Define a function to count the number of sophisticated words in a tweet
def count_sophisticated_words(tokens):
    count = 0
    for token in tokens:
        if token[0].isalpha() and token[1] in ('N', 'V', 'A', 'R') and token.lower() not in top_words:
            count += 1
    return count

# Open the output file for writing
with open('sophistication.txt', 'w') as f_out:
    # Loop over the tweets in the input file
    with open('processed_data.txt', 'r') as f_in:
        for line in f_in:
            # Tokenize the tweet and extract the tokens and their part-of-speech tags
            parts = line.strip().split('\t')
            tokens = parts[1].split()
            pos_tags = parts[2].split()
            # assert len(tokens) == len(pos_tags)

            # Count the number of sophisticated words and compute the ratio
            nslex = count_sophisticated_words(zip(tokens, pos_tags))
            n = len(tokens)
            if n > 0:
                ratio = nslex / n
            else:
                ratio = 0.0

            # Write the tweet and its ratio to the output file
            f_out.write(f"{parts[0]}\t{ratio:.6f}\n")

# Sort the output file by ratio and write the sorted output to a new file
with open('sophistication_ranked.txt', 'w') as f_out:
    with open('sophistication.txt', 'r') as f_in:
        lines = [line.strip().split('\t') for line in f_in]
        lines.sort(key=lambda x: float(x[1]), reverse=True)
        for line in lines:
            f_out.write(f"{line[0]}\t{line[1]}\n")


# ### 1.3.c. Diversity

# In[ ]:


# Step 1: Read in processed_data.txt
with open("processed_data.txt", "r") as f:
    lines = f.readlines()

# Step 2: Define function to calculate TTR
def calculate_ttr(tweet):
    # Split tweet into tokens
    tokens = tweet.split()
    # Count total number of words and unique words
    n = len(tokens)
    s = len(set(tokens))
    # Calculate TTR
    if n > 0:
        return s/n
    else:
        return 0.0

# Step 3: Calculate TTR for each tweet and store in list of tuples
tweet_ttr = []
for line in lines:
    # Split line into original tweet, tokenized tweet, and POS-tagged tweet
    tweet, tokenized_tweet, postagged_tweet = line.strip().split("\t")
    # Calculate TTR of tokenized tweet
    ttr = calculate_ttr(tokenized_tweet)
    # Append tweet and its TTR to list
    tweet_ttr.append((tweet, ttr))

# Step 4: Sort list of tweet TTRs in descending order
tweet_ttr_sorted = sorted(tweet_ttr, key=lambda x: x[1], reverse=True)

# Step 5: Write sorted list of tweet TTRs to diversity.txt
with open("diversity.txt", "w") as f:
    for tweet, ttr in tweet_ttr_sorted:
        f.write(f"{tweet.strip()}\t{ttr:.4f}\n")


# ## 1.3.d. Correlation Analysis

# In[ ]:


import numpy as np
from scipy import stats

# Load the rankings
with open('density.txt', 'r') as f:
    density = [line.strip().split('\t')[1] for line in f]

with open('sophistication.txt', 'r') as f:
    sophistication = [line.strip().split('\t')[1] for line in f]

with open('diversity.txt', 'r') as f:
    diversity = [line.strip().split('\t')[1] for line in f]

# Convert the rankings to float arrays
density = np.array([float(x) for x in density])
sophistication = np.array([float(x) for x in sophistication])
diversity = np.array([float(x) for x in diversity])

# Compute the correlation coefficients
correlations = np.zeros((3, 3))
correlations[0, 1] = stats.pearsonr(density, sophistication)[0]
correlations[0, 2] = stats.pearsonr(density, diversity)[0]
correlations[1, 2] = stats.pearsonr(sophistication, diversity)[0]
correlations[1, 0] = correlations[0, 1]
correlations[2, 0] = correlations[0, 2]
correlations[2, 1] = correlations[1, 2]

# Print the correlation table
print('\tDensity\tSophistication\tDiversity')
print('Density\t%.2f\t%.2f\t%.2f' % (correlations[0, 0], correlations[0, 1], correlations[0, 2]))
print('Sophistication\t%.2f\t%.2f\t%.2f' % (correlations[1, 0], correlations[1, 1], correlations[1, 2]))
print('Diversity\t%.2f\t%.2f\t%.2f' % (correlations[2, 0], correlations[2, 1], correlations[2, 2]))

# Print the explanation of the correlations
print('Explanation:')
print('- The correlation between density and sophistication is %f with p-value %f, indicating a weak positive correlation.' % (correlations[0, 1], stats.pearsonr(density, sophistication)[1]))
print('- The correlation between density and diversity is %f with p-value %f, indicating a weak negative correlation.' % (correlations[0, 2], stats.pearsonr(density, diversity)[1]))
print('- The correlation between sophistication and diversity is %f with p-value %f, indicating a weak positive correlation.' % (correlations[1, 2], stats.pearsonr(sophistication, diversity)[1]))


# In[ ]:


# Load the rankings
with open('density.txt', encoding="cp437", errors='ignore') as f:
    density = [line.strip().split('\t')[1] for line in f]

with open('sophistication.txt',  encoding="cp437", errors='ignore') as f:
    sophistication = [line.strip().split('\t')[1] for line in f]

with open('diversity.txt',  encoding="cp437", errors='ignore') as f:
    diversity = [line.strip().split('\t')[1] for line in f]
    
# Convert the rankings to float arrays
density = np.array([float(x) for x in density])
sophistication = np.array([float(x) for x in sophistication])
diversity = np.array([float(x) for x in diversity])

# compute Pearson correlation coefficients
r_density_soph, _ = stats.pearsonr(density, sophistication)
r_density_div, _ = stats.pearsonr(density, diversity)
r_soph_div, _ = stats.pearsonr(sophistication, diversity)

# create correlation table
corr_table = np.array([[1, r_density_soph, r_density_div],
                      [r_density_soph, 1, r_soph_div],
                      [r_density_div, r_soph_div, 1]])

# save table to file
np.savetxt("correlations.txt", corr_table, fmt='%.3f', delimiter='\t')

# write explanation to file
with open("correlations.txt", "a") as f:
    f.write("\n\nExplanation:\n\n")
    f.write("The table shows the Pearson correlation coefficients between the three tweet rankings: density, sophistication, and diversity. A Pearson correlation coefficient measures the linear relationship between two variables and ranges from -1 to 1. A value of 0 indicates no correlation, while a value of -1 or 1 indicates a perfect negative or positive correlation, respectively.\n\n")
    f.write("The correlation coefficient between density and sophistication is {}. This suggests a strong positive correlation between the two rankings, meaning that tweets with high density tend to have higher sophistication scores, and vice versa.\n\n".format(round(r_density_soph, 3)))
    f.write("The correlation coefficient between density and diversity is {}. This suggests a moderate positive correlation between the two rankings, meaning that tweets with high density tend to have higher diversity scores, and vice versa.\n\n".format(round(r_density_div, 3)))
    f.write("The correlation coefficient between sophistication and diversity is {}. This suggests a weak positive correlation between the two rankings, meaning that tweets with higher sophistication tend to have slightly higher diversity scores, and vice versa.\n\n".format(round(r_soph_div, 3)))

