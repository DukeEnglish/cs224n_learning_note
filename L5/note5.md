## Video
This video is about syntax parsing, dependency grammar and the way to parse them. 

- Part1 is about Syntactic Structure: Consistency and Dependency. They are two views of linguistic structure. 
  - Consistency (aka phrase structure grammar and context-free grammars, CEGs ) is just about the pos(part of speech) of each word and phrase combined by words in a sentence. *Phrase structure organizes words into nested constituents.*
  - Dependency is involved in context information in the sentence, which makes it contain more rich info. *Dependency structure shows which words depend on (modify or*
    *are arguments of) which other words.*
  - This lecture also provides us some ambiguity case including *Prepositional phrase attachment ambiguity* (介词短语依附歧义), coordination scope (协调范围歧义), Verb Phrase (VP) attachment ambiguity (动词短语依附歧义)

**Note: Why do we need sentence structure?**

**- We need to understand sentence structure in order to be able to interpret language correctly** 

**- Humans communicate complex ideas by composing words together into bigger units to convey complex meanings** 

**- We need to know what is connected to what** 

- Part2 is about Dependency Grammar and Treebank. They are two different 

"""I'd like to complete it after reading the note for this lecture""


句法分析例子：
例子来源：http://ltp.ai/demo.html
标注：https://ltp.readthedocs.io/zh_CN/latest/appendix.html#id5
1、元芳你怎么看
Root 元芳 你 怎么 看
按照数组来组织，每一个token都是一个index，所以会有如下的结果输出
<space> 4:SBV	4:SBV	4:ADV	0:HED
4:SBV：4代表第四个位置的token对这个位置【4:SBV所在的位置】的关系，是SBV关系：subject-verb，主谓关系。
0:HED：指这个位置的词语是整个句子的核心
