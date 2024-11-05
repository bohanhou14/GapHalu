**Task Overview**

You are tasked to classify categories of gaps between machine-generated and human-written legal analysis. 

**Definitions**

**generation**: machine-generated legal analysis.

**target**: human-written legal analysis. Note that the target is only one form of acceptable legal analysis. There are other acceptable legal analysis. It is possible for a generation to not match with the target but still considered acceptable.

**previous_context**: we set the goal of LLM to generate a paragraph of legal analysis and feed in the previous context to this paragraph as the input.

**cited_paragraphs**: in addition to the previous context, we also feed in the other paragraphs that are supposed to be cited in this generation.

**citation**: citation refers to the special string which points to a legal case, with style and format specified by the Bluebook.

**claim**: the sentence which is supported by the citation, i.e. the case referred to. Claim usually appears in the vicinity of the citation.

**Intrinsic Gaps**: the presence of instrinsic gaps signals that the machine-generated legal analysis is an unacceptable form. We can tell instrinsic gaps exist by _only_ looking at the previous context and the generation itself.

**Extrinsic Gaps**: extrinsic gaps, as its name suggests, can be discovered by comparing the generation with external texts, i.e. the cited paragraphs or the target paragraph that can be seen as the "answer". Extrinsic gaps contain two kinds: citation content mismatch and target mismatch. Target mismatch does not indicate that the generated legal analysis is necessarily wrong.

**Annotation Instructions**
Here are some reference articles for legal cases: 
# Reference case {case_key_1}
{text of cited case 1} 
# Reference case {case_key_2}
{text of cited case 2}
...
# Reference case {case_key_N} {text of cited case N}
Here is the text I’ve written so far: # Paragrah
{previous_text}
Continue to write it following the style of my writeup. Your answer contains 100 to 400 words. You must explicitly use the reference cases and mention their reference ids, i.e. {case_key_1}, {case_key_2} . . . {case_key_N}. Wrap your answer with <answer></answer>. Make your answer concise and avoid redundant languages.

Receiving the prompt above, a language model will generate a paragraph of legal analysis, but often times they make different kinds of errors and mismatches. 

The instructions for you to classify these errors and mismatches are as follows:

You should classify the LLM-generated legal analysis to these categories:

1. Intrinsic gap:
This category refers to generation that is unacceptable, due to the language model has fundamentally failed to follow the instruction, or make a lot of redundancy, or generate something that does not look like legal text (structural mismatch). More specificially, if it makes one or more of the following:

- Redundancy (sentence-level, appearing as neural degeneration): the generation appears to make repetitive statements that do not add more meaning to the analysis. For example, multiple occurences of an exact sentence or phrase.
- Citation Format Mismatch: the generation appears not matching with the citation format of the standard Bluebook.
    - Please be aware that, for example, 440 U.S. 48, 55' is a proper format. Although its full citation should be 'Butner v. United States, 440 U.S. 48, 55 (1979)', the format '440 U.S. 48, 55' is still acceptable as a concise form.
- Structural Mismatch: the generation appears to generate the document from scratch (like containing words such as "ORDER" which only appear in the beginning).
- Stylistic Mismatch: contain sentences that do not match the styles of legalese.

If this type of gaps is present, add the label `1`. Continue to item 2.

Side note: You should be able to classify this purely based on the generation itself, without having to look at cited examples.

2. Target mismatch:

While language model's generated text may be obviously wrong and substantively different from the target (i.e. the original/target text from the case), the claims it makes are still logically and factually sound and can be seen as acceptable. This could be because
 - Chain Cite: the citations appear in a chain cite but the generation cites them parallely, or the other way around.
    - Clarification: "The rule that certain acts of a creditor in the course of a bankruptcy proceeding and during the statutory period for filing proof of claim, may give rise to something equivalent to a proof of claim and afford a sufficient basis for allowing an amendment after the statutory period for filing, was recognized and applied in many cases decided before the 1938 amendment of the Bankruptcy Act. See In re Atlantic Gulf & Pacific S. S. Corporation, D.C., 26 F.2d 751; In re Fant, D.C., 21 F.2d 182; Globe Indemnity Co. of Newark, N. J., v. Keeble, 4 Cir., 20 F.2d 84; In re Coleman & Titus Corporation, D.C., 286 F. 303; In re Roeber, 2 Cir., 127 F. 122." would be a chain cite because all of these citations support the previous claim "The rule that certain acts of a creditor in the course of a bankruptcy proceeding and during the statutory period for filing proof of claim, may give rise to something equivalent to a proof of claim and afford a sufficient basis for allowing an amendment after the statutory period for filing, was recognized and applied in many cases decided before the 1938 amendment of the Bankruptcy Act."
 - Reverse Cite: the citations reverse the ruling in each other but the generation cites them parallely, or the other way around.
 - Compound Cite: the citations of different cases are cited together, separated by semicolons, or the other way around.
 
 Although it does not match with the target, it is still considered somewhat acceptable, but we should label it out.

 If this type of mismatch is present add the label `2`. Continue to item 3.

3. Citation Mismatch: The language model's generated text does not align with the content of the citation it points to. This might be because one or more of the following:

- Claim Hallucination: the claim supported by the citation is not truthful or not related to the context or from cited paragraphs or the previous context. The generated text makes different and possibly (although not necessarily) contradictory claims about one or more citations, which you can check from comparing to the reference case. Or, the generated text attributes information from one citation to a different citation.
- Retrieval Inaccuracy: the claims supported by the citation is not relevant because the cited paragraph looks irrelevant compared to the target paragraph. 
- Citation Hallucination: the citation is non-existent or pulled from a citation in the cited paragraphs or the previous context, or there misses a citation (the generated text fails to use one of the citations that were given to it).

If this type of mismatch is **present**, add the label `3` and move on to the next example. If none of the above errors are present, label '0'.

Note that where an example falls into multiple categories, you should include both labels, separated by a comma.