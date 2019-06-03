import spacy
from spacy.matcher import Matcher, PhraseMatcher

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)
matcher2 = PhraseMatcher(nlp.vocab)

sentences = [
    "The company said it expects to release third-quarter results in mid-November.",
    "The thrift announced the plan Aug. 21.",
    "The split and quarterly dividend will be payable Jan. 3 to stock of record Nov. 16, the company said.",
    "Ogden Projects, whose shares began trading on the New York Stock Exchange in August, closed yesterday at $26.875, down 75 cents.",
    "A spokeswoman for Crum amp Forster said employees were told early this week that numerous staff functions for the personal insurance lines were going to be centralized as a cost-cutting move.",
    "For the quarter ended Sept. 30, Delta posted net income of $133.1 million, or $2.53 a share, up from $100 million, or $2.03 a share, a year earlier."
    ]

def remove_overlapping_matches(matches):
    remove = []
    for m1 in range(len(matches)-1):
        if m1 in remove:
            continue
        for m2 in range(m1+1, len(matches)):
            if m2 in remove:
                continue
            _, s1, e1 = matches[m1]
            _, s2, e2 = matches[m2]
            if s1 >= s2 and e1 <= e2:
                remove.append(m1)
                break
            if s2 >= s1 and e2 <= e1:
                remove.append(m2)
                continue

    return [matches[m] for m in range(len(matches)) if m not in remove]

def markup_timex(doc, matches):
    matches = remove_overlapping_matches(matches)
    out = ""
    prev = 0
    for _, start, end in matches:
        out += str(doc[prev:start])+'<TIMEX>'+str(doc[start:end])+'</TIMEX>'
        prev = end
    out += str(doc[prev:])
    return out


if __name__ == "__main__":
    terms = ["a year earlier", "this week", "mid-November"] # "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sept", "Oct", "Nov", "Dec"
    patterns = [nlp.make_doc(text) for text in terms]

    months_ = r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sept|Oct|Nov|Dec)'
    month_regex = r'(January|February|March|April|May|June|July|August|September|October|November|December)'
    month = {"TEXT": {"REGEX":months_}}
    year = {"SHAPE": "dddd", "<=":"2019"}
    day = {"SHAPE": "dd", ">":"0", "<=":"31"}
    day2 = {"SHAPE": "d", ">":"0", "<=":"31"}

    date =[{**month}, {**day, 'OP':'?'}, {**day2, 'OP':'?'}, {**day, 'OP': '?'}]
    matcher.add("TIMEX", None, date)
    matcher2.add("TIMEX", None, *patterns)
    for s in sentences:
        doc = nlp(s)
        matches = matcher(doc)
        matches += matcher2(doc)
        print(markup_timex(doc, matches))
