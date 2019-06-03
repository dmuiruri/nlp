#! /usr/bin/env python

import re

# sentences = [
#     "Waxman Industries Inc. said holders of $6,542,000 face amount of its 6 1/4% convertible subordinated debentures, due March 15, 2007, have elected to convert the debt into about 683,000 common shares.",
#     "Seventy-five million copies of the rifle have been built since it entered production in February 1947, making it history's most widely distributed weapon.",
#     "Many of the local religious leaders who led the 1992 protests have moved."
#     ]

sentences = [
    "The company said it expects to release third-quarter results in mid-November.",
    "The thrift announced the plan Aug. 21.",
    "The split and quarterly dividend will be payable Jan. 3 to stock of record Nov. 16, the company said.",
    "Ogden Projects, whose shares began trading on the New York Stock Exchange in August, closed yesterday at $26.875, down 75 cents.",
    "A spokeswoman for Crum amp Forster said employees were told early this week that numerous staff functions for the personal insurance lines were going to be centralized as a cost-cutting move.",
    "For the quarter ended Sept. 30, Delta posted net income of $133.1 million, or $2.53 a share, up from $100 million, or $2.03 a share, a year earlier."
    ]

def exercise11(sentences):
    """
    Capture temporal expressions in sentences.
    """
    months = '(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sept|Oct|Nov|Dec)'
    other_time = '(August|yesterday|third-quarter|mid-November|early this week)'
    timex = r'((%s.?,?\s+)(\d{1,2}?\s+)?\d{1,2})' % months
    timex2 = r'(((%s)))' % other_time
    # timex = r'((%s.?\s+)?(\d{1,2},?\s+)?\d{2})' % months
    for s in sentences:
        text = "{}".format(re.sub(timex, r'<TIMEX>\1</TIMEX>', s))
        res = "{}".format(re.sub(timex2, r'<TIMEX>\1</TIMEX>', text))
        print("{}\n".format(res))

if __name__ == '__main__':
    exercise11(sentences)
