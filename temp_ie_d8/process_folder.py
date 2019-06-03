import re 
import os, sys

input_dir = sys.argv[1]
input_dir = os.path.abspath(input_dir)
output_dir = os.path.join(input_dir, "../sub")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

days_ = "(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"
week_ = "(week|this week|next week|last week|early this week|)"
months_ = "(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sept|Oct|Nov|Dec)"
#wm_ = "(week|month|year)"
other_ = "(August|yesterday|third-quarter|mid-Novemberthe third quarter|the corresponding period last year|This quarter|this year\'s third quarter|last year|last year\'s quarter|today|now|the weekend|the short term|earlier this month|Earlier this year|currently|A year earlier|the year|Now|centuries|Last year|the past|four-day|this month|this year|a year ago|earlier this year|)"
timex_1 = r'((%s.?,?\s+)(\d{1,2}?\s+)?\d{1,2})' % months_
timex_2 = r'((%s))' % other_
timex_3 = r'((%s))' % days_
timex_4 = r'((%s))' % week_
#timex_5 = r'(\d{1,2}(-%s))' % wm_

months = '(January|February|March|April|May|June|July|August|September|October|November|December)'
date = '((%s\s+)?(\d{1,2},?\s+)?\d{4})' % months
timex = '(%s|%s|%s|%s|%s|%s)' % (date, months, timex_1, timex_2, timex_3, timex_4) # timex_5

for d in os.listdir(input_dir):
    with open (os.path.join(input_dir,d), 'r') as inp:
        text = inp.read()

    with open (os.path.join(output_dir,d.replace('raw', 'sub')), 'w') as out:
        out.write(re.sub(timex, r'<TIMEX>\1</TIMEX>', text))
