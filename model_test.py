# -*- coding: utf-8 -*-
"""
Model Test
@author: Alexander Ngo
"""

import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

file = open("model.bin",'rb')
model = pickle.load(file)

df = pd.read_csv("news.csv")

file = open("vectorizer.bin",'rb')
vectorizer = pickle.load(file)

#d = df.text[9]

d = """Trump's grandfather was a pimp and tax evader; his father a member of the KKK
****************************************
Daily the Society online desk : As they say, 
the apple doesn't fall far from the tree;
 Donald Trump's racism can easily be traced through his lineage.
Most families of enormous wealth have a dark and sometimes scandalous, even monstrous past. Donald Trump's clan is no exception to that rule. His grandfather was a pimp and a tax evader, his father a racist who would in the course of his life, clash with New York City Police as a member of the Ku Klux Klan and then as a wealthy real estate magnate, refuse to rent to people of color.
---------------------------------------------------------------------------
Donald Trump's legacy is anything but a rag to riches story. His dad kicked the bucket with $250-$300 million in the bank. The man who wants to ban all people of a particular religion from travel wasn't born with a silver spoon in his mouth, his was white gold. The only thing more obnoxious than Donald Trump himself, is his family's money -grubbing, bigoted history.
------------------------------------------------------------------------
 Trump's sexual predator characteristics - His Grandfather was a pimp, but at least he paid the women he hired
Trump's Grandfather: Pimp and Tax Evader
*****************************************
Donald Trump's grandfather, Frederick or Friedrich Trump, made his money from operating a decadent restaurant and hotel during the Gold Rush at Klondike in the Yukon.
MORE...
TRUMP'S RACIST TWEETS BETRAY A STRATEGY OF RACIALISED CLASS WARFARE
SEND HIM BACK? DONALD DRUMPF’S ROOTS
IS TRUMP A RACIST?
PHONY RESISTANCE CAN’T BEAT TRUMP
That's a nice way of saying it.
"Trump made his first fortune operating boom-town hotels, restaurants and brothels", is more accurate, according to the CBC news report, "Donald Trump's grandfather ran Canadian brothel during gold rush, author says". Author Gwenda Blair simply wrote, "The bulk of the cash flow came from the sale of liquor and sex."
Trump's grandfather was born in Germany, to parents who were employed by a vineyard. He moved to New York City in 1885 where he became a barber. After six years of this, Frederick Trump moved across the United States to Seattle, Washington, where he owned and operated what he referred to as a "decadent restaurant" that was actually called "Poodle Dog" in Seattle's red light district. Interestingly, the name and concept that had already been established in San Francisco. (He named his restaurant after a dog but would later make money selling horse meat) Around this time Frederick Trump became a US citizen.
A Yukon Sun Newspaper writer described his business: "For single men the Arctic has excellent accommodations as well as the best restaurant in Bennett, but I would not advise respectable women to go there to sleep as they are liable to hear that which would be repugnant to their feelings – and uttered, too, by the depraved of their own sex".
Trump moved to Monte Cristo, Wash. in 1894, and then four years later, shortly after the Klondike gold rush began, he relocated again to Bennett, British Columbia.  Here he ran the "Arctic Restaurant and Hotel".  He would next build the "White Horse restaurant and Inn" in Whitehorse, Yukon.
 Trump’s Alleged Rape Of 13-Year-Old 
Girl Isn't Going Away*****************
An article published this year by Politico, explains that Frederick Trump sold off his investments and returned to Germany in 1901, as he sensed the end of the gold rush and a subsequent end to prostitution.  The following year, he married his former neighbor, Elizabeth Christ in his native German town of Kallstadt.  Then he came under heavy scrutiny by the German government,
The country had compulsory military service for men which had to be fulfilled by the age of 35.  Donald Trump's grandfather waited until he was 35 to go back to Germany.  He had already amassed great wealth worth well more than half a million US dollars, or 80,000 marks.  While his town council was eager to keep Trump and his money, who billed himself as a man who "avoided bars" and led "a quiet life", other German authorities had a different plan, the Politico article explains.  In their view, Trump had relocated to Germany in order to avoid both tax and military-service obligations.
"...the regional authorities refused to let Trump off the hook. Unlike his grandson, who would become too big to fail in business and, more recently, to ignore in politics, Friedrich Trump was not big enough to get away with being a draft-dodger. He and his wife, then pregnant with Fred, Donald’s father, would not be allowed to resume their German citizenship and it would not be extended to their daughter; instead, they were deported—the same fate that Donald would like to impose on undocumented immigrants in the U.S. today."  
The fact that Donald Trump has done so well in the Republican polls is quite amazing, as the party that we think of as conservative, would be expected to shun a man who created exploitative beauty pageants and was born rich strictly because of the nefarious activities of his ancestors.  
Trump's Father: 
a Lifetime of Racist Practices
******************************
Donald Trump has often said he made his money "the old-fashioned way," and this is true, in that he prospered from racism.
Send Him Back?
Donald Drumpf’s Roots
************************
A New York Times article published 01 June 1927, related Donald Trump's father Fred Trump's role in a Ku Klux Klan brawl that pitted 1,000 klansmen against 100 New York City Police in Queens.  Though he wasn't officially charged, Fred Trump was one of seven klansmen arrested during the incident.  It probably wasn't very shocking at the time as America's racist practices were in full swing generations after Abe Lincoln freed the country's African-American slaves.  In fact, the mid-20's saw a peak in KKK activity.  Donald Trump would later deny his father's involvement in the KKK brawl in spite of the fact that it happened two decades before he was born.  Fred Trump's enthusiasm for racist practices never changed until he was forced to do so by the law.
Presidential Candidate Donald Trump joined his father's real estate company in New York in 1971, and only two years later, the company was served with a civil rights lawsuit that was filed against the Trump organization because it refused to rent to Black people.  The Urban League got wind of the racist rental policy and actually sent both Black and White people in to apply for apartments that belonged to complexes owned by the Trumps.  What they proved, was that Black people were denied rentals across the board, and only Whites were approved.  A Village Voice article by Wayne Barrett, published in 1979, blew the lid off the Trump organization's brewing pot of racist practices.
"Three doormen were told to discourage blacks who came seeking apartments when the manager was out, either by claiming no vacancies or hiking up the rents. A super said he was instructed to send black applicants to the central office but to accept white applications on site. Another rental agent said that Fred Trump had instructed him not to rent to blacks. Further, the agent said Trump wanted 'to decrease the number of black tenants' already in the development 'by encouraging them to locate housing elsewhere.'"
The article explains that Trump's reaction was to claim that the suit was a "nationwide drive to force owners of moderate and luxury apartments to rent to welfare recipients."
"'We are not going to be forced by anyone to put people...in our buildings to the detriment of tenants who have, for many years, lived in these buildings, raised families in them, and who plan to continue to live there. That would be reverse discrimination,' he said. 'The government is not going to experiment with our buildings to the detriment of ourselves and the thousands who live in them now.'"   
Indeed, Trump's wild, largely uninformed and unintelligent rants are the legacy of men who walked over the backs of other Americans to gain and secure their wealth.  This may be symbolic of the American capitalist way, but it falls short of any form of greatness or real human success.  As they say, the apple doesn't fall far from the tree.
"""

d = [d]

test = vectorizer.transform(d)
#print(df.label[0])
print(model.predict(test))
#print(df.label[9])