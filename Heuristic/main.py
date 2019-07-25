from textstat import lexicon_count, syllable_count, dale_chall_readability_score
__all__ = [lexicon_count, syllable_count, dale_chall_readability_score]


with open('BIS-2A_ 2019-07-17 12_10_transcript.txt', 'r') as myfile:
  data = myfile.read()

num_words = lexicon_count(data, removepunct=True)
num_syllables = syllable_count(data, lang='en_US')

# 4.9 or lower	average 4th-grade student or lower
# 5.0–5.9	average 5th or 6th-grade student
# 6.0–6.9	average 7th or 8th-grade student
# 7.0–7.9	average 9th or 10th-grade student
# 8.0–8.9	average 11th or 12th-grade student
# 9.0–9.9	average 13th to 15th-grade (college) student
grade_level = dale_chall_readability_score(data)


print(num_words)
print(num_syllables / num_words)
print(grade_level)
