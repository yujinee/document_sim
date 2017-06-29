import jamo
import re
import Levenshtein

def decompose(s):
    return jamo.j2hcj(jamo.h2j(s))

s = 'ㄱㄲㄳㄴㄵㄶㄷㄸㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅃㅄㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ'

print("length :", len(s))

for i in s:
    print(i, int(Levenshtein.distance(i, 'ㄱ') - 12593))
