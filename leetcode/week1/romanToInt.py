#tldr: Given a roman numeral, convert it to an integer.
#https://leetcode.com/problems/roman-to-integer/description/

def romanToInt(self, s):

    roman_values = {
        'I': 1,
        'V': 5,
        'X': 10,
        'L': 50,
        'C': 100,
        'D': 500,
        'M': 1000
    }
    
    total = 0
    
    for i in range(len(s)):
        if i + 1 < len(s) and roman_values[s[i]] < roman_values[s[i + 1]]:
            total -= roman_values[s[i]]
        else:
            total += roman_values[s[i]]
    
    return total

